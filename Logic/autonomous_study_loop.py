"""Chapter-wise Planning mission generation.

Planning deliberately operates at chapter scope. Fine-grained content
concepts remain useful to the learning system, but students receive one calm,
ordered roadmap made from the chapter's compact published learning units.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

from Logic.agent_event_bus import event_bus
from Logic.analytics_engine import get_user_analytics
from Logic.agents.coach_agent import get_or_create_coach
from services.catalog_service import resolve_catalog_chapter_units


class PlanningChapterNotFoundError(ValueError):
    """Raised when Planning cannot ground the requested chapter and syllabus."""

    def __init__(self, chapter: str, subject: str = "", class_level: str = ""):
        scope = " ".join(part for part in (class_level.strip(), subject.strip()) if part)
        suffix = f" for {scope}" if scope else ""
        super().__init__(
            f'Chapter "{chapter.strip()}" is not available{suffix}. '
            "Choose a chapter from Planning and try again."
        )


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _display_label(value: Any, fallback: str = "Selected chapter") -> str:
    label = str(value or "").strip()
    if not label:
        return fallback
    if "_" in label or "-" in label:
        return label.replace("_", " ").replace("-", " ").title()
    return label


def _normalize_mission_profile(
    current_knowledge: str = "some_idea",
    learning_goal: str = "deep_understanding",
    preferred_style: str = "examples_first",
    prerequisite_confidence: str = "medium",
    class_level: str = "",
) -> Dict[str, Any]:
    knowledge = (current_knowledge or "some_idea").strip().lower()
    if knowledge not in {"new", "some_idea", "know_basics"}:
        knowledge = "some_idea"

    goal = (learning_goal or "deep_understanding").strip().lower()
    if goal == "quick_revision":
        goal = "fast_track"
    elif goal not in {"deep_understanding", "exam", "fast_track"}:
        goal = "deep_understanding"

    style = (preferred_style or "examples_first").strip().lower()
    if style not in {"examples_first", "short_explanations", "conceptual_detail"}:
        style = "examples_first"

    confidence = (prerequisite_confidence or "medium").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    return {
        "current_knowledge": knowledge,
        "learning_goal": goal,
        "preferred_style": style,
        "prerequisite_confidence": confidence,
        "class_level": (class_level or "").strip(),
    }


def _is_fast_track(profile: Dict[str, Any]) -> bool:
    return profile["learning_goal"] == "fast_track"


def _needs_prerequisite_block(profile: Dict[str, Any], mastery_band: str) -> bool:
    return (
        profile["current_knowledge"] == "new"
        or profile["prerequisite_confidence"] == "low"
        or mastery_band in {"baseline", "critical"}
    )


def _resolve_chapter_scope(
    db,
    *,
    current_chapter: str,
    subject: str,
    class_level: str,
) -> Dict[str, Any]:
    """Resolve only the requested chapter; never substitute another one."""
    resolved = resolve_catalog_chapter_units(
        db,
        chapter_ref=current_chapter,
        subject=subject or None,
        class_level=class_level or None,
    )
    if resolved:
        return resolved

    raise PlanningChapterNotFoundError(current_chapter, subject, class_level)


def _chapter_mastery_signal(
    analytics: Dict[str, Any],
    units: Sequence[Dict[str, Any]],
) -> tuple[float, int]:
    unit_keys = {
        key
        for unit in units
        for key in (
            _normalize_key(unit.get("id")),
            _normalize_key(unit.get("label")),
            *[_normalize_key(value) for value in unit.get("concept_ids") or []],
        )
        if key
    }
    values: List[float] = []
    seen: set[str] = set()
    for collection_name in ("weak_areas", "topic_heatmap"):
        for item in analytics.get(collection_name) or []:
            key = _normalize_key(item.get("topic"))
            if not key or key not in unit_keys or key in seen:
                continue
            seen.add(key)
            values.append(float(item.get("accuracy") or item.get("value") or 0))
    if not values:
        return (0.0, 0)
    return (sum(values) / len(values), len(values))


def _mastery_band(accuracy: float, signal_count: int) -> str:
    if signal_count == 0:
        return "baseline"
    if accuracy < 40:
        return "critical"
    if accuracy < 60:
        return "weak"
    if accuracy < 80:
        return "building"
    return "strong"


def _mission_priority(mastery_band: str) -> str:
    if mastery_band in {"baseline", "critical", "weak"}:
        return "high"
    if mastery_band == "building":
        return "medium"
    return "stretch"


def _estimate_mission_budget(
    profile: Dict[str, Any],
    mastery_band: str,
    learning_unit_count: int = 1,
) -> int:
    unit_count = max(1, int(learning_unit_count or 1))
    if _is_fast_track(profile):
        per_unit = 12
    elif profile["learning_goal"] == "deep_understanding":
        per_unit = 20
    else:
        per_unit = 16

    total = per_unit * unit_count
    if _needs_prerequisite_block(profile, mastery_band):
        total += min(20, 4 * unit_count)
    if mastery_band == "strong" or profile["current_knowledge"] == "know_basics":
        total -= min(20, 2 * unit_count)
    return max(12, min(240, total))


def _allocate_unit_minutes(total_minutes: int, unit_count: int) -> List[int]:
    count = max(1, unit_count)
    base, remainder = divmod(total_minutes, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _style_instruction(profile: Dict[str, Any]) -> str:
    return {
        "examples_first": "Begin with one clear example, then connect it to the idea.",
        "short_explanations": "Use a short explanation, then recall it without looking.",
        "conceptual_detail": "Understand why the idea works and how it connects to the chapter.",
    }[profile["preferred_style"]]


def _goal_instruction(profile: Dict[str, Any]) -> str:
    return {
        "deep_understanding": "Explain the reason in your own words before moving ahead.",
        "exam": "Finish with one standard school-exam application.",
        "fast_track": "Cover the essential rule, one example, and one quick check.",
    }[profile["learning_goal"]]


def _unit_prerequisite_check(
    *,
    chapter_label: str,
    unit_label: str,
    previous_label: str,
    sequence: int,
    needs_repair: bool,
) -> Dict[str, str]:
    if sequence == 1:
        status = "repair_first" if needs_repair else "ready"
        question = (
            f"Before Step 1, what is one fact you already know about {unit_label} "
            f"in {chapter_label}?"
        )
        guidance = (
            f"Stay in Planning and break {unit_label} into the ideas named in its title. "
            "Write what each part means, mark the first part you cannot explain, and keep "
            "Step 1 open until you can answer one accurate sentence."
            if needs_repair
            else f"Answer in one sentence about {unit_label}. If it is unclear, mark the exact missing idea and keep Step 1 open before continuing."
        )
    else:
        status = "connect_previous"
        question = (
            f"Before Step {sequence}, how does Step {sequence - 1} ({previous_label}) "
            f"prepare you for {unit_label}?"
        )
        guidance = (
            f"Stay in Planning and write one connection between {previous_label} and {unit_label}. "
            f"If the link is missing, keep Step {sequence} open and retry after reviewing the Step {sequence - 1} completion check."
        )
    return {"status": status, "question": question, "guidance": guidance}


def _unit_completion_check(
    unit_label: str,
    profile: Dict[str, Any],
    sequence: int,
) -> Dict[str, str]:
    if profile["learning_goal"] == "exam":
        question = f"For Step {sequence}, can you explain {unit_label} and complete one standard exam-style application without help?"
        expected = "A clear explanation, the correct method, and a checked final answer."
    elif _is_fast_track(profile):
        question = f"For Step {sequence}, can you recall the essential rule and one example for {unit_label} without notes?"
        expected = "The central rule and one correct example in your own words."
    else:
        question = f"For Step {sequence}, can you explain why {unit_label} works and connect it to the chapter without notes?"
        expected = "A correct explanation plus one meaningful chapter connection."
    return {"question": question, "expected_outcome": expected}


def _build_chapter_study_plan(
    chapter_scope: Dict[str, Any],
    mastery_band: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    units = chapter_scope["units"]
    estimated_minutes = _estimate_mission_budget(profile, mastery_band, len(units))
    durations = _allocate_unit_minutes(estimated_minutes, len(units))
    needs_repair = _needs_prerequisite_block(profile, mastery_band)
    study_plan: List[Dict[str, Any]] = []
    previous_label = ""
    for index, (unit, duration) in enumerate(zip(units, durations), start=1):
        label = unit["label"]
        study_plan.append(
            {
                "sequence": index,
                "unit_id": unit["id"],
                "title": f"Step {index}: {label}",
                "duration": f"{duration} min",
                "detail": (
                    f"Work through {label} as Step {index} of {chapter_scope['chapter_label']}. "
                    f"Use its in-page checkpoint before continuing. "
                    f"{_style_instruction(profile)} {_goal_instruction(profile)}"
                ),
                "focus": f"Step {index} · {label}",
                "prerequisite_check": _unit_prerequisite_check(
                    chapter_label=chapter_scope["chapter_label"],
                    unit_label=label,
                    previous_label=previous_label,
                    sequence=index,
                    needs_repair=needs_repair,
                ),
                "completion_check": _unit_completion_check(label, profile, index),
            }
        )
        previous_label = label
    return {
        "estimated_minutes": estimated_minutes,
        "study_plan": study_plan,
        "high_priority_concepts": [unit["label"] for unit in units],
    }


def _build_mission_plan(
    chapter_scope: Dict[str, Any],
    profile: Dict[str, Any],
    mastery_band: str,
) -> Dict[str, Any]:
    chapter = chapter_scope["chapter_label"]
    unit_count = len(chapter_scope["units"])
    if mastery_band == "baseline":
        why = "There is no reliable chapter mastery signal yet, so the route starts gently and checks understanding as you go."
    elif mastery_band in {"critical", "weak"}:
        why = "Earlier learning signals show that a guided sequence with small repairs will be more reliable than rushing."
    elif mastery_band == "building":
        why = "Known material can move faster while each remaining chapter connection is checked."
    else:
        why = "The chapter looks familiar, so the route emphasizes application, accuracy, and confident completion."
    return {
        "primary_agent": "mission_planner",
        "mode": "fast_track_mission" if _is_fast_track(profile) else "adaptive_mission",
        "difficulty": "easy" if _needs_prerequisite_block(profile, mastery_band) else "medium" if mastery_band != "strong" else "hard",
        "objective": f"Complete {chapter} comfortably in {unit_count} ordered learning steps.",
        "why": why,
        "steps": [
            "Begin with Step 1 and complete its short readiness check.",
            "Learn, practise, and check each step in order without leaving Planning.",
            "Finish the chapter confidence check before marking the route complete.",
        ],
        "next_actions": [
            "Start Step 1 in this Planning roadmap.",
            "Use the in-step repair only when a readiness check feels difficult.",
            "Continue to the next step after the completion check passes.",
        ],
    }


def _build_prerequisite_check(study_plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_step = study_plan[0]
    check = first_step["prerequisite_check"]
    return {
        "status": check["status"],
        "question": check["question"],
        "action": (
            f"{check['guidance']} You do not need to leave Planning."
            if check["status"] == "repair_first"
            else f"Answer briefly, then continue directly into {first_step['title']} here in Planning."
        ),
        "unit_id": first_step["unit_id"],
    }


def _build_chapter_diagnostic(chapter_scope: Dict[str, Any]) -> Dict[str, Any]:
    chapter = chapter_scope["chapter_label"]
    first_unit = chapter_scope["units"][0]["label"]
    correct = f"I can explain {first_unit} in my own words and give one correct example."
    return {
        "id": f"mission_{uuid.uuid4().hex[:8]}",
        "question": f"Which statement best shows that you are ready to continue the {chapter} roadmap?",
        "options": [
            correct,
            f"I recognise the words in {first_unit}, but cannot explain them yet.",
            "I will skip the checks and only read the final summary.",
            "I will memorise the headings without practising an example.",
        ],
        "correct": correct,
        "explanation": (
            f"Being able to explain {first_unit} and use an example is a reliable first completion signal. "
            "If you are not there yet, repeat Step 1 inside this plan and retry."
        ),
    }


def _build_adaptive_roadmap(chapter_scope: Dict[str, Any]) -> List[Dict[str, str]]:
    chapter = chapter_scope["chapter_label"]
    return [
        {
            "condition": "If a step check is clear",
            "next_step": "Continue to the next numbered step in this chapter plan.",
            "mentor_action": "Keep the pace comfortable and preserve the chapter order.",
        },
        {
            "condition": "If a step check is incorrect",
            "next_step": "Use that step's short repair, review its example, and retry the check here.",
            "mentor_action": "Repair only the missing idea instead of restarting the chapter.",
        },
        {
            "condition": "If the student feels unsure",
            "next_step": f"Pause the {chapter} roadmap at the current step and write one simple explanation before continuing.",
            "mentor_action": "Build confidence inside Planning without an unexpected page change.",
        },
    ]


def _build_success_criteria(chapter_scope: Dict[str, Any]) -> List[str]:
    chapter = chapter_scope["chapter_label"]
    return [
        "Complete every numbered learning step and its in-page check.",
        "Repair only the step that is unclear; do not restart the whole route.",
        f"Finish {chapter} by explaining the learning steps in order and completing one final application.",
    ]


def _build_agent_sequence(plan: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"agent": "Supervisor Orchestrator", "role": "diagnose", "status": "complete", "detail": "Reads chapter context and existing learning signals."},
        {"agent": "Personal Coach", "role": "plan", "status": "complete", "detail": "Turns the complete chapter into a comfortable ordered route."},
        {"agent": "Adaptive Tutor", "role": "execute", "status": "complete", "detail": "Adds a distinct readiness and completion check to each learning step."},
        {"agent": "Subject Reviewer", "role": "verify", "status": "complete", "detail": "Preserves chapter coverage and a reliable finish condition."},
    ]


def _build_checkpoints(plan: Dict[str, Any], success_criteria: List[str]) -> List[Dict[str, str]]:
    return [
        {"title": "Chapter selected", "owner": "Supervisor", "status": "complete", "detail": plan["objective"]},
        {"title": "Roadmap ready", "owner": "mission_planner", "status": "complete", "detail": plan["steps"][0]},
        {"title": "Student progress", "owner": "student", "status": "pending", "detail": success_criteria[0]},
        {"title": "Chapter completion", "owner": "student", "status": "pending", "detail": success_criteria[2]},
    ]


def _build_mission_contract(
    *,
    plan: Dict[str, Any],
    chapter_scope: Dict[str, Any],
    analytics: Dict[str, Any],
    profile: Dict[str, Any],
    mastery_band: str,
    accuracy: float,
    signal_count: int,
    estimated_minutes: int,
) -> Dict[str, Any]:
    summary = analytics.get("summary") or {}
    success_criteria = _build_success_criteria(chapter_scope)
    return {
        "mission_type": "chapter_plan",
        "priority": _mission_priority(mastery_band),
        "mastery_band": mastery_band,
        "estimated_minutes": estimated_minutes,
        "student_state": {
            "plan_scope": "chapter",
            "chapter_accuracy": accuracy,
            "chapter_signal_count": signal_count,
            "learning_unit_count": len(chapter_scope["units"]),
            "average_accuracy": float(summary.get("avg_accuracy") or 0),
            "streak": int(summary.get("streak") or 0),
            "current_knowledge": profile["current_knowledge"],
            "learning_goal": profile["learning_goal"],
            "preferred_style": profile["preferred_style"],
            "prerequisite_confidence": profile["prerequisite_confidence"],
        },
        "agent_sequence": _build_agent_sequence(plan),
        "success_criteria": success_criteria,
        "checkpoints": _build_checkpoints(plan, success_criteria),
        "completion_report": {
            "status": "awaiting_chapter_progress",
            "measure": success_criteria[0],
            "next_memory_event": "chapter_plan_completed",
            "coach_follow_up": plan["next_actions"][0],
            "final_report_sections": ["Steps completed", "Repairs used", "Final confidence", "Next chapter action"],
        },
    }


def _build_fast_revision_strategy(chapter_scope: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    chapter = chapter_scope["chapter_label"]
    if _is_fast_track(profile):
        return [
            f"Move through the numbered {chapter} steps using only the essential rule and example.",
            "Complete each short check before advancing.",
            "Repeat only the step that fails; keep completed steps complete.",
        ]
    return [
        f"After each {chapter} step, compress the idea into one recall line.",
        "Connect each new step to the previous one.",
        "Use the final pass to recall the complete chapter order.",
    ]


def _build_weakness_detection_points(chapter_scope: Dict[str, Any]) -> List[str]:
    unit_labels = [unit["label"] for unit in chapter_scope["units"]]
    points = [f"The student cannot explain {label} in one clear sentence." for label in unit_labels[:3]]
    points.append("The student completes a step but cannot connect it to the next one.")
    return points


def _build_final_confidence_check(chapter_scope: Dict[str, Any]) -> List[str]:
    chapter = chapter_scope["chapter_label"]
    return [
        f"Can I explain the {chapter} learning steps in the correct order?",
        "Can I connect every step to at least one example or application?",
        "Can I complete one final question without notes and explain my method?",
    ]


def _build_fast_track_strategy(chapter_scope: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    chapter = chapter_scope["chapter_label"]
    if not _is_fast_track(profile):
        return [
            f"Use the full {chapter} roadmap at a comfortable pace.",
            "Move faster only through steps whose checks are already clear.",
        ]
    return [
        f"Use one essential explanation and one example for every {chapter} step.",
        "Skip repeated reading after a completion check passes.",
        "Finish with one chapter-wide recall pass.",
    ]


def run_autonomous_study_loop(
    db,
    user_id: str,
    current_chapter: str,
    subject: str = "Chemistry",
    current_knowledge: str = "some_idea",
    learning_goal: str = "deep_understanding",
    preferred_style: str = "examples_first",
    prerequisite_confidence: str = "medium",
    class_level: str = "",
) -> Dict[str, Any]:
    started_at = time.time()
    mission_id = f"mission_{uuid.uuid4().hex[:12]}"
    session_id = f"autonomous-{user_id}-{mission_id}"

    event_bus.emit(
        "orchestrator",
        "task_start",
        {"task": f"Chapter planning mission {mission_id}", "message": "Building a complete, comfortable chapter roadmap.", "mission_id": mission_id, "user_id": user_id},
        session_id=session_id,
    )

    analytics = get_user_analytics(db, user_id)
    profile = _normalize_mission_profile(
        current_knowledge=current_knowledge,
        learning_goal=learning_goal,
        preferred_style=preferred_style,
        prerequisite_confidence=prerequisite_confidence,
        class_level=class_level,
    )
    chapter_scope = _resolve_chapter_scope(
        db,
        current_chapter=current_chapter,
        subject=subject,
        class_level=class_level,
    )
    accuracy, signal_count = _chapter_mastery_signal(analytics, chapter_scope["units"])
    mastery_band = _mastery_band(accuracy, signal_count)
    plan = _build_mission_plan(chapter_scope, profile, mastery_band)
    optimized_plan = _build_chapter_study_plan(chapter_scope, mastery_band, profile)
    contract = _build_mission_contract(
        plan=plan,
        chapter_scope=chapter_scope,
        analytics=analytics,
        profile=profile,
        mastery_band=mastery_band,
        accuracy=accuracy,
        signal_count=signal_count,
        estimated_minutes=optimized_plan["estimated_minutes"],
    )

    event_bus.emit(
        "orchestrator",
        "step",
        {"step": "chapter_plan", "message": plan["objective"], "mission_id": mission_id, "target_chapter": chapter_scope["chapter_slug"], "primary_agent": plan["primary_agent"]},
        session_id=session_id,
    )

    study_plan = optimized_plan["study_plan"]
    prerequisite_check = _build_prerequisite_check(study_plan)
    diagnostic_question = _build_chapter_diagnostic(chapter_scope)
    adaptive_roadmap = _build_adaptive_roadmap(chapter_scope)
    high_priority_concepts = optimized_plan["high_priority_concepts"]
    fast_revision_strategy = _build_fast_revision_strategy(chapter_scope, profile)
    weakness_detection_points = _build_weakness_detection_points(chapter_scope)
    final_confidence_check = _build_final_confidence_check(chapter_scope)
    fast_track_strategy = _build_fast_track_strategy(chapter_scope, profile)
    chapter_label = chapter_scope["chapter_label"]
    plan_lines = [
        f"Chapter Goal: Complete {chapter_label} one clear step at a time.",
        f"Estimated total: {contract['estimated_minutes']} minutes",
        "",
        "Chapter Roadmap:",
        *[f"- {item['title']} ({item['duration']}): {item['detail']}" for item in study_plan],
        "",
        "How to progress:",
        "- Answer each readiness check inside Planning.",
        "- Use the short repair only for the step that feels unclear.",
        "- Continue after the completion check passes.",
    ]
    result = {
        "type": "chapter_plan",
        "answer": "\n".join(plan_lines),
        "data": {
            "text": "\n".join(plan_lines),
            "questions": [diagnostic_question],
            "study_plan": study_plan,
            "adaptive_roadmap": adaptive_roadmap,
            "prerequisite_check": prerequisite_check,
            "high_priority_concepts": high_priority_concepts,
            "fast_revision_strategy": fast_revision_strategy,
            "weakness_detection_points": weakness_detection_points,
            "final_confidence_check": final_confidence_check,
            "fast_track_strategy": fast_track_strategy,
        },
        "metadata": {
            "agent": "chapter_planner",
            "mission_model": "chapter_completion_roadmap",
            "personalization": "step_checks_adapt_the_pace_without_page_redirects",
            "plan_scope": "chapter",
            "profile": profile,
        },
    }
    latency_ms = round((time.time() - started_at) * 1000)

    coach = get_or_create_coach(db, user_id)
    coach.next_best_action = plan["next_actions"][0]
    coach.daily_strategy = plan["objective"]
    coach.last_recommendation = {
        "mission_id": mission_id,
        "objective": plan["objective"],
        "target_chapter": chapter_scope["chapter_slug"],
        "primary_agent": plan["primary_agent"],
        "mission_type": contract["mission_type"],
        "mastery_band": contract["mastery_band"],
        "priority": contract["priority"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    coach.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    event_bus.emit(
        "orchestrator",
        "task_complete",
        {"status": "success", "message": "The complete chapter roadmap is ready.", "mission_id": mission_id, "latency_ms": latency_ms},
        session_id=session_id,
    )

    return {
        "mission_id": mission_id,
        "status": "ready",
        "subject": chapter_scope["subject"],
        "chapter": chapter_label,
        "plan_scope": "chapter",
        "learning_unit_count": len(chapter_scope["units"]),
        # Compatibility response field: older saved clients still read this
        # key, but it now carries the chapter label rather than a topic target.
        "target_topic": chapter_label,
        "target_source": chapter_scope["source"],
        "mission_type": contract["mission_type"],
        "priority": contract["priority"],
        "mastery_band": contract["mastery_band"],
        "estimated_minutes": contract["estimated_minutes"],
        "mission_goal": f"Complete {chapter_label} through a comfortable, ordered chapter route.",
        "prerequisite_check": prerequisite_check,
        "high_priority_concepts": high_priority_concepts,
        "fast_revision_strategy": fast_revision_strategy,
        "weakness_detection_points": weakness_detection_points,
        "final_confidence_check": final_confidence_check,
        "fast_track_strategy": fast_track_strategy,
        "primary_agent": plan["primary_agent"],
        "mode": plan["mode"],
        "difficulty": plan["difficulty"],
        "objective": plan["objective"],
        "why": plan["why"],
        "steps": plan["steps"],
        "next_actions": plan["next_actions"],
        "success_criteria": contract["success_criteria"],
        "study_plan": study_plan,
        "diagnostic_question": diagnostic_question,
        "adaptive_roadmap": adaptive_roadmap,
        "agent_sequence": contract["agent_sequence"],
        "checkpoints": contract["checkpoints"],
        "student_state": contract["student_state"],
        "completion_report": contract["completion_report"],
        "result": result,
        "analytics_summary": analytics.get("summary", {}),
        "latency_ms": latency_ms,
    }
