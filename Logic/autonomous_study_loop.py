# Logic/autonomous_study_loop.py

"""
Autonomous study execution loop.

This is the first production-safe slice of autonomy: pick the student's next
best study mission, dispatch the right specialist agent, and return a structured
mission object the frontend can render.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from Logic.agent_event_bus import event_bus
from Logic.analytics_engine import get_user_analytics
from Logic.agents.coach_agent import get_or_create_coach
from Logic.knowledge_graph import knowledge_graph


def _safe_topic(value: Optional[str]) -> str:
    topic = str(value or "").strip().lower()
    return topic or "matter_definition"


def _topic_accuracy_from_analytics(analytics: Dict[str, Any], topic: str) -> float:
    topic_key = topic.replace("_", " ").lower()
    for collection_name in ("weak_areas", "topic_heatmap"):
        for item in analytics.get(collection_name) or []:
            candidate = str(item.get("topic") or "").replace("_", " ").lower()
            if candidate == topic_key:
                return float(item.get("accuracy") or item.get("value") or 0)
    return 0.0


def _select_target_topic(
    analytics: Dict[str, Any],
    current_topic: Optional[str],
) -> Dict[str, Any]:
    if current_topic:
        topic = _safe_topic(current_topic)
        accuracy = _topic_accuracy_from_analytics(analytics, topic)
        return {
            "topic": topic,
            "accuracy": accuracy,
            "source": "selected_topic",
            "reason": (
                f"The student selected {topic.replace('_', ' ')}."
                if accuracy == 0
                else f"The student selected this topic and current mastery is about {accuracy}%."
            ),
        }

    weak_areas = analytics.get("weak_areas") or []
    if weak_areas:
        weakest = weak_areas[0]
        return {
            "topic": _safe_topic(weakest.get("topic")),
            "accuracy": float(weakest.get("accuracy") or 0),
            "source": "weak_area",
            "reason": f"Lowest active mastery signal: {weakest.get('accuracy', 0)}% accuracy.",
        }

    heatmap = analytics.get("topic_heatmap") or []
    if heatmap:
        sorted_topics = sorted(heatmap, key=lambda item: float(item.get("value") or 0))
        topic = sorted_topics[0]
        return {
            "topic": _safe_topic(topic.get("topic")),
            "accuracy": float(topic.get("value") or 0),
            "source": "topic_heatmap",
            "reason": f"Lowest available topic score: {topic.get('value', 0)}%.",
        }

    return {
        "topic": _safe_topic(current_topic),
        "accuracy": 0.0,
        "source": "current_context",
        "reason": "No weak-topic history yet, so the current study context is used.",
    }


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

    return {
        "current_knowledge": knowledge,
        "learning_goal": goal,
        "preferred_style": style,
        "prerequisite_confidence": (prerequisite_confidence or "medium").strip().lower(),
        "class_level": (class_level or "").strip(),
    }


def _topic_label(topic: str) -> str:
    return topic.replace("_", " ").title()


def _needs_prerequisite_block(profile: Dict[str, Any], mastery_band: str) -> bool:
    return (
        profile["current_knowledge"] == "new"
        or profile["prerequisite_confidence"] in {"low", "weak", "not_confident"}
        or mastery_band in {"baseline", "critical"}
    )


def _is_fast_track(profile: Dict[str, Any]) -> bool:
    return profile["learning_goal"] == "fast_track"


def _build_mission_plan(target: Dict[str, Any], analytics: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    topic = target["topic"]
    accuracy = float(target.get("accuracy") or 0)
    summary = analytics.get("summary") or {}
    total_topics = int(summary.get("total_topics") or 0)
    mastery_band = _mastery_band(accuracy, total_topics)
    topic_label = topic.replace("_", " ")
    fast_track = _is_fast_track(profile)

    if mastery_band == "baseline":
        why = "The coach needs one first signal before personalizing the roadmap."
    elif mastery_band in {"critical", "weak"}:
        why = f"{target['reason']} Start with the bottleneck first, then use a quick diagnostic check."
    elif mastery_band == "building":
        why = f"{target['reason']} Skip what is already known and move into application quickly."
    else:
        why = "The topic looks stronger, so the mission should focus on exam speed, traps, and confidence."

    return {
        "primary_agent": "mission_planner",
        "mode": "fast_track_mission" if fast_track else "adaptive_mission",
        "difficulty": "easy" if _needs_prerequisite_block(profile, mastery_band) else "medium" if mastery_band != "strong" else "hard",
        "objective": f"Complete {topic_label} through the fastest useful learning path.",
        "question": f"Run one checkpoint question on {topic_label} after the optimized roadmap.",
        "why": why,
        "steps": [
            "Answer the profile questions so the mission can remove unnecessary steps.",
            f"Follow the shortest high-value route for {topic_label}.",
            "Use the final checkpoint to decide whether to revise, practice, or move on.",
        ],
        "next_actions": [
            "Start the first timed block immediately.",
            "Pause only at checkpoint questions.",
            "Finish with the confidence check before leaving the topic.",
        ],
    }


def _find_topic_concept(topic: str) -> Optional[Dict[str, Any]]:
    topic_words = [word for word in topic.replace("_", " ").split() if len(word) > 2]
    for word in topic_words[:4]:
        matches = knowledge_graph.search_by_keyword(word, limit=1)
        if matches:
            return matches[0]
    return None


def _build_high_priority_concepts(topic: str, profile: Dict[str, Any]) -> List[str]:
    topic_label = topic.replace("_", " ")
    normalized = topic_label.lower()
    concept_map = {
        "alkanes": ["saturated hydrocarbons", "single covalent bonds", "general formula", "combustion pattern", "common naming traps"],
        "alkenes": ["carbon-carbon double bond", "unsaturation", "addition reactions", "general formula", "test with bromine water"],
        "alkynes": ["carbon-carbon triple bond", "unsaturation", "addition reactions", "acidic hydrogen basics", "naming rules"],
        "aromatics": ["benzene ring stability", "delocalized electrons", "substitution reactions", "resonance idea", "common examples"],
        "matter definition": ["mass", "space/volume", "particle nature", "examples vs non-examples", "definition wording"],
        "states of matter": ["particle arrangement", "inter-particle force", "movement", "change of state", "heating/cooling effect"],
        "properties of matter": ["mass", "volume", "density", "compressibility", "diffusion"],
        "mole concept": ["mole-mass conversion", "Avogadro number", "molar mass", "unit conversions", "standard numericals"],
    }
    for key, values in concept_map.items():
        if key in normalized:
            return values[:4] if _is_fast_track(profile) else values

    concept = _find_topic_concept(topic)
    if concept:
        key_points = [str(point) for point in concept.get("key_points") or [] if point]
        if key_points:
            return key_points[:4] if _is_fast_track(profile) else key_points[:6]

    return [
        f"core definition of {topic_label}",
        "most common exam wording",
        "one standard example",
        "one common mistake",
    ]


def _estimate_mission_budget(profile: Dict[str, Any], mastery_band: str) -> int:
    if _is_fast_track(profile):
        base = 35
    elif profile["learning_goal"] in {"deep_understanding", "conceptual"}:
        base = 75
    else:
        base = 55

    if _needs_prerequisite_block(profile, mastery_band):
        base += 20
    if mastery_band == "strong" or profile["current_knowledge"] in {"know_basics", "good", "strong"}:
        base -= 15

    return max(20, min(120, base))


def _format_duration(minutes: int) -> str:
    return f"{max(1, int(minutes))} min"


def _build_optimized_study_plan(
    topic: str,
    mastery_band: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    topic_label = topic.replace("_", " ")
    budget = _estimate_mission_budget(profile, mastery_band)
    fast_track = _is_fast_track(profile)
    needs_prereq = _needs_prerequisite_block(profile, mastery_band)
    high_priority = _build_high_priority_concepts(topic, profile)

    steps: List[Dict[str, str]] = []
    remaining = budget

    if needs_prereq:
        duration = max(6, round(budget * 0.18))
        remaining -= duration
        steps.append({
            "title": "Prerequisite repair",
            "duration": _format_duration(duration),
            "detail": f"Check the minimum basics needed for {topic_label}. Skip this only if the checkpoint feels easy.",
            "focus": "remove bottleneck",
        })

    core_duration = max(8, round(remaining * (0.28 if fast_track else 0.24)))
    remaining -= core_duration
    steps.append({
        "title": "Core idea",
        "duration": _format_duration(core_duration),
        "detail": f"Understand only the central idea of {topic_label}; avoid side theory until the main rule is clear.",
        "focus": "fast understanding",
    })

    priority_duration = max(8, round(remaining * (0.34 if fast_track else 0.30)))
    remaining -= priority_duration
    steps.append({
        "title": "High-yield concepts",
        "duration": _format_duration(priority_duration),
        "detail": "Cover the highest scoring points first: " + ", ".join(high_priority[:4]) + ".",
        "focus": "marks per minute",
    })

    application_duration = max(8, round(remaining * (0.48 if fast_track else 0.42)))
    remaining -= application_duration
    steps.append({
        "title": "Application sprint",
        "duration": _format_duration(application_duration),
        "detail": "Solve or mentally answer the most standard question types. Stop and repair only repeated mistakes.",
        "focus": "exam readiness",
    })

    checkpoint_duration = max(5, remaining)
    steps.append({
        "title": "Rapid checkpoint",
        "duration": _format_duration(checkpoint_duration),
        "detail": "Do one diagnostic question, one recall check, and one confidence rating before moving on.",
        "focus": "completion signal",
    })

    return {
        "estimated_minutes": sum(int(step["duration"].split()[0]) for step in steps),
        "study_plan": steps,
        "high_priority_concepts": high_priority,
    }


def _build_prerequisite_check(topic: str, profile: Dict[str, Any], mastery_band: str) -> Dict[str, Any]:
    topic_label = topic.replace("_", " ")
    needs_prereq = _needs_prerequisite_block(profile, mastery_band)
    return {
        "status": "repair_first" if needs_prereq else "skip_if_confident",
        "question": f"Before starting {topic_label}, can you explain the basic definition and one example without notes?",
        "action": (
            "Spend the first block repairing prerequisites before deeper learning."
            if needs_prereq
            else "Skip basic theory if you can answer this quickly; move to application."
        ),
    }


def _build_fast_revision_strategy(topic: str, profile: Dict[str, Any]) -> List[str]:
    topic_label = topic.replace("_", " ")
    if _is_fast_track(profile):
        return [
            f"Read only formulas/definitions and high-yield traps for {topic_label}.",
            "Do 3 standard questions before any deep theory.",
            "Convert every mistake into one short recall line.",
        ]
    return [
        f"After learning, compress {topic_label} into 5 bullet points.",
        "Revisit the weakest bullet after the diagnostic.",
        "Do one recall pass after a short break.",
    ]


def _build_weakness_detection_points(topic: str, profile: Dict[str, Any]) -> List[str]:
    topic_label = topic.replace("_", " ")
    return [
        f"Student cannot define {topic_label} in one clean sentence.",
        "Student gets the example right but misses the reason.",
        "Student chooses the correct option slowly or with low confidence.",
        "Student repeats a unit, formula, or keyword mistake twice.",
    ]


def _build_final_confidence_check(topic: str) -> List[str]:
    topic_label = topic.replace("_", " ")
    return [
        f"Can I explain {topic_label} in under 60 seconds?",
        "Can I solve one standard question without notes?",
        "Can I name one common trap and avoid it?",
    ]


def _build_fast_track_strategy(topic: str, profile: Dict[str, Any]) -> List[str]:
    if not _is_fast_track(profile):
        return [
            "Use full roadmap pacing unless the student reports time pressure.",
            "Move faster only after the prerequisite and core idea checkpoints pass.",
        ]
    topic_label = topic.replace("_", " ")
    return [
        f"Skip broad theory and start with the exam definition of {topic_label}.",
        "Prioritize high-yield concepts, common mistakes, and standard questions.",
        "Use quick recall, not long notes.",
    ]


def _build_single_diagnostic_question(topic: str) -> Dict[str, Any]:
    topic_label = topic.replace("_", " ")
    normalized_topic = topic.lower().replace("_", " ")
    topic_bank: Dict[str, List[Dict[str, Any]]] = {
        "alkane": [
            {
                "question": "In alkanes, what type of bond is present between carbon atoms?",
                "options": [
                    "Only single covalent bonds",
                    "At least one carbon-carbon double bond",
                    "At least one carbon-carbon triple bond",
                    "Only ionic bonds",
                ],
                "correct": "Only single covalent bonds",
                "explanation": "Alkanes are saturated hydrocarbons, so carbon atoms are connected by single covalent bonds.",
            },
            {
                "question": "Why are alkanes called saturated hydrocarbons?",
                "options": [
                    "They contain the maximum possible number of hydrogen atoms for single-bonded carbon chains",
                    "They dissolve completely in water",
                    "They always contain a benzene ring",
                    "They have at least one carbon-carbon triple bond",
                ],
                "correct": "They contain the maximum possible number of hydrogen atoms for single-bonded carbon chains",
                "explanation": "A saturated hydrocarbon has only single bonds, so each carbon can hold the maximum number of hydrogen atoms.",
            },
        ],
        "alkene": [
            {
                "question": "Which feature identifies an alkene?",
                "options": [
                    "At least one carbon-carbon double bond",
                    "Only carbon-carbon single bonds",
                    "A carbon-carbon triple bond",
                    "No hydrogen atoms",
                ],
                "correct": "At least one carbon-carbon double bond",
                "explanation": "Alkenes are unsaturated hydrocarbons because they contain at least one carbon-carbon double bond.",
            }
        ],
        "alkyne": [
            {
                "question": "Which bond is the key identifying feature of an alkyne?",
                "options": [
                    "A carbon-carbon triple bond",
                    "Only single covalent bonds",
                    "A benzene ring",
                    "A carbon-oxygen double bond",
                ],
                "correct": "A carbon-carbon triple bond",
                "explanation": "Alkynes are unsaturated hydrocarbons with at least one carbon-carbon triple bond.",
            }
        ],
        "aromatic": [
            {
                "question": "What makes an aromatic hydrocarbon different from a simple alkane?",
                "options": [
                    "It has a stable ring system with delocalized electrons",
                    "It has only single bonds in an open chain",
                    "It has no carbon atoms",
                    "It is always ionic",
                ],
                "correct": "It has a stable ring system with delocalized electrons",
                "explanation": "Aromatic hydrocarbons, such as benzene, are known for ring stability and delocalized electrons.",
            }
        ],
        "matter": [
            {
                "question": "Which statement correctly defines matter?",
                "options": [
                    "Anything that has mass and occupies space",
                    "Anything that produces light",
                    "Only things that are visible",
                    "Only solids and liquids",
                ],
                "correct": "Anything that has mass and occupies space",
                "explanation": "Matter is defined by two core properties: it has mass and it occupies space.",
            }
        ],
        "states": [
            {
                "question": "What mainly changes when matter changes from solid to liquid?",
                "options": [
                    "Particle arrangement and freedom of movement",
                    "The atoms stop existing",
                    "Mass becomes zero",
                    "The substance must become a gas first",
                ],
                "correct": "Particle arrangement and freedom of movement",
                "explanation": "During melting, particles gain freedom of movement while the substance remains matter.",
            }
        ],
    }

    for key, questions in topic_bank.items():
        if key in normalized_topic:
            question = questions[int(uuid.uuid4().hex[:2], 16) % len(questions)]
            return {
                "id": f"mission_{uuid.uuid4().hex[:8]}",
                "topic": topic,
                "subtopic": topic_label,
                "question": question["question"],
                "options": question["options"],
                "correct": question["correct"],
                "explanation": question["explanation"],
            }

    concept = _find_topic_concept(topic)
    definition = ""
    examples: List[str] = []
    mistakes: List[Any] = []
    if concept:
        definition = str(concept.get("definition") or "")
        examples = list(concept.get("examples") or [])[:2]
        mistakes = list(concept.get("common_mistakes") or [])[:2]

    correct_option = (
        definition
        if definition
        else f"It explains the main idea of {topic_label} using the correct property or rule."
    )
    example_text = f" For example: {examples[0]}." if examples else ""
    mistake_option = "It is only about memorizing words without understanding the reason."
    if mistakes:
        first = mistakes[0]
        if isinstance(first, dict) and first.get("mistake"):
            mistake_option = str(first["mistake"])
        elif first:
            mistake_option = str(first)

    options = [
        correct_option,
        mistake_option,
        f"It is unrelated to {topic_label} and can be skipped for exams.",
        "It is correct only when the answer is copied exactly from notes.",
    ]

    return {
        "id": f"mission_{uuid.uuid4().hex[:8]}",
        "topic": topic,
        "subtopic": topic_label,
        "question": f"Which statement correctly describes {topic_label}?",
        "options": options,
        "correct": correct_option,
        "explanation": (
            f"The best answer connects {topic_label} with its core meaning, property, or rule."
            f"{example_text} If this felt difficult, revise the definition and one example before more MCQs."
        ),
    }


def _build_adaptive_roadmap(topic: str, mastery_band: str) -> List[Dict[str, str]]:
    topic_label = topic.replace("_", " ")
    return [
        {
            "condition": "If the student answers correctly",
            "next_step": f"Move to two exam-style applications of {topic_label}.",
            "mentor_action": "Increase challenge slowly and check answer structure.",
        },
        {
            "condition": "If the student answers incorrectly",
            "next_step": f"Explain {topic_label} again using a simpler example and one common mistake.",
            "mentor_action": "Repair the exact misconception before giving more questions.",
        },
        {
            "condition": "If the student feels unsure",
            "next_step": "Ask the coach for a live example, then retry one similar question.",
            "mentor_action": "Build confidence before measuring speed.",
        },
    ]


def _mastery_band(accuracy: float, total_topics: int) -> str:
    if total_topics == 0:
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


def _estimate_minutes(mode: str, mastery_band: str) -> int:
    if mode == "adaptive_mission":
        return 15
    if mastery_band == "baseline":
        return 12
    if mode == "explain":
        return 18
    if mode == "exam":
        return 20
    if mode == "probable":
        return 25
    return 15


def _build_agent_sequence(plan: Dict[str, Any]) -> List[Dict[str, str]]:
    if plan["primary_agent"] in {"coach", "mission_planner"}:
        specialist = "Adaptive Tutor"
        specialist_action = "Creates the fastest personalized roadmap and checkpoint sequence"
    else:
        specialist = "Revision Specialist" if plan["primary_agent"] == "revision" else "Exam Generator"
        specialist_action = "Rebuilds the concept" if plan["primary_agent"] == "revision" else "Creates targeted practice"

    return [
        {
            "agent": "Supervisor Orchestrator",
            "role": "diagnose",
            "status": "complete",
            "detail": "Selects the highest-value mission from analytics and current context.",
        },
        {
            "agent": "Personal Coach",
            "role": "plan",
            "status": "complete",
            "detail": "Turns the mission into a student-friendly objective and next action.",
        },
        {
            "agent": specialist,
            "role": "execute",
            "status": "complete",
            "detail": specialist_action,
        },
        {
            "agent": "Subject Reviewer",
            "role": "verify",
            "status": "complete",
            "detail": "Checks usefulness, clarity, and exam readiness before delivery.",
        },
    ]


def _build_success_criteria(plan: Dict[str, Any], mastery_band: str) -> List[str]:
    if plan["mode"] in {"adaptive_mission", "fast_track_mission"}:
        return [
            "Complete every timed roadmap block without opening unrelated material.",
            "Pass the diagnostic and identify whether any miss came from concept clarity, memory, or application.",
            "Leave the topic only after the final confidence check reaches 80%+.",
        ]
    if mastery_band == "baseline":
        return [
            "Attempt every diagnostic question without notes.",
            "Save the result so the coach can create a first weak-topic signal.",
            "Review each wrong answer before starting the next mission.",
        ]
    if plan["mode"] == "explain":
        return [
            "Explain the concept back in your own words.",
            "Write one exam-ready answer without copying.",
            "Move to MCQs only after the common mistakes feel clear.",
        ]
    if plan["mode"] == "probable":
        return [
            "Write one probable answer in exam language.",
            "Check whether your answer includes the expected keywords.",
            "Convert missing points into a short revision note.",
        ]
    return [
        "Attempt the full practice set.",
        "Score at least 80% or review every wrong answer.",
        "Ask for mistake analysis if the same concept fails twice.",
    ]


def _build_checkpoints(plan: Dict[str, Any], success_criteria: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "title": "Mission selected",
            "owner": "Supervisor",
            "status": "complete",
            "detail": plan["objective"],
        },
        {
            "title": "Specialist output ready",
            "owner": plan["primary_agent"],
            "status": "complete",
            "detail": plan["steps"][0],
        },
        {
            "title": "Student action required",
            "owner": "student",
            "status": "pending",
            "detail": success_criteria[0],
        },
        {
            "title": "Memory update",
            "owner": "coach",
            "status": "pending",
            "detail": "Save the result or continue the chat so the coach can update memory.",
        },
    ]


def _build_mission_contract(
    plan: Dict[str, Any],
    target: Dict[str, Any],
    analytics: Dict[str, Any],
    profile: Dict[str, Any],
    estimated_minutes: int,
) -> Dict[str, Any]:
    summary = analytics.get("summary") or {}
    total_topics = int(summary.get("total_topics") or 0)
    accuracy = float(target.get("accuracy") or 0)
    mastery_band = _mastery_band(accuracy, total_topics)
    success_criteria = _build_success_criteria(plan, mastery_band)
    mission_type = "adaptive_diagnostic" if plan["mode"] == "adaptive_mission" else (
        "diagnostic" if mastery_band == "baseline" else plan["mode"]
    )

    return {
        "mission_type": mission_type,
        "priority": _mission_priority(mastery_band),
        "mastery_band": mastery_band,
        "estimated_minutes": estimated_minutes,
        "student_state": {
            "accuracy": accuracy,
            "source": target.get("source", "current_context"),
            "total_topics": total_topics,
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
            "status": "awaiting_diagnostic_answer" if plan["mode"] == "adaptive_mission" else "awaiting_student_action",
            "measure": success_criteria[0],
            "next_memory_event": "mission_result_saved",
            "coach_follow_up": plan["next_actions"][0],
            "final_report_sections": [
                "What you understood",
                "Weak point detected",
                "Confidence signal",
                "Next 80% roadmap",
            ],
        },
    }


def run_autonomous_study_loop(
    db,
    user_id: str,
    current_topic: Optional[str] = None,
    current_chapter: Optional[str] = None,
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
        {
            "task": f"Autonomous study mission {mission_id}",
            "message": "Selecting the next best study mission from analytics.",
            "mission_id": mission_id,
            "user_id": user_id,
        },
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
    target = _select_target_topic(analytics, current_topic)
    plan = _build_mission_plan(target, analytics, profile)
    summary = analytics.get("summary") or {}
    target_mastery_band = _mastery_band(
        float(target.get("accuracy") or 0),
        int(summary.get("total_topics") or 0),
    )
    optimized_plan = _build_optimized_study_plan(
        target["topic"],
        target_mastery_band,
        profile,
    )
    contract = _build_mission_contract(
        plan=plan,
        target=target,
        analytics=analytics,
        profile=profile,
        estimated_minutes=optimized_plan["estimated_minutes"],
    )

    event_bus.emit(
        "orchestrator",
        "step",
        {
            "step": "mission_plan",
            "message": plan["objective"],
            "mission_id": mission_id,
            "target_topic": target["topic"],
            "primary_agent": plan["primary_agent"],
        },
        session_id=session_id,
    )

    study_plan = optimized_plan["study_plan"]
    diagnostic_question = _build_single_diagnostic_question(target["topic"])
    adaptive_roadmap = _build_adaptive_roadmap(target["topic"], contract["mastery_band"])
    prerequisite_check = _build_prerequisite_check(target["topic"], profile, contract["mastery_band"])
    high_priority_concepts = optimized_plan["high_priority_concepts"]
    fast_revision_strategy = _build_fast_revision_strategy(target["topic"], profile)
    weakness_detection_points = _build_weakness_detection_points(target["topic"], profile)
    final_confidence_check = _build_final_confidence_check(target["topic"])
    fast_track_strategy = _build_fast_track_strategy(target["topic"], profile)
    plan_lines = [
        f"Mission Goal: Complete {target['topic'].replace('_', ' ')} in the shortest useful path.",
        f"Estimated Time: {contract['estimated_minutes']} minutes",
        "",
        "Optimized Study Roadmap:",
        *[f"- {item['title']} ({item['duration']}): {item['detail']}" for item in study_plan],
        "",
        "High Priority Concepts:",
        *[f"- {item}" for item in high_priority_concepts],
        "",
        "One-Question Diagnostic:",
        diagnostic_question["question"],
        *[f"- {option}" for option in diagnostic_question["options"]],
        "",
        "After You Answer:",
        "- Correct: move to exam application.",
        "- Wrong or unsure: rebuild the exact weak point with the coach.",
    ]
    result = {
        "type": "adaptive_mission",
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
            "agent": "adaptive_tutor",
            "mission_model": "profiled_adaptive_roadmap",
            "personalization": "roadmap_updates_after_student_answer",
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
        "target_topic": target["topic"],
        "primary_agent": plan["primary_agent"],
        "mission_type": contract["mission_type"],
        "mastery_band": contract["mastery_band"],
        "priority": contract["priority"],
        "generated_at": datetime.utcnow().isoformat(),
    }
    coach.updated_at = datetime.utcnow()
    db.commit()

    event_bus.emit(
        "orchestrator",
        "task_complete",
        {
            "status": "success",
            "message": f"Autonomous mission delivered by {plan['primary_agent']}.",
            "mission_id": mission_id,
            "latency_ms": latency_ms,
        },
        session_id=session_id,
    )

    return {
        "mission_id": mission_id,
        "status": "ready",
        "subject": subject,
        "chapter": current_chapter or "",
        "target_topic": target["topic"],
        "target_source": target["source"],
        "mission_type": contract["mission_type"],
        "priority": contract["priority"],
        "mastery_band": contract["mastery_band"],
        "estimated_minutes": contract["estimated_minutes"],
        "mission_goal": f"Complete {_topic_label(target['topic'])} through a focused, achievable learning route.",
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
