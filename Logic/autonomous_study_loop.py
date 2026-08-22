"""Fast, chapter-grounded Planning briefs for school students.

Planning is a decision aid, not another place to study. A request resolves one
approved chapter, asks the configured model to rank only its published learning
units, validates the complete response, and falls back to a deterministic
metadata/analytics ranking whenever the model is unavailable or unsafe.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Dict, List, Sequence

from Logic.agent_event_bus import event_bus
from Logic.agents.coach_agent import get_or_create_coach
from Logic.analytics_engine import get_user_analytics
from Logic.coach.model_gateway import model_gateway
from Logic.planning.curriculum_registry import resolve_planning_curriculum
from Logic.planning.recommendation_engine import build_planning_roadmap
from services.catalog_service import resolve_catalog_chapter_units
from services.planning_progress_service import planning_learning_states


logger = logging.getLogger("ai_educator.planning")
RANK_SCORE_TOLERANCE = 0.75


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


def _analytics_for_units(
    analytics: Dict[str, Any],
    units: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Attach a signal only when analytics exactly matches a published unit."""
    unit_keys: Dict[str, set[str]] = {}
    for unit in units:
        keys = {
            _normalize_key(unit.get("id")),
            _normalize_key(unit.get("label")),
            *[_normalize_key(value) for value in unit.get("concept_ids") or []],
        }
        unit_keys[str(unit["id"])] = {key for key in keys if key}

    samples: Dict[str, List[float]] = {str(unit["id"]): [] for unit in units}
    seen: set[tuple[str, str]] = set()
    for collection_name in ("weak_areas", "topic_heatmap"):
        for item in analytics.get(collection_name) or []:
            topic_key = _normalize_key(item.get("topic"))
            if not topic_key:
                continue
            for unit_id, keys in unit_keys.items():
                marker = (unit_id, topic_key)
                if topic_key not in keys or marker in seen:
                    continue
                seen.add(marker)
                try:
                    accuracy = float(item.get("accuracy") or item.get("value") or 0)
                except (TypeError, ValueError):
                    continue
                samples[unit_id].append(max(0.0, min(100.0, accuracy)))

    return {
        unit_id: {
            "accuracy": round(sum(values) / len(values), 2),
            "signal_count": float(len(values)),
        }
        for unit_id, values in samples.items()
        if values
    }


def _chapter_mastery_signal(
    analytics: Dict[str, Any],
    units: Sequence[Dict[str, Any]],
) -> tuple[float, int]:
    signals = _analytics_for_units(analytics, units)
    if not signals:
        return (0.0, 0)
    accuracies = [signal["accuracy"] for signal in signals.values()]
    return (sum(accuracies) / len(accuracies), len(accuracies))


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


def _build_focus_area_scopes(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Consolidate adjacent units into at most five visible, lossless areas."""
    if not units:
        return []
    target = min(5, len(units))
    base, remainder = divmod(len(units), target)
    sizes = [base + (1 if index < remainder else 0) for index in range(target)]
    areas: List[Dict[str, Any]] = []
    offset = 0
    for size in sizes:
        members = list(units[offset : offset + size])
        offset += size
        unit_ids = [str(member["id"]) for member in members]
        unit_titles = [str(member["label"]).strip() for member in members]

        area_candidates: List[Dict[str, Any]] = []
        for member_index, member in enumerate(members):
            source_candidates = member.get("subtopic_candidates") or [
                {
                    "title": title,
                    # Older catalog rows do not carry per-title aliases. The
                    # containing unit aliases still let learner analytics make
                    # that unit's representative visible.
                    "concept_ids": list(member.get("concept_ids") or []),
                    "focus_signals": member.get("focus_signals") or {},
                    "source_order": title_index,
                }
                for title_index, title in enumerate(
                    member.get("subtopics") or [member["label"]]
                )
            ]
            for title_index, raw_candidate in enumerate(source_candidates):
                title = " ".join(str(raw_candidate.get("title") or "").split())
                if not title:
                    continue
                candidate_id = (
                    f"{member['id']}::subtopic::"
                    f"{raw_candidate.get('source_order', title_index)}"
                )
                area_candidates.append(
                    {
                        "id": candidate_id,
                        "label": title,
                        "title": title,
                        "unit_id": str(member["id"]),
                        # Analytics historically uses a mix of concept IDs,
                        # unit IDs, and unit titles. Carry all three aliases so
                        # the visible representative always matches the signal
                        # that elevated its consolidated area.
                        "concept_ids": list(
                            dict.fromkeys(
                                [
                                    str(member["id"]),
                                    str(member["label"]),
                                    *(raw_candidate.get("concept_ids") or []),
                                ]
                            )
                        ),
                        "focus_signals": raw_candidate.get("focus_signals")
                        or member.get("focus_signals")
                        or {},
                        "source_order": len(area_candidates),
                        "unit_order": member_index,
                    }
                )
        if not area_candidates:
            area_candidates = [
                {
                    "id": f"{unit_ids[0]}::subtopic::0",
                    "label": unit_titles[0],
                    "title": unit_titles[0],
                    "unit_id": unit_ids[0],
                    "concept_ids": [],
                    "focus_signals": {},
                    "source_order": 0,
                    "unit_order": 0,
                }
            ]

        metadata_rows = [member.get("focus_signals") or {} for member in members]
        total_concepts = sum(int(row.get("concept_count") or 0) for row in metadata_rows)
        weighted_difficulty = sum(
            float(row.get("average_difficulty") or 1)
            * max(1, int(row.get("concept_count") or 1))
            for row in metadata_rows
        )
        difficulty_weight = sum(
            max(1, int(row.get("concept_count") or 1)) for row in metadata_rows
        )
        area_id = f"focus_{sha1('|'.join(unit_ids).encode('utf-8')).hexdigest()[:12]}"
        if len(unit_titles) == 1:
            display_title = _brief_label(unit_titles[0], 240)
        elif len(unit_titles) == 2:
            display_title = _brief_label(" & ".join(unit_titles), 240)
        else:
            display_title = _brief_label(
                f"{unit_titles[0]} + {len(unit_titles) - 1} connected areas",
                240,
            )
        areas.append(
            {
                "id": area_id,
                "label": display_title,
                "focus_area_id": area_id,
                "unit_ids": unit_ids,
                "unit_titles": unit_titles,
                "subtopics": [],
                "_subtopic_candidates": area_candidates,
                # Include source unit IDs in matching keys so existing learner
                # analytics still elevate the correct consolidated area.
                "concept_ids": list(
                    dict.fromkeys(
                        [
                            *unit_ids,
                            *unit_titles,
                            *[
                                str(concept_id)
                                for member in members
                                for concept_id in member.get("concept_ids") or []
                            ],
                        ]
                    )
                ),
                "focus_signals": {
                    "concept_count": total_concepts,
                    "importance_score": max(
                        (float(row.get("importance_score") or 0) for row in metadata_rows),
                        default=0,
                    ),
                    "exam_weightage_score": max(
                        (float(row.get("exam_weightage_score") or 0) for row in metadata_rows),
                        default=0,
                    ),
                    "average_difficulty": round(
                        weighted_difficulty / max(1, difficulty_weight), 2
                    ),
                },
            }
        )
    return _prioritize_focus_area_subtopics(areas, {})


def _focus_score(unit: Dict[str, Any], signal: Dict[str, float] | None) -> float:
    metadata = unit.get("focus_signals") or {}
    score = (
        float(metadata.get("importance_score") or 0) * 1.6
        + float(metadata.get("exam_weightage_score") or 0) * 1.2
        + float(metadata.get("average_difficulty") or 1) * 0.45
        + min(1.5, float(metadata.get("concept_count") or 1) * 0.25)
    )
    if signal:
        accuracy = float(signal.get("accuracy") or 0)
        if accuracy < 40:
            score += 4.0
        elif accuracy < 60:
            score += 2.5
        elif accuracy < 80:
            score += 1.0
        elif accuracy >= 90:
            score -= 1.0
    return score


def _prioritize_focus_area_subtopics(
    areas: Sequence[Dict[str, Any]],
    analytics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Choose visible, exact titles that explain each area's priority.

    Consolidated areas can contain more than four source units. We retain full
    unit coverage in ``unit_ids``/``unit_titles`` while rendering at most four
    approved subtopics. Learner weaknesses win first, followed by grounded
    syllabus metadata; selected representatives are then restored to syllabus
    order for readability.
    """
    prioritized: List[Dict[str, Any]] = []
    for original in areas:
        area = dict(original)
        candidates = [dict(candidate) for candidate in area.get("_subtopic_candidates") or []]
        signals = _analytics_for_units(analytics, candidates)

        def priority(candidate: Dict[str, Any]) -> tuple[float, float, float, int]:
            signal = signals.get(str(candidate["id"]))
            accuracy = float(signal.get("accuracy") or 0) if signal else 100.0
            weak = 1.0 if signal and accuracy < 60 else 0.0
            weakness = 100.0 - accuracy if weak else 0.0
            return (
                weak,
                weakness,
                _focus_score(candidate, signal),
                -int(candidate.get("source_order") or 0),
            )

        best_by_unit: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            unit_id = str(candidate["unit_id"])
            current = best_by_unit.get(unit_id)
            if current is None or priority(candidate) > priority(current):
                best_by_unit[unit_id] = candidate

        representatives = list(best_by_unit.values())
        if len(representatives) > 4:
            selected = sorted(representatives, key=priority, reverse=True)[:4]
        else:
            selected = list(representatives)
            selected_ids = {str(candidate["id"]) for candidate in selected}
            remaining = [
                candidate
                for candidate in candidates
                if str(candidate["id"]) not in selected_ids
            ]
            selected.extend(
                sorted(remaining, key=priority, reverse=True)[: 4 - len(selected)]
            )
        selected.sort(key=lambda candidate: int(candidate.get("source_order") or 0))
        area["subtopics"] = [str(candidate["title"]) for candidate in selected]

        weak_candidates = [
            candidate
            for candidate in candidates
            if (signal := signals.get(str(candidate["id"])))
            and float(signal.get("accuracy") or 0) < 60
        ]
        driver = (
            max(weak_candidates, key=priority)
            if weak_candidates
            else max(candidates, key=priority)
        )
        driver_signals = driver.get("focus_signals") or {}
        driver_learner_signal = signals.get(str(driver["id"]))
        if driver_learner_signal and float(driver_learner_signal.get("accuracy") or 0) < 60:
            driver_kind = "learner_weakness"
        elif float(driver_signals.get("importance_score") or 0) >= 3:
            driver_kind = "syllabus_importance"
        elif float(driver_signals.get("exam_weightage_score") or 0) >= 3:
            driver_kind = "exam_weightage"
        else:
            driver_kind = "general"
        area["_priority_driver"] = {
            "title": str(driver["title"]),
            "kind": driver_kind,
        }
        prioritized.append(area)
    return prioritized


def _brief_label(value: Any, limit: int = 96) -> str:
    label = " ".join(str(value or "").split())
    return label if len(label) <= limit else f"{label[: limit - 1].rstrip()}…"


def _focus_reason(
    level: str,
    _signal: Dict[str, float] | None,
    unit: Dict[str, Any],
) -> str:
    driver = unit.get("_priority_driver") or {}
    driver_title = _brief_label(driver.get("title") or unit.get("label"))
    if driver.get("kind") == "learner_weakness":
        return f"Earlier learning signals show that {driver_title} needs a stronger pass."
    if driver.get("kind") == "syllabus_importance":
        return f"Published syllabus metadata marks {driver_title} as especially important."
    if driver.get("kind") == "exam_weightage":
        return f"Published syllabus metadata gives {driver_title} higher exam relevance."
    return {
        "high": "Give this area the strongest attention in your first chapter pass.",
        "medium": "Understand the main idea and connect it to the high-focus areas.",
        "light": "Keep this pass brief, but include it so chapter coverage stays complete.",
    }[level]


def _focus_guidance(level: str, title: str) -> str:
    label = _brief_label(title, 100)
    return {
        "high": f"Explain {label} in your own words, then recall or use it once without notes.",
        "medium": f"Learn the key idea in {label} and connect it to one example.",
        "light": f"Read {label} once and confirm that you can recall its central idea.",
    }[level]


def _focus_band_sequence(count: int) -> List[str]:
    if count <= 0:
        return []
    high_count = 1 if count <= 3 else max(1, math.ceil(count * 0.34))
    light_count = 0 if count < 3 else max(1, math.floor(count * 0.25))
    return [
        "high" if rank < high_count else "light" if rank >= count - light_count else "medium"
        for rank in range(count)
    ]


def _focus_ranking_context(
    chapter_scope: Dict[str, Any],
    analytics: Dict[str, Any],
) -> tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, float]],
    Dict[str, float],
    List[str],
]:
    areas = _prioritize_focus_area_subtopics(
        _build_focus_area_scopes(chapter_scope["units"]),
        analytics,
    )
    analytics_by_area = _analytics_for_units(analytics, areas)
    scored = [
        (index, area, _focus_score(area, analytics_by_area.get(str(area["id"]))))
        for index, area in enumerate(areas)
    ]
    ranked = sorted(scored, key=lambda row: (-row[2], row[0]))
    return (
        areas,
        analytics_by_area,
        {str(area["id"]): score for _index, area, score in scored},
        [str(area["id"]) for _index, area, _score in ranked],
    )


def _grounded_focus_brief(
    chapter_scope: Dict[str, Any],
    area_scopes: Sequence[Dict[str, Any]],
    analytics_by_area: Dict[str, Dict[str, float]],
    levels: Dict[str, str],
) -> Dict[str, Any]:
    focus_areas: List[Dict[str, Any]] = []
    for area_scope in area_scopes:
        area_id = str(area_scope["id"])
        title = str(area_scope["label"]).strip()
        level = levels[area_id]
        focus_areas.append(
            {
                "focus_area_id": area_scope["focus_area_id"],
                "unit_ids": list(area_scope["unit_ids"]),
                "unit_id": area_scope["unit_ids"][0],
                "unit_titles": list(area_scope["unit_titles"]),
                "title": title,
                "subtopics": list(area_scope["subtopics"]),
                "focus_level": level,
                "reason": _focus_reason(
                    level,
                    analytics_by_area.get(area_id),
                    area_scope,
                ),
                "guidance": _focus_guidance(level, title),
            }
        )
    return {
        "chapter_summary": (
            f"Use this focus map to complete {chapter_scope['chapter_label']} without turning Planning into another study session."
        ),
        "focus_areas": focus_areas,
        "guidance_steps": _fallback_guidance_steps(focus_areas),
        "completion_signal": (
            "Finish when you can explain every High-focus area, connect the Medium areas, and recall the Light areas."
        ),
    }


def _fallback_guidance_steps(focus_areas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ids_by_level = {
        level: [
            unit_id
            for area in focus_areas
            if area["focus_level"] == level
            for unit_id in area["unit_ids"]
        ]
        for level in ("high", "medium", "light")
    }
    steps: List[Dict[str, Any]] = []
    if ids_by_level["high"]:
        steps.append(
            {
                "title": "Start with deep focus",
                "instruction": "Learn the High-focus areas first and recall or use each one without notes.",
                "focus_unit_ids": ids_by_level["high"],
            }
        )
    if ids_by_level["medium"]:
        steps.append(
            {
                "title": "Build the chapter links",
                "instruction": "Connect the Medium-focus areas to what you have just learned.",
                "focus_unit_ids": ids_by_level["medium"],
            }
        )
    if ids_by_level["light"]:
        steps.append(
            {
                "title": "Make a light pass",
                "instruction": "Cover the Light-focus areas briefly so no syllabus unit is missed.",
                "focus_unit_ids": ids_by_level["light"],
            }
        )
    all_ids = [unit_id for area in focus_areas for unit_id in area["unit_ids"]]
    steps.append(
        {
            "title": "Close the chapter",
            "instruction": "Recall the full focus map in order and check one mixed recall or application.",
            "focus_unit_ids": all_ids,
        }
    )
    if len(steps) < 3:
        steps.insert(
            -1,
            {
                "title": "Connect the ideas",
                "instruction": "Explain how the chapter areas fit together before the final check.",
                "focus_unit_ids": all_ids,
            },
        )
    return [dict(step, sequence=index) for index, step in enumerate(steps[:5], start=1)]


def _deterministic_focus_brief(
    chapter_scope: Dict[str, Any],
    analytics: Dict[str, Any],
) -> Dict[str, Any]:
    areas, analytics_by_area, _scores, ranked_ids = _focus_ranking_context(
        chapter_scope,
        analytics,
    )
    levels = dict(zip(ranked_ids, _focus_band_sequence(len(ranked_ids))))
    return _grounded_focus_brief(chapter_scope, areas, analytics_by_area, levels)


def _extract_json_object(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _validate_model_focus_ranking(
    payload: Dict[str, Any],
    chapter_scope: Dict[str, Any],
    analytics: Dict[str, Any],
) -> Dict[str, str]:
    areas, _analytics_by_area, score_by_id, _deterministic_ids = _focus_ranking_context(
        chapter_scope,
        analytics,
    )
    expected_ids = {str(area["focus_area_id"]) for area in areas}
    if set(payload) != {"focus_ranking"}:
        raise ValueError("Model ranking contains unsupported content")
    raw_ranking = payload.get("focus_ranking")
    if not isinstance(raw_ranking, list) or len(raw_ranking) != len(areas):
        raise ValueError("Focus ranking must contain every approved area exactly once")

    ranked_ids: List[str] = []
    for item in raw_ranking:
        if not isinstance(item, str):
            raise ValueError("Model ranking contains unsupported content")
        area_id = item.strip()
        if area_id not in expected_ids or area_id in ranked_ids:
            raise ValueError("Model ranking is not grounded in the selected chapter")
        ranked_ids.append(area_id)
    if set(ranked_ids) != expected_ids:
        raise ValueError("Model ranking does not preserve complete chapter coverage")

    # The model may break close metadata ties, but it may not place a clearly
    # lower-scored area ahead of a stronger syllabus or learner signal.
    for earlier_index, earlier_id in enumerate(ranked_ids):
        for later_id in ranked_ids[earlier_index + 1 :]:
            if score_by_id[later_id] > score_by_id[earlier_id] + RANK_SCORE_TOLERANCE:
                raise ValueError("Model ranking contradicts grounded priority signals")
    return dict(zip(ranked_ids, _focus_band_sequence(len(ranked_ids))))


def _llm_focus_brief(
    chapter_scope: Dict[str, Any],
    analytics: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    focus_area_scopes = _prioritize_focus_area_subtopics(
        _build_focus_area_scopes(chapter_scope["units"]),
        analytics,
    )
    analytics_by_unit = _analytics_for_units(analytics, focus_area_scopes)
    grounded_areas = [
        {
            "focus_area_id": area["focus_area_id"],
            "unit_ids": area["unit_ids"],
            "unit_titles": area["unit_titles"],
            "title": area["label"],
            "subtopics": area["subtopics"],
            "metadata": area.get("focus_signals") or {},
            "learner_signal": analytics_by_unit.get(str(area["id"])),
        }
        for area in focus_area_scopes
    ]
    prompt = {
        "class": chapter_scope.get("class_level") or profile.get("class_level") or "",
        "subject": chapter_scope.get("subject") or "",
        "chapter": chapter_scope["chapter_label"],
        "approved_focus_areas": grounded_areas,
    }
    raw = model_gateway.complete(
        "profiler",
        [
            {
                "role": "system",
                "content": (
                    "Rank the approved chapter focus areas from most to least attention. "
                    "Use every focus_area_id exactly once. Return no student-facing prose or extra keys. "
                    "Use only supplied metadata and learner signals. Return only JSON: "
                    '{"focus_ranking":["most important exact id","next exact id"]}.'
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        complexity="fast",
        agent_name="mission_planner",
        task="build_chapter_focus_brief",
        student_visible=False,
        safety_tier="strict_source_grounding",
        temperature=0.1,
        max_tokens=400,
    )
    levels = _validate_model_focus_ranking(
        _extract_json_object(raw),
        chapter_scope,
        analytics,
    )
    return _grounded_focus_brief(
        chapter_scope,
        focus_area_scopes,
        analytics_by_unit,
        levels,
    )


def _build_focus_brief(
    chapter_scope: Dict[str, Any],
    analytics: Dict[str, Any],
    profile: Dict[str, Any],
) -> tuple[Dict[str, Any], str]:
    try:
        return _llm_focus_brief(chapter_scope, analytics, profile), "llm"
    except Exception as exc:  # noqa: BLE001 - deterministic brief must always remain available
        logger.warning("Planning focus model fell back to grounded ranking: %s", exc)
        return _deterministic_focus_brief(chapter_scope, analytics), "deterministic_fallback"


def _legacy_study_plan(brief: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Truthful response-only bridge during the focus-brief rollout.

    New clients ignore this structure. It retains fields required by the
    previously deployed validator without adding a visible time estimate or
    changing the canonical brief.
    """
    return [
        {
            "sequence": index,
            "unit_id": area["unit_id"],
            "unit_ids": area["unit_ids"],
            "focus_area_id": area["focus_area_id"],
            "title": area["title"],
            "duration": "Self-paced",
            "detail": area["guidance"],
            "focus": area["reason"],
            "focus_level": area["focus_level"],
            "prerequisite_check": {
                "status": "ready",
                "question": f"For focus area {index}, what will you focus on first in {area['title']}?",
                "guidance": "Use the focus level and approved subtopics shown in this brief.",
            },
            "completion_check": {
                "question": f"Can you recall the central idea in {area['title']}?",
                "expected_outcome": "A short explanation in your own words.",
            },
        }
        for index, area in enumerate(brief["focus_areas"], start=1)
    ]


def _v1_focus_brief_from_roadmap(roadmap: Dict[str, Any], chapter_label: str) -> Dict[str, Any]:
    """Derive rollout aliases from the canonical ordered roadmap.

    The compatibility grouping never mutates or reorders v2 learning units.
    Older clients receive at most five adjacent groups while v2 clients retain
    the complete NCERT sequence and metadata.
    """
    units = list(roadmap["learning_units"])
    target = min(5, len(units))
    base, remainder = divmod(len(units), target)
    sizes = [base + (1 if index < remainder else 0) for index in range(target)]
    groups: List[List[Dict[str, Any]]] = []
    offset = 0
    for size in sizes:
        groups.append(units[offset : offset + size])
        offset += size

    # The deployed v1 UI groups High, then Medium, then Light.  A neutral label
    # on every rollout group preserves the canonical array order without
    # falsely teaching that later NCERT units are less important.  Canonical
    # importance remains available only on the v2 learning units.
    levels = dict.fromkeys(range(len(groups)), "medium")

    focus_areas: List[Dict[str, Any]] = []
    for index, group in enumerate(groups):
        unit_ids = [str(unit["id"]) for unit in group]
        unit_titles = [str(unit["title"]) for unit in group]
        concepts = list(
            dict.fromkeys(
                str(concept.get("title") if isinstance(concept, dict) else concept)
                for unit in group
                for concept in unit.get("concepts") or []
                if str(concept.get("title") if isinstance(concept, dict) else concept).strip()
            )
        )[:4]
        area_id = f"roadmap_{sha1('|'.join(unit_ids).encode('utf-8')).hexdigest()[:12]}"
        focus_areas.append(
            {
                "focus_area_id": area_id,
                "unit_ids": unit_ids,
                "unit_id": unit_ids[0],
                "unit_titles": unit_titles,
                "title": " & ".join(unit_titles),
                "subtopics": concepts or unit_titles[:4],
                "focus_level": levels[index],
                "reason": (
                    f"These are NCERT steps {group[0]['order']}–{group[-1]['order']}; "
                    "this neutral rollout group preserves their learning order."
                ),
                "guidance": "Complete these learning units in their numbered NCERT order and use each mastery check.",
            }
        )

    guidance_steps = [
        {
            "sequence": index,
            "title": f"Follow NCERT steps {group[0]['order']}–{group[-1]['order']}",
            "instruction": (
                f"Begin with {group[0]['title']} and continue in order through {group[-1]['title']}."
            ),
            "focus_unit_ids": [str(unit["id"]) for unit in group],
        }
        for index, group in enumerate(groups, start=1)
    ]
    return {
        "chapter_summary": (
            f"Your next step and the complete NCERT-ordered roadmap for {chapter_label} are ready."
        ),
        "focus_areas": focus_areas,
        "guidance_steps": guidance_steps,
        "completion_signal": "Finish when every learning unit meets its specific mastery criteria.",
    }


def _registered_roadmap_response(
    *,
    db,
    user_id: str,
    mission_id: str,
    session_id: str,
    started_at: float,
    curriculum: Dict[str, Any],
    analytics: Dict[str, Any],
    profile: Dict[str, Any],
    study_time_today: str,
) -> Dict[str, Any]:
    persisted_states = planning_learning_states(
        db,
        user_id=user_id,
        curriculum_key=str(curriculum["curriculum_key"]),
        valid_unit_ids=[str(unit["id"]) for unit in curriculum["units"]],
    )
    roadmap = build_planning_roadmap(
        curriculum,
        study_time_today=study_time_today,
        analytics=analytics,
        persisted_states=persisted_states,
        profile=profile,
    )
    chapter_label = curriculum["chapter_title"]
    brief = _v1_focus_brief_from_roadmap(roadmap, chapter_label)
    study_plan = _legacy_study_plan(brief)
    next_action = roadmap["daily_route"]["items"][0]["activity"]
    objective = f"Take the next clear NCERT step in {chapter_label}."
    high_priority = [
        unit["title"]
        for unit in roadmap["learning_units"]
        if unit["importance"] in {"very_high", "high"}
    ]
    plan_text = "\n".join(
        [
            brief["chapter_summary"],
            f"Next: {roadmap['next_step']['title']} — {roadmap['next_step']['reason']}",
            *[
                f"{unit['order']}. {unit['title']}"
                for unit in roadmap["learning_units"]
            ],
        ]
    )
    result = {
        "type": "planning_roadmap_v2",
        "answer": plan_text,
        "data": {
            "text": plan_text,
            **roadmap,
            **brief,
            "study_plan": study_plan,
            "questions": [],
        },
        "metadata": {
            "agent": "mission_planner",
            "brief_version": "chapter_focus_v1",
            "roadmap_version": "planning_roadmap_v2",
            "generation_source": "deterministic_curriculum",
            "plan_scope": "chapter",
            "profile": profile,
        },
    }

    coach = get_or_create_coach(db, user_id)
    coach.next_best_action = next_action
    coach.daily_strategy = objective
    coach.last_recommendation = {
        "mission_id": mission_id,
        "objective": objective,
        "target_chapter": curriculum["chapter_slug"],
        "target_unit_id": roadmap["next_step"]["unit_id"],
        "mission_type": "planning_roadmap_v2",
        "study_time_today": study_time_today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    coach.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    latency_ms = round((time.time() - started_at) * 1000)
    event_bus.emit(
        "orchestrator",
        "task_complete",
        {
            "status": "success",
            "message": "The NCERT-ordered learning roadmap is ready.",
            "mission_id": mission_id,
            "latency_ms": latency_ms,
        },
        session_id=session_id,
    )
    return {
        **roadmap,
        "mission_id": mission_id,
        "status": "ready",
        "subject": curriculum["subject"],
        "chapter": chapter_label,
        "plan_scope": "chapter",
        "brief_version": "chapter_focus_v1",
        "chapter_summary": brief["chapter_summary"],
        "focus_areas": brief["focus_areas"],
        "guidance_steps": brief["guidance_steps"],
        "completion_signal": brief["completion_signal"],
        # This remains the visible v1 group count; v2 progress.total_units is
        # the canonical number of NCERT learning units.
        "learning_unit_count": len(brief["focus_areas"]),
        "target_topic": chapter_label,
        "target_source": "ncert_planning_manifest",
        "mission_type": "planning_roadmap_v2",
        "priority": "high",
        "mastery_band": "roadmap",
        "estimated_minutes": 0,
        "mission_goal": objective,
        "prerequisite_check": {},
        "high_priority_concepts": high_priority,
        "fast_revision_strategy": [step["instruction"] for step in brief["guidance_steps"]],
        "weakness_detection_points": [],
        "final_confidence_check": list(roadmap["completion_criteria"]),
        "fast_track_strategy": [],
        "primary_agent": "mission_planner",
        "mode": "planning_roadmap_v2",
        "difficulty": roadmap["learning_units"][0]["difficulty"],
        "objective": objective,
        "why": roadmap["next_step"]["reason"],
        "steps": [item["activity"] for item in roadmap["daily_route"]["items"]],
        "next_actions": [next_action],
        "success_criteria": list(roadmap["completion_criteria"]),
        "study_plan": study_plan,
        "diagnostic_question": {
            "id": f"legacy_{mission_id}",
            "question": f"What is your next NCERT step in {chapter_label}?",
            "options": [roadmap["next_step"]["title"], "Skip to the highest-importance unit"],
            "correct": roadmap["next_step"]["title"],
            "explanation": "Learning order follows NCERT; importance changes depth, not sequence.",
        },
        "adaptive_roadmap": [],
        "agent_sequence": [],
        "checkpoints": [],
        "student_state": {
            "plan_scope": "chapter",
            "roadmap_version": "planning_roadmap_v2",
            "current_knowledge": profile["current_knowledge"],
            "learning_goal": profile["learning_goal"],
            "preferred_style": profile["preferred_style"],
            "prerequisite_confidence": profile["prerequisite_confidence"],
        },
        "completion_report": {"status": "roadmap_ready"},
        "result": result,
        "analytics_summary": analytics.get("summary", {}),
        "latency_ms": latency_ms,
    }


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
    study_time_today: str = "no_limit",
) -> Dict[str, Any]:
    started_at = time.time()
    mission_id = f"mission_{uuid.uuid4().hex[:12]}"
    session_id = f"planning-{user_id}-{mission_id}"
    model_gateway.begin_turn(session_id)
    event_bus.emit(
        "orchestrator",
        "task_start",
        {
            "task": f"Chapter learning roadmap {mission_id}",
            "message": "Building a curriculum-grounded chapter roadmap.",
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
    curriculum = resolve_planning_curriculum(
        chapter_ref=current_chapter,
        subject=subject or None,
        class_level=class_level or None,
    )
    if curriculum:
        # End the analytics read transaction before deterministic assembly and
        # keep coach persistence in its own short transaction.
        db.commit()
        return _registered_roadmap_response(
            db=db,
            user_id=user_id,
            mission_id=mission_id,
            session_id=session_id,
            started_at=started_at,
            curriculum=curriculum,
            analytics=analytics,
            profile=profile,
            study_time_today=study_time_today,
        )
    chapter_scope = _resolve_chapter_scope(
        db,
        current_chapter=current_chapter,
        subject=subject,
        class_level=class_level,
    )
    accuracy, signal_count = _chapter_mastery_signal(analytics, chapter_scope["units"])
    mastery_band = _mastery_band(accuracy, signal_count)

    # Analytics and chapter scope are fully materialized dictionaries. End the
    # read transaction before external model I/O so provider latency/retries do
    # not retain a pooled database connection. Coach persistence below starts a
    # separate, short transaction after the brief is ready.
    db.commit()
    brief, generation_source = _build_focus_brief(chapter_scope, analytics, profile)
    chapter_label = chapter_scope["chapter_label"]
    unit_ids = [str(unit["id"]) for unit in chapter_scope["units"]]
    coverage = {
        "status": "complete",
        "included_unit_ids": unit_ids,
        "unit_count": len(unit_ids),
    }
    study_plan = _legacy_study_plan(brief)
    high_priority = [
        area["title"] for area in brief["focus_areas"] if area["focus_level"] == "high"
    ]
    plan_text = "\n".join(
        [
            brief["chapter_summary"],
            *[
                f"{area['focus_level'].title()}: {area['title']} — {area['guidance']}"
                for area in brief["focus_areas"]
            ],
            *[
                f"{step['sequence']}. {step['title']}: {step['instruction']}"
                for step in brief["guidance_steps"]
            ],
            brief["completion_signal"],
        ]
    )
    result = {
        "type": "chapter_focus_brief",
        "answer": plan_text,
        "data": {
            "text": plan_text,
            **brief,
            "coverage": coverage,
            "study_plan": study_plan,
            "questions": [],
        },
        "metadata": {
            "agent": "mission_planner",
            "brief_version": "chapter_focus_v1",
            "generation_source": generation_source,
            "plan_scope": "chapter",
            "profile": profile,
        },
    }

    objective = f"See what matters most in {chapter_label} and leave Planning with a clear route."
    next_action = brief["guidance_steps"][0]["instruction"]
    coach = get_or_create_coach(db, user_id)
    coach.next_best_action = next_action
    coach.daily_strategy = objective
    coach.last_recommendation = {
        "mission_id": mission_id,
        "objective": objective,
        "target_chapter": chapter_scope["chapter_slug"],
        "mission_type": "chapter_focus_brief",
        "mastery_band": mastery_band,
        "priority": _mission_priority(mastery_band),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    coach.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    latency_ms = round((time.time() - started_at) * 1000)
    event_bus.emit(
        "orchestrator",
        "task_complete",
        {
            "status": "success",
            "message": "The short chapter focus brief is ready.",
            "mission_id": mission_id,
            "latency_ms": latency_ms,
        },
        session_id=session_id,
    )
    return {
        "mission_id": mission_id,
        "status": "ready",
        "subject": chapter_scope["subject"],
        "chapter": chapter_label,
        "plan_scope": "chapter",
        "brief_version": "chapter_focus_v1",
        "chapter_summary": brief["chapter_summary"],
        "focus_areas": brief["focus_areas"],
        "guidance_steps": brief["guidance_steps"],
        "completion_signal": brief["completion_signal"],
        "coverage": coverage,
        # Rollout alias: the old client equates this with visible study_plan
        # entries. Canonical source coverage remains coverage.unit_count.
        "learning_unit_count": len(brief["focus_areas"]),
        # Response aliases remain safe for saved clients; requests still ignore topics.
        "target_topic": chapter_label,
        "target_source": chapter_scope["source"],
        "mission_type": "chapter_focus_brief",
        "priority": _mission_priority(mastery_band),
        "mastery_band": mastery_band,
        "estimated_minutes": 0,
        "mission_goal": objective,
        "prerequisite_check": {},
        "high_priority_concepts": high_priority,
        "fast_revision_strategy": [step["instruction"] for step in brief["guidance_steps"]],
        "weakness_detection_points": [],
        "final_confidence_check": [brief["completion_signal"]],
        "fast_track_strategy": [],
        "primary_agent": "mission_planner",
        "mode": "chapter_focus_brief",
        "difficulty": "easy",
        "objective": objective,
        "why": brief["chapter_summary"],
        "steps": [step["instruction"] for step in brief["guidance_steps"]],
        "next_actions": [next_action],
        "success_criteria": [brief["completion_signal"]],
        "study_plan": study_plan,
        "diagnostic_question": {
            "id": f"legacy_{mission_id}",
            "question": f"What should guide your first pass through {chapter_label}?",
            "options": [
                "Start with the High-focus areas.",
                "Treat every area as equally demanding.",
            ],
            "correct": "Start with the High-focus areas.",
            "explanation": "The focus map already ranks the complete published chapter for you.",
        },
        "adaptive_roadmap": [],
        "agent_sequence": [],
        "checkpoints": [],
        "student_state": {
            "plan_scope": "chapter",
            "chapter_accuracy": accuracy,
            "chapter_signal_count": signal_count,
            "learning_unit_count": len(brief["focus_areas"]),
            "source_unit_count": len(unit_ids),
            "current_knowledge": profile["current_knowledge"],
            "learning_goal": profile["learning_goal"],
            "preferred_style": profile["preferred_style"],
            "prerequisite_confidence": profile["prerequisite_confidence"],
        },
        "completion_report": {"status": "brief_ready"},
        "result": result,
        "analytics_summary": analytics.get("summary", {}),
        "latency_ms": latency_ms,
    }
