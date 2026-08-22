"""Deterministic NCERT-ordered roadmap and daily-route recommendations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .curriculum_registry import STATUS_VALUES


TIME_BUDGETS: Dict[str, Optional[int]] = {
    "15": 15,
    "30": 30,
    "60": 60,
    "120_plus": 120,
    "no_limit": None,
}


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _unit_aliases(unit: Mapping[str, Any]) -> set[str]:
    values: List[Any] = [
        unit.get("id"),
        unit.get("title"),
        unit.get("primary_topic_id"),
        *(unit.get("legacy_topic_ids") or []),
    ]
    for concept in unit.get("concepts") or []:
        if isinstance(concept, dict):
            values.extend((concept.get("id"), concept.get("title")))
        else:
            values.append(concept)
    return {normalized for value in values if (normalized := _normalized(value))}


def _analytics_attempts(item: Mapping[str, Any]) -> int:
    try:
        return max(0, int(item.get("attempts") or 0))
    except (TypeError, ValueError):
        return 0


def _analytics_accuracy(item: Mapping[str, Any]) -> float:
    value = item.get("accuracy") if item.get("accuracy") is not None else item.get("value")
    try:
        return max(0.0, min(100.0, float(value or 0)))
    except (TypeError, ValueError):
        return 0.0


def _mastery_attempt_requirement(unit: Mapping[str, Any]) -> int:
    """Use the curriculum-authored practice floor as the evidence threshold."""
    practice = unit.get("practice") if isinstance(unit.get("practice"), Mapping) else {}
    try:
        return max(1, int(practice.get("minimum_items") or 1))
    except (TypeError, ValueError):
        return 1


def derive_unit_statuses(
    curriculum: Mapping[str, Any],
    analytics: Mapping[str, Any],
    persisted_states: Sequence[Any] = (),
) -> Dict[str, str]:
    """Merge durable state with exact assessment signals without false mastery.

    Generic topic aliases can prove that practice happened or that review is
    needed.  Mastery is stricter: only scored evidence recorded against the
    unit's canonical ``primary_topic_id`` can advance it.  Exposure and elapsed
    time are never treated as mastery.
    """
    units = list(curriculum.get("units") or [])
    statuses = {str(unit["id"]): "not_started" for unit in units}
    units_by_id = {str(unit["id"]): unit for unit in units}

    persisted_by_unit: Dict[str, Any] = {
        str(getattr(row, "unit_id", "")): row
        for row in persisted_states
        if str(getattr(row, "unit_id", "")) in statuses
    }
    for unit_id, row in persisted_by_unit.items():
        raw_status = str(getattr(row, "status", "not_started") or "not_started")
        evidence_count = int(getattr(row, "evidence_count", 0) or 0)
        mastery_score = getattr(row, "mastery_score", None)
        required_evidence = _mastery_attempt_requirement(units_by_id[unit_id])
        if raw_status == "mastered":
            if (
                evidence_count >= required_evidence
                and mastery_score is not None
                and float(mastery_score) >= 80
            ):
                statuses[unit_id] = "mastered"
            elif evidence_count > 0:
                statuses[unit_id] = "practising"
            else:
                statuses[unit_id] = "learning"
        elif raw_status in STATUS_VALUES:
            statuses[unit_id] = raw_status

    analytics_rows = [
        item
        for collection in ("topic_heatmap", "weak_areas")
        for item in analytics.get(collection) or []
        if isinstance(item, dict) and _normalized(item.get("topic"))
    ]
    for unit in units:
        unit_id = str(unit["id"])
        if statuses[unit_id] == "mastered":
            continue
        aliases = _unit_aliases(unit)
        matching: Dict[str, Dict[str, Any]] = {}
        for item in analytics_rows:
            key = _normalized(item.get("topic"))
            if key in aliases:
                matching[key] = item
        if not matching:
            continue
        canonical_keys = {
            _normalized(unit.get("primary_topic_id")),
            _normalized(unit.get("title")),
        }
        canonical_evidence = [
            item
            for item in analytics_rows
            if _normalized(item.get("topic")) in canonical_keys
            and _analytics_attempts(item) > 0
        ]
        canonical_attempts = sum(_analytics_attempts(item) for item in canonical_evidence)
        canonical_accuracy = (
            sum(
                _analytics_accuracy(item) * _analytics_attempts(item)
                for item in canonical_evidence
            )
            / canonical_attempts
            if canonical_attempts
            else 0.0
        )
        if (
            canonical_attempts >= _mastery_attempt_requirement(unit)
            and canonical_accuracy >= 80
        ):
            statuses[unit_id] = "mastered"
            continue

        attempts = sum(max(1, _analytics_attempts(item)) for item in matching.values())
        weighted_accuracy = sum(
            _analytics_accuracy(item)
            * max(1, _analytics_attempts(item))
            for item in matching.values()
        ) / max(attempts, 1)
        statuses[unit_id] = "needs_review" if weighted_accuracy < 60 else "practising"
    return statuses


def derive_concept_progress(
    curriculum: Mapping[str, Any],
    analytics: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Map only exact concept-ID assessment evidence to child progress."""
    rows = [
        item
        for item in analytics.get("topic_heatmap") or []
        if isinstance(item, dict) and _normalized(item.get("topic"))
    ]
    by_topic: Dict[str, List[Mapping[str, Any]]] = {}
    for item in rows:
        by_topic.setdefault(_normalized(item.get("topic")), []).append(item)

    progress: Dict[str, Dict[str, Any]] = {}
    for unit in curriculum.get("units") or []:
        for concept in unit.get("concepts") or []:
            concept_id = str(concept["id"])
            evidence = by_topic.get(_normalized(concept_id), [])
            attempts = sum(_analytics_attempts(item) for item in evidence)
            accuracy = (
                sum(
                    _analytics_accuracy(item) * _analytics_attempts(item)
                    for item in evidence
                )
                / attempts
                if attempts
                else 0.0
            )
            if attempts == 0:
                status = "not_started"
            elif accuracy < 60:
                status = "needs_review"
            elif attempts >= 3 and accuracy >= 80:
                status = "mastered"
            else:
                status = "practising"
            progress[concept_id] = {
                "status": status,
                "evidence_count": attempts,
            }
    return progress


def _recommended_minutes(unit: Mapping[str, Any], profile: Mapping[str, Any]) -> int:
    estimate = unit["estimated_minutes"]
    minimum, maximum = int(estimate["min"]), int(estimate["max"])
    factors = [
        {"very_high": 7, "high": 4, "moderate": 2, "low": 0}[str(unit["importance"])],
        {"foundation": 0, "steady": 3, "challenging": 7}[str(unit["difficulty"])],
        {"very_high": 5, "high": 3, "moderate": 1, "low": 0}[str(unit["exam_relevance"])],
        {"very_high": 5, "high": 3, "moderate": 1, "low": 0}[
            str(unit["conceptual_importance"])
        ],
        {"overview": 0, "working": 3, "mastery": 6}[str(unit["depth"])],
    ]
    extra = sum(factors)
    if profile.get("learning_goal") == "fast_track" or profile.get("current_knowledge") == "know_basics":
        extra = round(extra * 0.4)
    elif profile.get("learning_goal") in {"deep_understanding", "exam"}:
        extra += 4
    if profile.get("current_knowledge") == "new":
        extra += 3
    target = min(maximum, minimum + max(0, extra))
    rounded = int(round(target / 5.0) * 5)
    return max(minimum, min(maximum, rounded))


def _route_reason(unit: Mapping[str, Any], status: str) -> str:
    if status == "needs_review":
        return "Earlier answers show this unit needs a calmer second pass before the route moves on."
    if status == "practising":
        return "You already know part of this unit, so today's step concentrates on supported practice."
    if status == "learning":
        return "This continues the unit you already started, without adding a new decision."
    difficulty_phrase = {
        "foundation": "builds an accessible foundation",
        "steady": "needs a steady understanding pass",
        "challenging": "is challenging and benefits from protected practice",
    }[str(unit["difficulty"])]
    priority_phrase = {
        "very_high": "It deserves very high chapter attention",
        "high": "It deserves strong chapter attention",
        "moderate": "It needs a balanced chapter pass",
        "low": "A short chapter pass is enough",
    }[str(unit["importance"])]
    importance_phrase = {
        "very_high": "It is central to the chapter",
        "high": "It supports several later ideas",
        "moderate": "It keeps the chapter connected",
        "low": "A concise pass preserves full coverage",
    }[str(unit["conceptual_importance"])]
    exam_phrase = {
        "very_high": " and appears often in exam-style work.",
        "high": " and has strong exam relevance.",
        "moderate": " and is useful in common questions.",
        "low": ".",
    }[str(unit["exam_relevance"])]
    return (
        f"This unit {difficulty_phrase}. {priority_phrase}. "
        f"{importance_phrase}{exam_phrase}"
    )


def _next_unit(
    units: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, str],
) -> tuple[Mapping[str, Any], List[str], bool]:
    mastered = {unit_id for unit_id, status in statuses.items() if status == "mastered"}
    for unit in units:
        if statuses[str(unit["id"])] == "mastered":
            continue
        unmet = [
            dependency
            for dependency in unit.get("prerequisite_unit_ids") or []
            if dependency not in mastered
        ]
        if unmet:
            earliest = next(candidate for candidate in units if candidate["id"] == unmet[0])
            return earliest, unmet, False
        return unit, [], False
    return units[-1], [], True


def _next_reason(
    unit: Mapping[str, Any],
    status: str,
    unmet_dependencies: Sequence[str],
    completed: bool,
    by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    if completed:
        return "You have met the recorded mastery evidence for every unit. Use this final unit for a short recall pass."
    if unmet_dependencies:
        names = ", ".join(str(by_id[unit_id]["title"]) for unit_id in unmet_dependencies)
        return f"Start with {names} as a short prerequisite refresh, then continue in NCERT order."
    if status == "needs_review":
        return "This is the earliest NCERT unit whose assessment signals need a stronger pass."
    if status == "practising":
        return "You have started this NCERT unit; the clearest next step is to practise it before moving on."
    if status == "learning":
        return "Continue the earliest NCERT unit you already started."
    return "This is the first unfinished unit in the locked NCERT learning sequence."


def _activity(
    unit: Mapping[str, Any],
    status: str,
    profile: Mapping[str, Any],
    *,
    completed: bool = False,
) -> str:
    title = str(unit["title"])
    if completed:
        return f"Recall {title} and try one mixed check"
    if status == "needs_review":
        return f"Review {title}, then retry its weakest skill"
    if status == "practising":
        return f"Practise {title} against its mastery criteria"
    if status == "learning":
        return f"Continue {title} and complete one understanding check"
    if profile.get("learning_goal") == "fast_track":
        return f"Recall the essential ideas in {title}, then try one quick check"
    if profile.get("learning_goal") == "exam":
        return f"Learn {title}, then complete one school-exam application"
    if profile.get("current_knowledge") == "know_basics":
        return f"Confirm the key ideas in {title}, then practise one application"
    if profile.get("current_knowledge") == "new":
        return f"Build the foundations of {title}, then explain the main idea once"
    return f"Learn {title}, explain why it works, and complete its first mastery check"


def _build_daily_route(
    units: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, str],
    time_preference: str,
    start_unit_id: str,
    *,
    chapter_completed: bool,
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    budget = TIME_BUDGETS[time_preference]
    start_index = next(index for index, unit in enumerate(units) if unit["id"] == start_unit_id)
    if chapter_completed:
        unit = units[-1]
        minutes = 10 if budget is None else min(budget, 10)
        return {
            "time_preference": time_preference,
            "budget_minutes": budget,
            "total_minutes": minutes,
            "items": [
                {
                    "unit_id": unit["id"],
                    "title": unit["title"],
                    "activity": _activity(
                        unit,
                        statuses[str(unit["id"])],
                        profile,
                        completed=True,
                    ),
                    "reason": "The chapter is mastered; this optional recall keeps the final link fresh.",
                    "minutes": minutes,
                    "scope": "partial",
                }
            ],
        }

    mastered_before = {
        str(unit["id"])
        for unit in units
        if statuses[str(unit["id"])] == "mastered"
    }
    remaining = budget
    items: List[Dict[str, Any]] = []
    available = set(mastered_before)
    for unit in units[start_index:]:
        unit_id = str(unit["id"])
        if statuses[unit_id] == "mastered":
            available.add(unit_id)
            continue
        if any(dependency not in available for dependency in unit.get("prerequisite_unit_ids") or []):
            break
        target = _recommended_minutes(unit, profile)
        minimum = int(unit["estimated_minutes"]["min"])
        if remaining is None:
            minutes = target
            scope = "complete"
        elif remaining >= minimum:
            minutes = min(target, remaining)
            scope = "complete"
        elif remaining >= 5:
            minutes = remaining
            scope = "partial"
        else:
            break
        items.append(
            {
                "unit_id": unit_id,
                "title": unit["title"],
                "activity": _activity(unit, statuses[unit_id], profile),
                "reason": _route_reason(unit, statuses[unit_id]),
                "minutes": int(minutes),
                "scope": scope,
            }
        )
        if remaining is None:
            break
        remaining -= int(minutes)
        if scope == "partial":
            break
        available.add(unit_id)
        if remaining < 5:
            break

    if not items:
        unit = units[start_index]
        minutes = min(int(unit["estimated_minutes"]["min"]), budget or 15)
        items.append(
            {
                "unit_id": unit["id"],
                "title": unit["title"],
                "activity": _activity(unit, statuses[str(unit["id"])], profile),
                "reason": _route_reason(unit, statuses[str(unit["id"])]),
                "minutes": max(5, minutes),
                "scope": "partial",
            }
        )
    return {
        "time_preference": time_preference,
        "budget_minutes": budget,
        "total_minutes": sum(int(item["minutes"]) for item in items),
        "items": items,
    }


def _public_unit(
    unit: Mapping[str, Any],
    status: str,
    concept_progress: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    practice = unit["practice"]
    modes = ", ".join(str(mode).replace("_", " ") for mode in practice.get("modes") or [])
    minimum_items = int(practice.get("minimum_items") or 0)
    practice_summary = (
        f"Complete at least {minimum_items} {modes} practice item"
        f"{'s' if minimum_items != 1 else ''}."
    )
    return {
        "id": unit["id"],
        "order": unit["order"],
        "title": unit["title"],
        "short_description": unit["short_description"],
        "ncert_sections": [
            {"id": section["id"], "title": section["title"]}
            for section in unit["ncert_sections"]
        ],
        "concepts": [
            {
                "id": str(concept["id"]),
                "title": str(concept["title"]),
                "status": str(
                    concept_progress.get(str(concept["id"]), {}).get(
                        "status",
                        "not_started",
                    )
                ),
                "evidence_count": int(
                    concept_progress.get(str(concept["id"]), {}).get(
                        "evidence_count",
                        0,
                    )
                ),
            }
            for concept in unit["concepts"]
        ],
        "skills": list(unit["skills"]),
        "practice": [practice_summary, *list(practice.get("source_refs") or [])],
        "importance": unit["importance"],
        "difficulty": unit["difficulty"],
        "estimated_minutes": dict(unit["estimated_minutes"]),
        "prerequisite_unit_ids": list(unit["prerequisite_unit_ids"]),
        "dependent_unit_ids": list(unit["dependent_unit_ids"]),
        "learning_types": list(unit["learning_types"]),
        "depth": unit["depth"],
        "exam_relevance": unit["exam_relevance"],
        "conceptual_importance": unit["conceptual_importance"],
        "why_it_matters": unit["why_it_matters"],
        "learning_route": list(unit["learning_route"]),
        "mastery_criteria": list(unit["mastery_criteria"]),
        "status": status,
        "primary_topic_id": unit["primary_topic_id"],
    }


def _completion_criteria(units: Sequence[Mapping[str, Any]]) -> List[str]:
    explicit = [
        str(criterion).strip()
        for unit in units
        for criterion in unit.get("mastery_criteria") or []
        if str(criterion).strip()
    ]
    # The detailed criteria remain on each unit.  Keep the chapter-level list
    # small enough to guide completion without becoming another syllabus dump.
    return [
        "Meet the specific mastery criteria shown inside every learning unit.",
        "Explain the dependency chain from measurement through stoichiometry in NCERT order.",
        "Complete a mixed chapter check without relying on elapsed time as proof of mastery.",
        *([f"Final application check: {explicit[-1]}"] if explicit else []),
    ]


def build_planning_roadmap(
    curriculum: Mapping[str, Any],
    *,
    study_time_today: str = "no_limit",
    analytics: Optional[Mapping[str, Any]] = None,
    persisted_states: Sequence[Any] = (),
    profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the v2 contract without allowing a model to change curriculum."""
    if study_time_today not in TIME_BUDGETS:
        study_time_today = "no_limit"
    analytics = analytics or {}
    profile = profile or {
        "current_knowledge": "some_idea",
        "learning_goal": "deep_understanding",
    }
    units = list(curriculum["units"])
    statuses = derive_unit_statuses(curriculum, analytics, persisted_states)
    concept_progress = derive_concept_progress(curriculum, analytics)
    next_unit, unmet_dependencies, completed = _next_unit(units, statuses)
    by_id = {str(unit["id"]): unit for unit in units}
    public_units = [
        _public_unit(
            unit,
            statuses[str(unit["id"])],
            concept_progress,
        )
        for unit in units
    ]
    mastered_count = sum(status == "mastered" for status in statuses.values())
    learning_count = sum(status == "learning" for status in statuses.values())
    practising_count = sum(status == "practising" for status in statuses.values())
    needs_review_count = sum(status == "needs_review" for status in statuses.values())
    unit_count = len(units)

    return {
        "roadmap_version": "planning_roadmap_v2",
        "study_time_today": study_time_today,
        "class_level": curriculum["class_level"],
        "chapter_slug": curriculum["chapter_slug"],
        "curriculum": {
            "key": curriculum["curriculum_key"],
            "source": "NCERT Class XI Chemistry Part I",
            "source_reference": dict(curriculum["source"]),
            "edition": curriculum["edition"],
            "chapter_number": curriculum["chapter_number"],
            "content_order_locked": True,
        },
        "learning_units": public_units,
        "next_step": {
            "unit_id": next_unit["id"],
            "title": next_unit["title"],
            "reason": _next_reason(
                next_unit,
                statuses[str(next_unit["id"])],
                unmet_dependencies,
                completed,
                by_id,
            ),
            "estimated_minutes": dict(next_unit["estimated_minutes"]),
        },
        "daily_route": _build_daily_route(
            units,
            statuses,
            study_time_today,
            str(next_unit["id"]),
            chapter_completed=completed,
            profile=profile,
        ),
        "progress": {
            "mastered_units": mastered_count,
            "learning_units": learning_count,
            "practising_units": practising_count,
            "needs_review_units": needs_review_count,
            "total_units": unit_count,
            "percentage": round((mastered_count / unit_count) * 100) if unit_count else 0,
        },
        "completion_criteria": _completion_criteria(units),
        "coverage": {
            "status": "complete",
            "included_unit_ids": [str(unit["id"]) for unit in units],
            "unit_count": unit_count,
        },
    }
