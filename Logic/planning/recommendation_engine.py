"""Deterministic NCERT-ordered roadmap and daily-route recommendations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .curriculum_registry import STATUS_VALUES


DEFAULT_ROUTE_CEILING_MINUTES = 30
MIN_SESSION_MINUTES = 15
MAX_SESSION_MINUTES = 120
MIN_FOLLOW_ON_BLOCK_MINUTES = 15
STUDY_TIME_BUDGETS: Dict[str, Optional[int]] = {
    "15": 15,
    "30": 30,
    "60": 60,
    "120_plus": 120,
    "no_limit": None,
}
PROFICIENCY_VALUES = {
    "new_to_it",
    "know_a_little",
    "know_the_basics",
    "mostly_confident",
}


def _proficiency(profile: Mapping[str, Any]) -> str:
    value = _normalized(profile.get("chapter_proficiency"))
    if value in PROFICIENCY_VALUES:
        return value
    # Compatibility for server-side callers that still pass the retired
    # knowledge vocabulary. Fast Track now maps into the one Quick Revision
    # behaviour; it is never a separate planning path.
    if _normalized(profile.get("learning_goal")) in {"fast_track", "quick_revision"}:
        return "mostly_confident"
    return {
        "new": "new_to_it",
        "some_idea": "know_a_little",
        "know_basics": "know_the_basics",
        "weak_basics": "know_a_little",
    }.get(_normalized(profile.get("current_knowledge")), "know_a_little")


def _route_budget(
    study_time_today: Optional[str],
    session_duration_minutes: Optional[int],
) -> tuple[Optional[int], str, Optional[str]]:
    """Resolve a ceiling without turning available time into a target.

    An explicit student choice takes precedence over ambient session state.
    Absence is a calm automatic route capped at 30 minutes; ``no_limit`` has no
    ceiling but still schedules only the current content-sized learning unit.
    """

    preference = _normalized(study_time_today)
    if preference in STUDY_TIME_BUDGETS:
        return STUDY_TIME_BUDGETS[preference], "student_choice", preference
    if session_duration_minutes is not None:
        try:
            duration = int(session_duration_minutes)
        except (TypeError, ValueError):
            duration = DEFAULT_ROUTE_CEILING_MINUTES
        duration = max(MIN_SESSION_MINUTES, min(MAX_SESSION_MINUTES, duration))
        # Ambient session state can be arbitrary (for example 48 minutes).
        # Floor it to a calm five-minute block so the plan never exceeds the
        # time actually available; the response keeps the original context.
        floored_duration = max(MIN_SESSION_MINUTES, int(duration // 5 * 5))
        return floored_duration, "session_state", None
    return DEFAULT_ROUTE_CEILING_MINUTES, "default_focus", None


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


def _normalized_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    # Test-history confidence is defined by the public API as a 0-100 value.
    # Do not guess at alternate scales: a genuine low score such as ``1`` must
    # stay low rather than becoming false mastery evidence.
    return max(0.0, min(100.0, confidence))


def _canonical_evidence_dimensions(
    analytics: Mapping[str, Any],
    canonical_keys: set[str],
) -> Dict[str, Any]:
    """Summarise independent evidence without treating time as mastery."""
    rows = [
        item
        for item in analytics.get("topic_evidence") or []
        if isinstance(item, Mapping)
        and _normalized(item.get("topic")) in canonical_keys
    ]
    session_types = {
        _normalized(item.get("session_type"))
        for item in rows
        if _normalized(item.get("session_type"))
    }
    recall_types = {"recall", "revision", "quick_revision"}
    practice_types = {
        "practice",
        "exam",
        "mcq",
        "assessment",
        "question_paper",
        "written_practice",
    }
    confidences = [
        confidence
        for item in rows
        if (confidence := _normalized_confidence(item.get("confidence_after"))) is not None
    ]
    session_tokens = {
        token
        for session_type in session_types
        for token in session_type.split("_")
        if token
    }
    return {
        "has_recall": bool(
            session_types.intersection(recall_types)
            or session_tokens.intersection(recall_types)
        ),
        "has_practice": bool(
            session_types.intersection(practice_types)
            or session_tokens.intersection(practice_types)
        ),
        "confidence": sum(confidences) / len(confidences) if confidences else None,
    }


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
        was_mastered = statuses[unit_id] == "mastered"
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
        dimensions = _canonical_evidence_dimensions(analytics, canonical_keys)
        study_exposure = int(
            getattr(persisted_by_unit.get(unit_id), "evidence_count", 0) or 0
        ) > 0
        independent_dimensions = 1 + sum(
            (
                study_exposure,
                bool(dimensions["has_recall"]),
                bool(dimensions["has_practice"]),
                dimensions["confidence"] is not None and dimensions["confidence"] >= 70,
            )
        )
        if (
            canonical_attempts >= _mastery_attempt_requirement(unit)
            and canonical_accuracy >= 80
            and independent_dimensions >= 2
        ):
            statuses[unit_id] = "mastered"
            continue

        attempts = sum(max(1, _analytics_attempts(item)) for item in matching.values())
        weighted_accuracy = sum(
            _analytics_accuracy(item)
            * max(1, _analytics_attempts(item))
            for item in matching.values()
        ) / max(attempts, 1)
        if weighted_accuracy < 60:
            statuses[unit_id] = "needs_review"
        elif not was_mastered:
            statuses[unit_id] = "practising"
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
    proficiency = _proficiency(profile)
    if proficiency == "new_to_it":
        extra += 4
    elif proficiency == "know_a_little":
        extra = round(extra * 0.82)
    elif proficiency == "know_the_basics":
        extra = round(extra * 0.55)
    else:
        extra = round(extra * 0.32)
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
    profile: Mapping[str, Any],
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
    return {
        "new_to_it": (
            "This is the first unfinished NCERT unit. Starting here builds the foundation "
            "before guided examples and practice."
        ),
        "know_a_little": (
            "This is the earliest unfinished NCERT unit. A short reinforcement pass here "
            "will make the later applications more reliable."
        ),
        "know_the_basics": (
            "This is the earliest unverified NCERT unit. Confirm it quickly, then use an "
            "application to expose any gap before moving forward."
        ),
        "mostly_confident": (
            "This is the earliest NCERT unit without demonstrated mastery. Begin with a "
            "quick diagnostic scan and spend effort only where the check finds a gap."
        ),
    }[_proficiency(profile)]


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
    return {
        "new_to_it": f"Understand {title} with a guided example, then explain the main idea once",
        "know_a_little": f"Reinforce {title} with one example and a moderate practice check",
        "know_the_basics": f"Scan {title}, apply it once, and use the result to find any gap",
        "mostly_confident": f"Diagnose {title} quickly, then revise or practise only the weak part",
    }[_proficiency(profile)]


def _approach(
    unit: Mapping[str, Any],
    status: str,
    profile: Mapping[str, Any],
) -> List[str]:
    if status == "needs_review":
        return ["Revisit prerequisite", "Worked example", "Targeted practice", "Quick check"]
    if status == "practising":
        return ["Recall", "Apply", "Practise", "Quick check"]
    if status == "learning":
        return ["Continue", "Example", "Practice", "Quick check"]
    return {
        "new_to_it": ["Understand", "Guided example", "Practice", "Quick check"],
        "know_a_little": ["Recall", "Reinforce", "Practice", "Quick check"],
        "know_the_basics": ["Concept scan", "Application", "Gap check", "Practice"],
        "mostly_confident": ["Quick scan", "Diagnostic", "Weak-area practice", "Revision"],
    }[_proficiency(profile)]


def _outcome(unit: Mapping[str, Any]) -> str:
    criteria = [str(value).strip() for value in unit.get("mastery_criteria") or [] if str(value).strip()]
    if criteria:
        return criteria[0]
    return f"Explain and apply the central idea in {unit['title']} without relying on notes."


def _build_daily_route(
    units: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, str],
    study_time_today: Optional[str],
    session_duration_minutes: Optional[int],
    start_unit_id: str,
    *,
    chapter_completed: bool,
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    budget, source, _ = _route_budget(study_time_today, session_duration_minutes)
    start_index = next(index for index, unit in enumerate(units) if unit["id"] == start_unit_id)
    if chapter_completed:
        unit = units[-1]
        # Mastered work needs a light recall, not an invented 20-minute floor.
        recall_minutes = 10 if budget is None else min(10, budget)
        return {
            "source": source,
            "budget_minutes": budget,
            "estimated_minutes": {
                "min": max(5, recall_minutes - 5),
                "max": recall_minutes,
            },
            "total_minutes": recall_minutes,
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
                    "role": "quick_check",
                    "minutes": recall_minutes,
                    "scope": "partial",
                }
            ],
        }

    # Size every scheduled block from its authored content range.  A selected
    # duration is a ceiling, never a target to pad.  When a student genuinely
    # has enough time after the current unit, continue in NCERT order instead
    # of hard-stopping every route at the former one-unit/60-minute cap.
    # ``no_limit`` deliberately remains one content-sized unit so it cannot
    # turn into an unbounded whole-chapter demand.
    remaining = budget
    items: List[Dict[str, Any]] = []
    total = 0
    estimate_minimum = 0
    for unit in units[start_index:]:
        unit_id = str(unit["id"])
        if statuses.get(unit_id) == "mastered":
            continue
        if remaining is not None and items and remaining < MIN_FOLLOW_ON_BLOCK_MINUTES:
            break

        natural_total = _recommended_minutes(unit, profile)
        allocated = natural_total if remaining is None else min(natural_total, remaining)
        allocated = max(5, int(allocated))
        unit_minimum = int(unit["estimated_minutes"]["min"])
        scope = "full_unit" if allocated >= unit_minimum else "partial"

        # The demonstrated check belongs inside each content-sized block. This
        # keeps a 15-minute selection at 15 minutes rather than adding time.
        check_minutes = 5 if allocated >= 10 else 0
        focus_minutes = allocated - check_minutes
        continued = bool(items)
        items.append(
            {
                "unit_id": unit_id,
                "title": unit["title"],
                "activity": _activity(unit, statuses[unit_id], profile),
                "reason": (
                    "Continue here only after the previous quick check is secure; "
                    "this is the next NCERT-ordered, prerequisite-linked unit."
                    if continued
                    else _route_reason(unit, statuses[unit_id])
                ),
                "role": "main_focus",
                "minutes": focus_minutes,
                "scope": scope,
            }
        )
        if check_minutes:
            items.append(
                {
                    "unit_id": unit_id,
                    "title": unit["title"],
                    "activity": f"Quick check: {_outcome(unit)}",
                    "reason": "A short check turns study into evidence and decides whether to continue or revisit.",
                    "role": "quick_check",
                    "minutes": check_minutes,
                    "scope": "partial",
                }
            )

        total += allocated
        estimate_minimum += allocated if scope == "partial" else min(unit_minimum, allocated)
        if remaining is None:
            break
        remaining -= allocated
        if remaining <= 0:
            break

    estimate = {"min": estimate_minimum, "max": total}
    return {
        "source": source,
        "budget_minutes": budget,
        "estimated_minutes": estimate,
        "total_minutes": total,
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
        "ncert_subtopics": [
            {
                "id": subtopic["id"],
                "title": subtopic["title"],
                "section_id": subtopic["section_id"],
            }
            for subtopic in unit["ncert_subtopics"]
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
    sequence_criterion = (
        f"Explain how {units[0]['title']} builds toward {units[-1]['title']} in NCERT order."
        if units
        else "Explain how each learning unit connects to the next in NCERT order."
    )
    return [
        "Meet the specific mastery criteria shown inside every learning unit.",
        sequence_criterion,
        "Complete a mixed chapter check without relying on elapsed time as proof of mastery.",
        *([f"Final application check: {explicit[-1]}"] if explicit else []),
    ]


def build_planning_roadmap(
    curriculum: Mapping[str, Any],
    *,
    chapter_proficiency: str = "know_a_little",
    study_time_today: Optional[str] = None,
    session_duration_minutes: Optional[int] = None,
    analytics: Optional[Mapping[str, Any]] = None,
    persisted_states: Sequence[Any] = (),
    profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the v2 contract without allowing a model to change curriculum."""
    analytics = analytics or {}
    normalized_proficiency = _normalized(chapter_proficiency)
    if normalized_proficiency not in PROFICIENCY_VALUES:
        normalized_proficiency = "know_a_little"
    profile = dict(profile or {})
    profile["chapter_proficiency"] = _proficiency(
        {**profile, "chapter_proficiency": normalized_proficiency}
    )
    units = list(curriculum["units"])
    statuses = derive_unit_statuses(curriculum, analytics, persisted_states)
    concept_progress = derive_concept_progress(curriculum, analytics)
    next_unit, unmet_dependencies, completed = _next_unit(units, statuses)
    if not completed and statuses[str(next_unit["id"])] == "not_started":
        statuses[str(next_unit["id"])] = "recommended"
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
    recommended_count = sum(status == "recommended" for status in statuses.values())
    unit_count = len(units)
    daily_route = _build_daily_route(
        units,
        statuses,
        study_time_today,
        session_duration_minutes,
        str(next_unit["id"]),
        chapter_completed=completed,
        profile=profile,
    )
    if completed:
        next_step_estimate = dict(daily_route["estimated_minutes"])
    else:
        next_route_items = [
            item
            for item in daily_route["items"]
            if str(item["unit_id"]) == str(next_unit["id"])
        ]
        next_route_total = sum(int(item["minutes"]) for item in next_route_items)
        next_unit_minimum = int(next_unit["estimated_minutes"]["min"])
        next_step_estimate = {
            "min": (
                next_route_total
                if next_route_total < next_unit_minimum
                else min(next_unit_minimum, next_route_total)
            ),
            "max": next_route_total,
        }

    return {
        "roadmap_version": "planning_roadmap_v2",
        "chapter_proficiency": profile["chapter_proficiency"],
        "study_time_today": (
            _normalized(study_time_today)
            if _normalized(study_time_today) in STUDY_TIME_BUDGETS
            else None
        ),
        "session_duration_minutes": session_duration_minutes,
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
                profile,
            ),
            # Unit cards retain their full authored range. The next action is
            # only the first unit-sized block, while ``daily_route`` may include
            # later conditional units when a longer selected ceiling permits.
            "estimated_minutes": next_step_estimate,
            "importance": next_unit["importance"],
            "learning_types": list(next_unit["learning_types"]),
            "approach": _approach(
                next_unit,
                statuses[str(next_unit["id"])],
                profile,
            ),
            "outcome": _outcome(next_unit),
        },
        "daily_route": daily_route,
        "progress": {
            "mastered_units": mastered_count,
            "learning_units": learning_count,
            "practising_units": practising_count,
            "needs_review_units": needs_review_count,
            "recommended_units": recommended_count,
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
