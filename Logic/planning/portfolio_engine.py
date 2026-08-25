"""Deterministic multi-chapter Planning portfolio orchestration.

Each registered chapter keeps its own curriculum order, dependency rules, and
proficiency-specific roadmap.  This module compares only the already-eligible
next step from each chapter, chooses one global next step, and exposes one
Today route.  A time choice is therefore applied once to the portfolio rather
than once per selected chapter.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence

from Logic.analytics_engine import get_user_analytics
from services.planning_progress_service import planning_learning_states

from .curriculum_registry import resolve_planning_curriculum
from .recommendation_engine import build_planning_roadmap


class PlanningPortfolioError(ValueError):
    """Raised when a portfolio cannot be resolved without ambiguity."""


_STATUS_SCORES = {
    "needs_review": 500,
    "practising": 420,
    "learning": 380,
    "recommended": 300,
    "not_started": 280,
    "mastered": 80,
}
_IMPORTANCE_SCORES = {"very_high": 60, "high": 45, "moderate": 25, "low": 10}
_EXAM_RELEVANCE_SCORES = {"very_high": 45, "high": 30, "moderate": 15, "low": 5}


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _selection_value(selection: Any, field: str, default: Any = "") -> Any:
    if isinstance(selection, Mapping):
        return selection.get(field, default)
    return getattr(selection, field, default)


def _resolve_selections(
    selections: Sequence[Any],
    *,
    subject: str,
    class_level: str,
) -> tuple[list[Dict[str, Any]], int]:
    resolved: list[Dict[str, Any]] = []
    by_curriculum_key: Dict[str, Dict[str, Any]] = {}

    for selection_index, selection in enumerate(selections):
        chapter_ref = str(_selection_value(selection, "chapter_ref") or "").strip()
        proficiency = _normalized(
            _selection_value(selection, "chapter_proficiency", "know_a_little")
        )
        curriculum = resolve_planning_curriculum(
            chapter_ref=chapter_ref,
            subject=subject,
            class_level=class_level,
        )
        if curriculum is None:
            raise PlanningPortfolioError(
                f'Chapter "{chapter_ref}" is not available for {class_level} {subject}. '
                "Choose a registered Planning chapter and try again."
            )

        curriculum_key = str(curriculum["curriculum_key"])
        existing = by_curriculum_key.get(curriculum_key)
        if existing is not None:
            if existing["chapter_proficiency"] != proficiency:
                raise PlanningPortfolioError(
                    f'{curriculum["chapter_title"]} was selected more than once with '
                    "different proficiency levels. Keep one proficiency for each chapter."
                )
            continue

        row = {
            "selection_index": selection_index,
            "chapter_proficiency": proficiency,
            "curriculum": curriculum,
        }
        by_curriculum_key[curriculum_key] = row
        resolved.append(row)

    if not resolved:
        raise PlanningPortfolioError("Select at least one registered chapter")
    if len(resolved) > 6:
        raise PlanningPortfolioError("Select no more than six chapters")
    return resolved, len(selections) - len(resolved)


def _persisted_states(db, *, user_id: str, curriculum: Mapping[str, Any]):
    valid_unit_ids = [str(unit["id"]) for unit in curriculum["units"]]
    unit_aliases = {
        str(alias): str(unit["id"])
        for unit in curriculum["units"]
        for alias in unit.get("legacy_topic_ids") or []
    }
    return planning_learning_states(
        db,
        user_id=user_id,
        curriculum_key=str(curriculum["curriculum_key"]),
        valid_unit_ids=valid_unit_ids,
        unit_aliases=unit_aliases,
    )


def _selection_factors(
    roadmap: Mapping[str, Any],
    *,
    selection_index: int,
) -> tuple[int, list[Dict[str, Any]], str]:
    next_id = str(roadmap["next_step"]["unit_id"])
    next_unit = next(
        unit for unit in roadmap["learning_units"] if str(unit["id"]) == next_id
    )
    progress = roadmap["progress"]
    status = str(next_unit["status"])
    status_score = _STATUS_SCORES[status]

    active_units = int(progress["learning_units"]) + int(progress["practising_units"])
    completed_units = int(progress["mastered_units"])
    continuity_score = min(80, (active_units * 24) + (completed_units * 8))
    importance = str(next_unit["importance"])
    importance_score = _IMPORTANCE_SCORES[importance]
    exam_relevance = str(next_unit["exam_relevance"])
    exam_score = _EXAM_RELEVANCE_SCORES[exam_relevance]
    total_score = status_score + continuity_score + importance_score + exam_score

    status_explanations = {
        "needs_review": "Earlier assessed performance shows a gap, so this eligible step needs attention first.",
        "practising": "The student has already started practising this eligible step, so continuity is valuable.",
        "learning": "This eligible step is already being learned and should be continued before adding another start.",
        "recommended": "This is the first unfinished, prerequisite-ready unit in its chapter.",
        "not_started": "This is the earliest prerequisite-ready unit in its chapter.",
        "mastered": "This chapter is mastered; the candidate is only a light recall check.",
    }
    factors = [
        {
            "id": "status_urgency",
            "label": "Learning status",
            "value": status,
            "score": status_score,
            "explanation": status_explanations[status],
        },
        {
            "id": "chapter_continuity",
            "label": "Chapter continuity",
            "value": f"{completed_units} mastered, {active_units} active",
            "score": continuity_score,
            "explanation": "Progress already made in this chapter is protected without skipping its NCERT order.",
        },
        {
            "id": "importance",
            "label": "Concept importance",
            "value": importance,
            "score": importance_score,
            "explanation": "Importance changes attention and depth; it never reorders prerequisites inside the chapter.",
        },
        {
            "id": "exam_relevance",
            "label": "Exam relevance",
            "value": exam_relevance,
            "score": exam_score,
            "explanation": "Exam relevance helps break cross-chapter priority ties without an Exam Target control.",
        },
        {
            "id": "effort_sizing",
            "label": "Effort sizing",
            "value": str(next_unit["difficulty"]),
            "score": 0,
            "explanation": "Difficulty sizes the activity and time range; it is not used to skip curriculum order.",
        },
        {
            "id": "selection_order",
            "label": "Student selection order",
            "value": str(selection_index + 1),
            "score": 0,
            "explanation": "When educational priority is tied, the student's chapter order is the stable tie-breaker.",
        },
    ]
    selection_reason = (
        f'{next_unit["title"]} is the strongest eligible next step across the selected '
        f"chapters because its status is {status.replace('_', ' ')} and its grounded "
        f"importance is {importance.replace('_', ' ')}."
    )
    return total_score, factors, selection_reason


def _aggregate_progress(chapters: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    mastered = sum(int(item["roadmap"]["progress"]["mastered_units"]) for item in chapters)
    active = sum(
        int(item["roadmap"]["progress"]["learning_units"])
        + int(item["roadmap"]["progress"]["practising_units"])
        + int(item["roadmap"]["progress"]["recommended_units"])
        for item in chapters
    )
    needs_review = sum(
        int(item["roadmap"]["progress"]["needs_review_units"]) for item in chapters
    )
    total = sum(int(item["roadmap"]["progress"]["total_units"]) for item in chapters)
    return {
        "mastered_units": mastered,
        "active_units": active,
        "needs_review_units": needs_review,
        "total_units": total,
        "percentage": round((mastered / total) * 100) if total else 0,
    }


def build_planning_portfolio(
    db,
    *,
    user_id: str,
    class_level: str,
    subject: str,
    selections: Sequence[Any],
    study_time_today: str | None = None,
    session_duration_minutes: int | None = None,
) -> Dict[str, Any]:
    """Build independent chapter roadmaps and one globally bounded Today route."""
    requested_count = len(selections)
    resolved, deduplicated_count = _resolve_selections(
        selections,
        subject=subject,
        class_level=class_level,
    )
    analytics = get_user_analytics(db, user_id)

    candidates: list[Dict[str, Any]] = []
    for item in resolved:
        curriculum = item["curriculum"]
        roadmap = build_planning_roadmap(
            curriculum,
            chapter_proficiency=item["chapter_proficiency"],
            study_time_today=study_time_today,
            session_duration_minutes=session_duration_minutes,
            analytics=analytics,
            persisted_states=_persisted_states(
                db,
                user_id=user_id,
                curriculum=curriculum,
            ),
            profile={"chapter_proficiency": item["chapter_proficiency"]},
        )
        score, factors, selection_reason = _selection_factors(
            roadmap,
            selection_index=item["selection_index"],
        )
        candidates.append(
            {
                **item,
                "roadmap": roadmap,
                "candidate_score": score,
                "selection_factors": factors,
                "selection_reason": selection_reason,
            }
        )

    selected = sorted(
        candidates,
        key=lambda item: (-item["candidate_score"], item["selection_index"]),
    )[0]
    selected_key = str(selected["curriculum"]["curriculum_key"])

    chapter_payloads = []
    for item in candidates:
        curriculum = item["curriculum"]
        roadmap = item["roadmap"]
        chapter_payloads.append(
            {
                "curriculum_key": curriculum["curriculum_key"],
                "chapter_slug": curriculum["chapter_slug"],
                "chapter": curriculum["chapter_title"],
                "chapter_proficiency": item["chapter_proficiency"],
                "roadmap_version": roadmap["roadmap_version"],
                "curriculum": roadmap["curriculum"],
                "learning_units": roadmap["learning_units"],
                "next_step": roadmap["next_step"],
                "progress": roadmap["progress"],
                "coverage": roadmap["coverage"],
                "completion_criteria": roadmap["completion_criteria"],
                "candidate_score": item["candidate_score"],
                "selection_factors": item["selection_factors"],
                "selected_for_today": str(curriculum["curriculum_key"]) == selected_key,
            }
        )

    selected_curriculum = selected["curriculum"]
    selected_roadmap = selected["roadmap"]
    selected_next = selected_roadmap["next_step"]
    selected_route = selected_roadmap["daily_route"]
    route_items = [
        {
            **route_item,
            "curriculum_key": selected_curriculum["curriculum_key"],
            "chapter_slug": selected_curriculum["chapter_slug"],
            "chapter": selected_curriculum["chapter_title"],
        }
        for route_item in selected_route["items"]
    ]

    return {
        "portfolio_version": "planning_portfolio_v1",
        "user_id": user_id,
        "class_level": class_level,
        "subject": subject,
        "requested_chapter_count": requested_count,
        "chapter_count": len(candidates),
        "deduplicated_chapter_count": deduplicated_count,
        "study_time_today": selected_roadmap["study_time_today"],
        "session_duration_minutes": session_duration_minutes,
        "chapters": chapter_payloads,
        "global_next_step": {
            **selected_next,
            "curriculum_key": selected_curriculum["curriculum_key"],
            "chapter_slug": selected_curriculum["chapter_slug"],
            "chapter": selected_curriculum["chapter_title"],
            "chapter_proficiency": selected["chapter_proficiency"],
            "selection_reason": selected["selection_reason"],
            "candidate_score": selected["candidate_score"],
        },
        "today_route": {
            "source": selected_route["source"],
            "budget_minutes": selected_route["budget_minutes"],
            "estimated_minutes": selected_route["estimated_minutes"],
            "total_minutes": selected_route["total_minutes"],
            "items": route_items,
        },
        "selection_factors": selected["selection_factors"],
        "aggregate_progress": _aggregate_progress(candidates),
    }


__all__ = ["PlanningPortfolioError", "build_planning_portfolio"]
