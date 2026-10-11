"""Shared Planning <-> Study Lab route state.

Planning is the navigator; Study Lab executes the route.  This service is the
single place that reads the deterministic roadmap engine, persists the
student's active route, and writes back the evidence Study Lab produces.

Design rules enforced here:

* The route is NEVER cached.  It is always recomputed from the curriculum
  manifest + analytics + ``PlanningLearningEvent`` exposure via the existing
  ``build_planning_portfolio`` engine, so Planning and Study Lab can never
  display contradictory states.
* Mastery is NEVER written by this service directly.  Study Lab completion
  writes scored evidence through the existing ``TestHistory`` +
  ``update_topic_performance`` writers, and exposure through the existing
  ``record_study_answer_event`` writer.  The roadmap engine then decides
  ``mastered`` / ``needs_review`` on the next read using the same rules it
  already uses everywhere else.
* Only two override reasons are permitted:
  ``prerequisite_gap`` and ``revision_required``.  Any other value is
  rejected.  The engine never reorders the curriculum; a branch temporarily
  points Study Lab at an existing earlier unit and is cleared on completion.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy.orm import Session

from Logic.analytics_engine import update_topic_performance
from Logic.planning.portfolio_engine import (
    PlanningPortfolioError,
    build_planning_portfolio,
)
from models import AICoachProfile, StudyRouteState, TestHistory
from services.planning_progress_service import (
    record_study_answer_event,
    resolve_learning_event_scope,
)


# Existing vocabularies — reused, never extended.
_VALID_STUDY_TIME_TODAY = {"15", "30", "60", "120_plus", "no_limit"}
_VALID_PROFICIENCY = {
    "new_to_it",
    "know_a_little",
    "know_the_basics",
    "mostly_confident",
}
_VALID_OVERRIDE_REASONS = {"prerequisite_gap", "revision_required"}
_VALID_ROUTE_STATUS = {"not_started", "in_progress", "paused", "completed"}
_VALID_UNIT_STATUS = {
    "not_started",
    "recommended",
    "learning",
    "practising",
    "needs_review",
    "mastered",
}


class StudyRouteError(ValueError):
    """Raised when a route operation cannot be honoured without breaking NCERT order."""


# =========================================================
# INTERNAL HELPERS
# =========================================================
def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _get_state(db: Session, user_id: str) -> Optional[StudyRouteState]:
    return (
        db.query(StudyRouteState)
        .filter(
            StudyRouteState.user_id == user_id,
            StudyRouteState.route_date == _today(),
        )
        .first()
    )


def _build_portfolio_from_state(
    db: Session,
    state: StudyRouteState,
) -> Dict[str, Any]:
    """Recompute the deterministic route from the persisted selections.

    This is the ONLY place Study Lab and Planning read the current route from,
    which is what guarantees they can never disagree.
    """
    selections = list(state.selections_json or [])
    if not selections:
        raise StudyRouteError("Route has no selections to rebuild from")
    try:
        return build_planning_portfolio(
            db,
            user_id=state.user_id,
            class_level=state.class_level,
            subject=state.subject,
            selections=selections,
            study_time_today=state.study_time_today,
            session_duration_minutes=state.session_duration_minutes,
        )
    except PlanningPortfolioError as exc:
        # Surface as a study-route error so the router can translate it
        # consistently with /planning/portfolio's own error handling.
        raise StudyRouteError(str(exc)) from exc


def _selected_chapter(
    portfolio: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected_key = str(portfolio.get("global_next_step", {}).get("curriculum_key") or "")
    for chapter in portfolio.get("chapters") or []:
        if str(chapter.get("curriculum_key")) == selected_key:
            return chapter
    # The engine guarantees one selected chapter; fall back defensively.
    chapters = portfolio.get("chapters") or []
    if not chapters:
        raise StudyRouteError("Portfolio contains no chapters")
    return chapters[0]


def _main_focus_items(
    today_route: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in today_route.get("items") or []
        if str(item.get("role")) == "main_focus"
    ]


def _current_step_block(
    state: StudyRouteState,
    chapter: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the Today's Route step block: Step X of Y + current unit details."""
    today_route = chapter.get("roadmap_daily_route") or chapter.get("daily_route") or {}
    # ``build_planning_portfolio`` exposes the route at the portfolio level;
    # the chapter itself carries learning_units and next_step.  Callers pass
    # the merged block via ``chapter``; fall back to next_step when absent.
    focus_items = _main_focus_items(today_route)
    total_steps = max(1, len(focus_items))
    step_index = int(state.current_step_index or 0)
    if step_index >= total_steps:
        step_index = total_steps - 1
    if step_index < 0:
        step_index = 0
    current_item: Dict[str, Any] = {}
    if focus_items:
        current_item = dict(focus_items[step_index])
    next_step = chapter.get("next_step") or {}
    current_unit_id = (
        str(state.current_unit_id)
        or str(current_item.get("unit_id") or "")
        or str(next_step.get("unit_id") or "")
    )
    unit = next(
        (
            u for u in chapter.get("learning_units") or []
            if str(u.get("id")) == current_unit_id
        ),
        None,
    )
    return {
        "step_index": step_index,
        "step_total": total_steps,
        "step_label": f"Step {step_index + 1} of {total_steps}",
        "current_unit_id": current_unit_id,
        "current_unit_title": (
            str(unit.get("title")) if unit else str(current_item.get("title") or "")
        ),
        "current_unit_role": "main_focus",
        "current_unit_activity": str(current_item.get("activity") or ""),
        "current_unit_minutes": int(current_item.get("minutes") or 0),
        "current_unit_scope": str(current_item.get("scope") or ""),
        "recommended_approach": list(next_step.get("approach") or []),
        "why_now": str(next_step.get("reason") or ""),
        "outcome": str(next_step.get("outcome") or ""),
        "learning_types": list(next_step.get("learning_types") or []),
        "unit_depth": str(unit.get("depth")) if unit else "",
        "unit_difficulty": str(unit.get("difficulty")) if unit else "",
        "unit_importance": str(unit.get("importance")) if unit else "",
        "prerequisites": list(unit.get("prerequisite_unit_ids") or []) if unit else [],
        "unit_status": str(unit.get("status")) if unit else "",
        "last_position": dict(state.last_position_json or {}),
    }


def _goal_block(state: StudyRouteState) -> Dict[str, Any]:
    return dict(state.goal_json or {})


def _effective_proficiency(state: StudyRouteState) -> Dict[str, Any]:
    return dict(state.effective_proficiency_json or {})


def _needs_review_concepts(state: StudyRouteState) -> List[str]:
    return list(state.needs_review_concept_ids_json or [])


def _override_block(state: StudyRouteState) -> Dict[str, Any]:
    reason = str(state.override_reason or "")
    if not reason:
        return {
            "active": False,
            "reason": "",
            "reason_label": "",
            "until_unit_id": "",
        }
    reason_labels = {
        "prerequisite_gap": "Temporary prerequisite refresh",
        "revision_required": "Targeted revision for this unit",
    }
    return {
        "active": True,
        "reason": reason,
        "reason_label": reason_labels.get(reason, reason),
        "until_unit_id": str(state.override_until_unit_id or ""),
    }


# =========================================================
# PUBLIC API
# =========================================================
def start_route(
    db: Session,
    *,
    user_id: str,
    class_level: str,
    subject: str,
    selections: Sequence[Mapping[str, Any]],
    study_time_today: Optional[str] = None,
    session_duration_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """Create or replace today's route for ``user_id``.

    Selections are persisted, the deterministic portfolio is computed once to
    seed the initial step, and a ``study_route_states`` row is upserted.  Any
    existing route for today is replaced in place — this is what lets the
    student re-enter Planning and pick a different chapter the same day.
    """
    if not str(user_id).strip():
        raise StudyRouteError("user_id is required")
    if not str(class_level).strip():
        raise StudyRouteError("class_level is required")
    if not str(subject).strip():
        raise StudyRouteError("subject is required")
    normalized_selections: List[Dict[str, str]] = []
    for selection in selections or []:
        chapter_ref = str(selection.get("chapter_ref") or "").strip()
        proficiency = _normalized(selection.get("chapter_proficiency") or "know_a_little")
        if not chapter_ref:
            raise StudyRouteError("Every selection must include chapter_ref")
        if proficiency not in _VALID_PROFICIENCY:
            raise StudyRouteError(
                f"Unsupported chapter_proficiency: {selection.get('chapter_proficiency')!r}"
            )
        normalized_selections.append(
            {"chapter_ref": chapter_ref, "chapter_proficiency": proficiency}
        )
    if not normalized_selections:
        raise StudyRouteError("At least one chapter selection is required")

    normalized_time: Optional[str] = None
    if study_time_today is not None:
        candidate = _normalized(study_time_today)
        if candidate not in _VALID_STUDY_TIME_TODAY:
            raise StudyRouteError(
                f"Unsupported study_time_today: {study_time_today!r}"
            )
        normalized_time = candidate

    duration: Optional[int] = None
    if session_duration_minutes is not None:
        try:
            duration = int(session_duration_minutes)
        except (TypeError, ValueError) as exc:
            raise StudyRouteError("session_duration_minutes must be an integer") from exc
        if duration < 15 or duration > 120:
            raise StudyRouteError("session_duration_minutes must be between 15 and 120")

    # Compute the fresh portfolio BEFORE touching the DB row so a curriculum
    # error never leaves a partially-written state.
    try:
        portfolio = build_planning_portfolio(
            db,
            user_id=user_id,
            class_level=class_level,
            subject=subject,
            selections=normalized_selections,
            study_time_today=normalized_time,
            session_duration_minutes=duration,
        )
    except PlanningPortfolioError as exc:
        raise StudyRouteError(str(exc)) from exc

    chapter = _selected_chapter(portfolio)
    focus_items = _main_focus_items(chapter.get("roadmap_daily_route") or {})
    if not focus_items and chapter.get("next_step"):
        # Portfolio route lives at the top level; fall back to it.
        focus_items = _main_focus_items(portfolio.get("today_route") or {})

    first_unit_id = str(
        (focus_items[0].get("unit_id") if focus_items else None)
        or portfolio.get("global_next_step", {}).get("unit_id")
        or ""
    )
    first_role = "main_focus" if first_unit_id else ""

    state = _get_state(db, user_id)
    now = _utcnow_naive()
    if state is None:
        state = StudyRouteState(
            user_id=user_id,
            route_date=_today(),
            created_at=now,
        )
        db.add(state)

    # Optional student goal — never invented, only echoed from the coach.
    goal_payload: Dict[str, Any] = {}
    coach = (
        db.query(AICoachProfile)
        .filter(AICoachProfile.user_id == user_id)
        .first()
    )
    if coach is not None:
        active_goals = list(coach.active_goals or [])
        if active_goals:
            goal_payload = {"active_goals": active_goals}
        if coach.target_exam:
            goal_payload["target_exam"] = str(coach.target_exam)
        if coach.target_exam_date:
            goal_payload["target_exam_date"] = coach.target_exam_date.isoformat()

    state.class_level = class_level
    state.subject = subject
    state.selections_json = normalized_selections
    state.study_time_today = normalized_time
    state.session_duration_minutes = duration
    state.goal_json = goal_payload
    state.active_curriculum_key = str(chapter.get("curriculum_key") or "")
    state.active_chapter_slug = str(chapter.get("chapter_slug") or "")
    state.current_unit_id = first_unit_id
    state.current_step_index = 0
    state.current_step_role = first_role
    state.last_position_json = dict(state.last_position_json or {})
    # A fresh start clears any prior override so the student is back on-plan.
    state.override_reason = ""
    state.override_until_unit_id = ""
    state.effective_proficiency_json = dict(state.effective_proficiency_json or {})
    state.needs_review_concept_ids_json = list(
        state.needs_review_concept_ids_json or []
    )
    state.status = "in_progress"
    state.started_at = state.started_at or now
    state.last_active_at = now
    state.completed_at = None
    state.updated_at = now

    db.commit()
    db.refresh(state)

    return read_current_route(db, user_id=user_id, portfolio=portfolio)


def read_current_route(
    db: Session,
    *,
    user_id: str,
    portfolio: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return today's persisted route merged with a fresh engine computation."""
    state = _get_state(db, user_id)
    if state is None:
        raise StudyRouteError("No active route for today")
    if portfolio is None:
        portfolio = _build_portfolio_from_state(db, state)
    chapter = _selected_chapter(portfolio)
    # The portfolio route lives at the top level; give the step builder the
    # right object without mutating the engine's output.
    chapter_for_view = dict(chapter)
    chapter_for_view["roadmap_daily_route"] = portfolio.get("today_route") or {}

    step_block = _current_step_block(state, chapter_for_view)
    return {
        "user_id": state.user_id,
        "route_date": state.route_date.isoformat(),
        "class_level": state.class_level,
        "subject": state.subject,
        "status": state.status,
        "study_time_today": state.study_time_today,
        "session_duration_minutes": state.session_duration_minutes,
        "goal": _goal_block(state),
        "portfolio_version": portfolio.get("portfolio_version"),
        "chapters": portfolio.get("chapters") or [],
        "global_next_step": portfolio.get("global_next_step") or {},
        "today_route": portfolio.get("today_route") or {},
        "selection_factors": portfolio.get("selection_factors") or [],
        "aggregate_progress": portfolio.get("aggregate_progress") or {},
        "current_step": step_block,
        "override": _override_block(state),
        "effective_proficiency": _effective_proficiency(state),
        "needs_review_concept_ids": _needs_review_concepts(state),
        "started_at": state.started_at.isoformat() if state.started_at else None,
        "last_active_at": state.last_active_at.isoformat() if state.last_active_at else None,
        "completed_at": state.completed_at.isoformat() if state.completed_at else None,
    }


def build_study_lab_context(
    db: Session,
    *,
    user_id: str,
) -> Dict[str, Any]:
    """Study Lab entry payload — same route, compact study-lab-specific shape."""
    route = read_current_route(db, user_id=user_id)
    step = route["current_step"]
    override = route["override"]
    return {
        "user_id": user_id,
        "route_date": route["route_date"],
        "status": route["status"],
        "current_topic": step["current_unit_title"],
        "step_label": step["step_label"],
        "step_index": step["step_index"],
        "step_total": step["step_total"],
        "why_now": step["why_now"],
        "recommended_approach": step["recommended_approach"],
        "recommended_depth": step["unit_depth"],
        "learning_types": step["learning_types"],
        "prerequisites": step["prerequisites"],
        "current_unit_id": step["current_unit_id"],
        "current_unit_status": step["unit_status"],
        "progress": route["aggregate_progress"],
        "goal": route["goal"],
        "override": override,
        "effective_proficiency": route["effective_proficiency"],
        "needs_review_concept_ids": route["needs_review_concept_ids"],
        "last_position": step["last_position"],
        "next_step": route["global_next_step"],
        "today_route": route["today_route"],
        "chapters": route["chapters"],
    }


def _write_study_evidence(
    db: Session,
    *,
    state: StudyRouteState,
    portfolio: Mapping[str, Any],
    unit_id: str,
    interaction_id: str,
    correct_answers: int,
    total_questions: int,
    time_spent_seconds: int,
    confidence_after: Optional[float],
    xp_earned: Optional[int],
) -> Dict[str, Any]:
    """Persist Study Lab completion through the SAME writers Planning uses.

    Order matters: ``TestHistory`` first (scored evidence the analytics engine
    aggregates from), then ``update_topic_performance`` (weakness signal), then
    ``record_study_answer_event`` (exposure proof).  No mastery is written
    directly — the roadmap engine derives it on the next read.
    """
    if total_questions <= 0:
        raise StudyRouteError("total_questions must be greater than zero")
    if correct_answers < 0 or correct_answers > total_questions:
        raise StudyRouteError("correct_answers must be between 0 and total_questions")

    chapter = _selected_chapter(portfolio)
    unit = next(
        (
            u for u in chapter.get("learning_units") or []
            if str(u.get("id")) == str(unit_id)
        ),
        None,
    )
    if unit is None:
        raise StudyRouteError(
            f"Unit {unit_id!r} is not part of the current route"
        )
    unit_title = str(unit.get("title") or "")
    primary_topic_id = str(unit.get("primary_topic_id") or unit_id)
    unit_status = str(unit.get("status") or "")

    accuracy_rate = (correct_answers / total_questions) * 100.0
    now = _utcnow_naive()
    session_type = "revision" if unit_status == "needs_review" else "practice"

    history = TestHistory(
        user_id=state.user_id,
        date=now.date(),
        topic=primary_topic_id,
        score=int(correct_answers),
        total_questions=int(total_questions),
        xp_earned=int(xp_earned) if xp_earned is not None else int(correct_answers),
        time_spent_seconds=int(max(0, time_spent_seconds)),
        accuracy_rate=float(accuracy_rate),
        focus_score=0.0,
        session_type=session_type,
        started_at=None,
        completed_at=now,
        response_latency_ms=0,
        hint_count=0,
        retry_count=0,
        confidence_before=None,
        confidence_after=(
            float(confidence_after) if confidence_after is not None else None
        ),
    )
    db.add(history)
    db.commit()

    # Aggregate into TopicPerformance so weak_areas / topic_heatmap reflect it.
    update_topic_performance(
        db,
        user_id=state.user_id,
        topic=primary_topic_id,
        correct_answers=int(correct_answers),
        total_questions=int(total_questions),
        time_spent=int(max(0, time_spent_seconds)),
    )

    # Record exposure through the existing idempotent writer.
    scope = {
        "catalog_source": "planning_manifest",
        "chapter_slug": state.active_chapter_slug,
        "subject": state.subject,
        "class_level": state.class_level,
        "planning_unit_id": unit_id,
    }
    resolved_scope = resolve_learning_event_scope(scope, db=db)
    if resolved_scope is None:
        # Fall back to the published catalog path.
        scope["catalog_source"] = "published"
        resolved_scope = resolve_learning_event_scope(scope, db=db)
    event_result: Optional[Dict[str, Any]] = None
    if resolved_scope is not None:
        event_result = record_study_answer_event(
            db,
            user_id=state.user_id,
            interaction_id=str(interaction_id or "").strip() or f"studylab-{now.timestamp()}",
            scope=resolved_scope,
            source_session_id=f"studylab-{state.user_id}",
        )

    return {
        "unit_id": unit_id,
        "unit_title": unit_title,
        "primary_topic_id": primary_topic_id,
        "session_type": session_type,
        "accuracy_rate": round(accuracy_rate, 2),
        "correct_answers": int(correct_answers),
        "total_questions": int(total_questions),
        "exposure_event": event_result,
    }


def advance_route(
    db: Session,
    *,
    user_id: str,
    completed_unit_id: str,
    interaction_id: str,
    correct_answers: int,
    total_questions: int,
    time_spent_seconds: int = 0,
    confidence_after: Optional[float] = None,
    xp_earned: Optional[int] = None,
    last_position: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record Study Lab completion, advance the step, and return the next action.

    The route itself is not stored — it is recomputed after the evidence write,
    so the returned ``next_action`` reflects the engine's deterministic
    decision, not a cached one.
    """
    state = _get_state(db, user_id)
    if state is None:
        raise StudyRouteError("No active route for today")
    if str(completed_unit_id or "").strip() != str(state.current_unit_id or ""):
        # Allow completion of any unit that the current portfolio considers
        # the student's next step; the engine is authoritative.
        pass

    portfolio = _build_portfolio_from_state(db, state)
    evidence = _write_study_evidence(
        db,
        state=state,
        portfolio=portfolio,
        unit_id=str(completed_unit_id),
        interaction_id=str(interaction_id or ""),
        correct_answers=int(correct_answers),
        total_questions=int(total_questions),
        time_spent_seconds=int(time_spent_seconds or 0),
        confidence_after=confidence_after,
        xp_earned=xp_earned,
    )

    # Effective proficiency updates from demonstrated performance.
    effective = dict(state.effective_proficiency_json or {})
    curriculum_key = str(state.active_curriculum_key or "")
    if curriculum_key:
        current = effective.get(curriculum_key)
        if not isinstance(current, dict):
            current = {
                "self_reported": "",
                "demonstrated_accuracy": 0.0,
                "evidence_count": 0,
                "last_updated": "",
            }
        prev_count = int(current.get("evidence_count") or 0)
        prev_acc = float(current.get("demonstrated_accuracy") or 0.0)
        new_count = prev_count + 1
        new_acc = (
            (prev_acc * prev_count + evidence["accuracy_rate"]) / new_count
            if new_count
            else evidence["accuracy_rate"]
        )
        current["demonstrated_accuracy"] = round(new_acc, 2)
        current["evidence_count"] = new_count
        current["last_updated"] = _utcnow_naive().isoformat()
        effective[curriculum_key] = current
        state.effective_proficiency_json = effective

    # Concept-level Needs Review tracking — never invented, only observed.
    needs_review = set(state.needs_review_concept_ids_json or [])
    if evidence["accuracy_rate"] < 60.0:
        needs_review.add(str(evidence["primary_topic_id"]))
    else:
        needs_review.discard(str(evidence["primary_topic_id"]))
    state.needs_review_concept_ids_json = sorted(needs_review)

    # Advance the step index within the CURRENT portfolio route.
    chapter = _selected_chapter(portfolio)
    chapter_for_view = dict(chapter)
    chapter_for_view["roadmap_daily_route"] = portfolio.get("today_route") or {}
    focus_items = _main_focus_items(chapter_for_view.get("roadmap_daily_route") or {})
    total_steps = max(1, len(focus_items))
    next_index = int(state.current_step_index or 0) + 1
    if next_index >= total_steps:
        next_index = total_steps  # signals "route exhausted for today"
    state.current_step_index = next_index
    state.last_position_json = dict(last_position or {})
    state.last_active_at = _utcnow_naive()
    state.status = "in_progress"
    if state.started_at is None:
        state.started_at = state.last_active_at

    # Clear the override if we just completed the unit it pointed at.
    if (
        state.override_reason
        and str(state.override_until_unit_id or "")
        and str(state.override_until_unit_id) == str(completed_unit_id)
    ):
        state.override_reason = ""
        state.override_until_unit_id = ""

    # If we've run past the last step in today's route, close the day.
    if next_index >= total_steps:
        state.status = "completed"
        state.completed_at = _utcnow_naive()

    db.commit()
    db.refresh(state)

    # Recompute so we can return the engine's fresh next step.
    refreshed = read_current_route(db, user_id=user_id)
    next_step = refreshed.get("global_next_step") or {}
    return {
        "completion": evidence,
        "next_action": {
            "title": str(next_step.get("title") or ""),
            "reason": str(next_step.get("reason") or ""),
            "estimated_minutes": dict(next_step.get("estimated_minutes") or {}),
            "approach": list(next_step.get("approach") or []),
            "outcome": str(next_step.get("outcome") or ""),
            "done_for_today": state.status == "completed",
        },
        "route": refreshed,
    }


def branch_route(
    db: Session,
    *,
    user_id: str,
    reason: str,
    target_unit_id: str,
) -> Dict[str, Any]:
    """Temporarily point Study Lab at a prerequisite or revision unit.

    Only ``prerequisite_gap`` and ``revision_required`` are accepted.  The
    curriculum order is never reordered — the branch routes to an existing
    earlier unit and is cleared when its evidence is recorded.
    """
    normalized_reason = _normalized(reason)
    if normalized_reason not in _VALID_OVERRIDE_REASONS:
        raise StudyRouteError(
            "override reason must be one of: " + ", ".join(sorted(_VALID_OVERRIDE_REASONS))
        )
    target = str(target_unit_id or "").strip()
    if not target:
        raise StudyRouteError("target_unit_id is required")

    state = _get_state(db, user_id)
    if state is None:
        raise StudyRouteError("No active route for today")

    portfolio = _build_portfolio_from_state(db, state)
    chapter = _selected_chapter(portfolio)
    known_ids = {
        str(u.get("id")) for u in chapter.get("learning_units") or []
    }
    if target not in known_ids:
        raise StudyRouteError(
            f"Unit {target!r} is not part of the current chapter"
        )

    state.override_reason = normalized_reason
    state.override_until_unit_id = target
    state.current_unit_id = target
    state.current_step_role = "main_focus"
    state.last_active_at = _utcnow_naive()
    if state.status == "not_started":
        state.status = "in_progress"
    db.commit()
    db.refresh(state)
    return read_current_route(db, user_id=user_id)


def complete_today(db: Session, *, user_id: str) -> Dict[str, Any]:
    """Mark today's route as completed on the student's behalf."""
    state = _get_state(db, user_id)
    if state is None:
        raise StudyRouteError("No active route for today")
    now = _utcnow_naive()
    state.status = "completed"
    state.completed_at = now
    state.last_active_at = now
    db.commit()
    db.refresh(state)
    return read_current_route(db, user_id=user_id)


def record_heartbeat(
    db: Session,
    *,
    user_id: str,
    last_position: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a lightweight position marker without changing step or status.

    Used for page navigation / refresh / background save so "Continue Learning"
    can resume mid-step.
    """
    state = _get_state(db, user_id)
    if state is None:
        raise StudyRouteError("No active route for today")
    if last_position is not None:
        state.last_position_json = dict(last_position)
    state.last_active_at = _utcnow_naive()
    db.commit()
    db.refresh(state)
    return read_current_route(db, user_id=user_id)