"""Authenticated multi-chapter Planning portfolio endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import config
from app.security import require_same_user_or_admin, verify_firebase_user
from database import get_db
from Logic.planning.portfolio_engine import (
    PlanningPortfolioError,
    build_planning_portfolio,
)
from schemas import (
    PlanningPortfolioRequest,
    PlanningPortfolioResponse,
    StudyRouteAdvanceRequest,
    StudyRouteAdvanceResponse,
    StudyRouteBranchRequest,
    StudyRouteHeartbeatRequest,
    StudyRouteStartRequest,
    StudyRouteStateResponse,
)
from services.study_route_service import (
    StudyRouteError,
    advance_route,
    branch_route,
    complete_today,
    read_current_route,
    prepare_check,
    record_heartbeat,
    start_route,
)


router = APIRouter(prefix="/planning", tags=["planning"])


def _require_integration_enabled() -> None:
    """Return 404 when the shared route feature flag is disabled.

    Deployments may explicitly disable shared routes during rollout.
    """
    if not config.STUDY_LAB_PLANNING_INTEGRATION:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )


@router.post("/portfolio/{user_id}", response_model=PlanningPortfolioResponse)
def planning_portfolio(
    user_id: str,
    payload: PlanningPortfolioRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Build up to six chapter roadmaps with one globally bounded Today route."""
    require_same_user_or_admin(user_id, current_user)
    try:
        return build_planning_portfolio(
            db,
            user_id=user_id,
            class_level=payload.class_level,
            subject=payload.subject,
            selections=payload.chapters,
            study_time_today=payload.study_time_today,
            session_duration_minutes=payload.session_duration_minutes,
        )
    except PlanningPortfolioError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


# =========================================================
# PLANNING <-> STUDY LAB — SHARED ROUTE ENDPOINTS
# =========================================================
# All endpoints below are additive.  When STUDY_LAB_PLANNING_INTEGRATION is
# explicitly disabled they return 404 and the application behaves exactly
# as before.  No existing endpoint, response shape, or error handler is
# modified by this section.
@router.post(
    "/route/start/{user_id}",
    response_model=StudyRouteStateResponse,
)
def planning_route_start(
    user_id: str,
    payload: StudyRouteStartRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Persist today's selections and hand Study Lab the current route.

    Study Lab reads the same ``build_planning_portfolio`` output Planning
    reads, so the two surfaces can never disagree about Today's Route.
    """
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return start_route(
            db,
            user_id=user_id,
            class_level=payload.class_level,
            subject=payload.subject,
            selections=[item.model_dump() for item in payload.chapters],
            study_time_today=payload.study_time_today,
            session_duration_minutes=payload.session_duration_minutes,
        )
    except StudyRouteError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get(
    "/route/current/{user_id}",
    response_model=StudyRouteStateResponse,
)
def planning_route_current(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Return today's persisted route merged with a fresh engine computation.

    This is what Planning UI re-reads after returning from Study Lab, and what
    Study Lab re-reads after a page navigation / refresh / app relaunch.
    """
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return read_current_route(db, user_id=user_id)
    except StudyRouteError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/route/advance/{user_id}",
    response_model=StudyRouteAdvanceResponse,
)
def planning_route_advance(
    user_id: str,
    payload: StudyRouteAdvanceRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Record Study Lab completion, advance the step, return the next action.

    Evidence is written through the same writers Planning already reads from
    (``TestHistory`` insert, ``update_topic_performance``, and
    ``record_study_answer_event``), so mastery and weakness are never derived
    from a parallel state store.
    """
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return advance_route(
            db,
            user_id=user_id,
            answers=payload.answers,
            completed_unit_id=payload.completed_unit_id,
            interaction_id=payload.interaction_id,
            correct_answers=payload.correct_answers,
            total_questions=payload.total_questions,
            time_spent_seconds=payload.time_spent_seconds,
            confidence_after=payload.confidence_after,
            xp_earned=payload.xp_earned,
            last_position=payload.last_position,
        )
    except StudyRouteError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "/route/branch/{user_id}",
    response_model=StudyRouteStateResponse,
)
def planning_route_branch(
    user_id: str,
    payload: StudyRouteBranchRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Temporarily branch Study Lab to a prerequisite or revision unit.

    Only ``prerequisite_gap`` and ``revision_required`` are accepted.  The
    curriculum order is never reordered — the branch points Study Lab at an
    existing earlier unit and is cleared the moment its evidence is recorded.
    """
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return branch_route(
            db,
            user_id=user_id,
            reason=payload.reason,
            target_unit_id=payload.target_unit_id,
        )
    except StudyRouteError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "/route/complete-today/{user_id}",
    response_model=StudyRouteStateResponse,
)
def planning_route_complete_today(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Mark today's route completed on the student's behalf."""
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return complete_today(db, user_id=user_id)
    except StudyRouteError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/route/heartbeat/{user_id}",
    response_model=StudyRouteStateResponse,
)
def planning_route_heartbeat(
    user_id: str,
    payload: StudyRouteHeartbeatRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Persist a lightweight mid-step position for Continue Learning.

    Called on page navigation / refresh / background save.  It updates only
    ``last_position`` and ``last_active_at``; it never advances the step or
    changes the status.
    """
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return record_heartbeat(
            db,
            user_id=user_id,
            last_position=payload.last_position,
        )
    except StudyRouteError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

@router.post("/route/check/{user_id}")
def planning_route_check(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Resume or create a grounded check without exposing its answer key."""
    _require_integration_enabled()
    require_same_user_or_admin(user_id, current_user)
    try:
        return prepare_check(db, user_id=user_id)
    except StudyRouteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


__all__ = ["router"]
