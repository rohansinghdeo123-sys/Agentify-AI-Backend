"""Authenticated multi-chapter Planning portfolio endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.security import require_same_user_or_admin, verify_firebase_user
from database import get_db
from Logic.planning.portfolio_engine import (
    PlanningPortfolioError,
    build_planning_portfolio,
)
from schemas import PlanningPortfolioRequest, PlanningPortfolioResponse


router = APIRouter(prefix="/planning", tags=["planning"])


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


__all__ = ["router"]
