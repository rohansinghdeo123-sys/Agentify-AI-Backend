"""Progress, sessions, analytics, leaderboard, and dashboard endpoints."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.security import (
    is_backend_admin,
    require_same_user_or_admin,
    verify_firebase_user,
)
from app.request_models import SubmitSessionRequest
from app.serializers import format_test_session, normalize_topic, progress_payload
from database import get_db
from Logic.analytics_engine import get_user_analytics, calculate_cognitive_metrics
from models import TestHistory, TopicPerformance, UserProgress
from schemas import (
    ProgressResponse,
    ProgressUpdate,
    TestHistoryCreate,
    TestHistoryResponse,
)
from services.leaderboard_service import build_leaderboard
from services.progress_service import apply_streak, create_test_history, get_or_create_progress
from services.profile_service import profile_learning_context
from services.revision_scheduler import build_revision_queue, summarize_queue
from services.ttl_cache import TTLCache

router = APIRouter(tags=["progress"])

_pulse_cache = TTLCache(max_entries=4)


@router.post("/update-progress")
def update_progress(
    progress: ProgressUpdate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(progress.user_id, current_user)
    allow_client_overwrite = os.getenv("ALLOW_CLIENT_PROGRESS_OVERWRITE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not allow_client_overwrite and not is_backend_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Direct progress overwrite is disabled. Submit completed sessions instead.",
        )

    user = get_or_create_progress(db, progress.user_id)

    user.total_tests = progress.total_tests
    user.total_questions = progress.total_questions
    user.total_correct = progress.total_correct
    user.xp = progress.xp

    apply_streak(user)

    db.commit()
    db.refresh(user)

    return {
        "message": "Progress updated successfully",
        "progress": progress_payload(user),
        "streak": user.streak,
    }


@router.get("/get-progress/{user_id}", response_model=ProgressResponse)
def get_progress(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)

    user = get_or_create_progress(db, user_id)
    payload = progress_payload(user)

    return ProgressResponse(**payload)


@router.post("/submit-session")
def submit_session(
    payload: SubmitSessionRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(payload.user_id, current_user)

    if payload.total_questions <= 0:
        raise HTTPException(status_code=400, detail="total_questions must be greater than zero")

    correct = max(0, min(payload.score, payload.total_questions))
    # XP is server-authoritative: it feeds the leaderboard and rival battles,
    # so the client's xp_earned is accepted for compatibility but ignored.
    xp_earned = correct * 10

    test = create_test_history(
        db=db,
        user_id=payload.user_id,
        topic=payload.topic,
        score=correct,
        total_questions=payload.total_questions,
        xp_earned=xp_earned,
        time_spent_seconds=payload.time_spent_seconds,
        focus_score=payload.focus_score,
        session_type=payload.session_type,
        replay_data=payload.replay_data,
        started_at=payload.started_at,
        completed_at=payload.completed_at,
        response_latency_ms=payload.response_latency_ms,
        hint_count=payload.hint_count,
        retry_count=payload.retry_count,
        confidence_before=payload.confidence_before,
        confidence_after=payload.confidence_after,
    )

    user = get_or_create_progress(db, payload.user_id)
    user.total_tests += 1
    user.total_questions += payload.total_questions
    user.total_correct += correct
    user.xp += xp_earned
    user.focus_score = payload.focus_score

    apply_streak(user)

    db.commit()
    db.refresh(test)
    db.refresh(user)

    return {
        "message": "Session submitted successfully",
        "session": format_test_session(
            test,
            profile_learning_context(db, payload.user_id).get("class_level", ""),
        ),
        "progress": progress_payload(user),
    }


@router.post("/save-test")
def save_test(
    test: TestHistoryCreate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(test.user_id, current_user)

    topic = normalize_topic(test.topic)
    total = int(test.total_questions)
    correct = max(0, min(int(test.score), total)) if total > 0 else max(0, int(test.score))
    # Server-authoritative XP (see /submit-session).
    xp_earned = correct * 10

    new_test = create_test_history(
        db=db,
        user_id=test.user_id,
        topic=topic,
        score=correct,
        total_questions=total,
        xp_earned=xp_earned,
        time_spent_seconds=int(test.time_spent_seconds or 0),
        focus_score=float(test.focus_score or 0),
        session_type=test.session_type or "exam",
        replay_data=test.replay_data,
        started_at=test.started_at,
        completed_at=test.completed_at,
        response_latency_ms=test.response_latency_ms,
        hint_count=test.hint_count,
        retry_count=test.retry_count,
        confidence_before=test.confidence_before,
        confidence_after=test.confidence_after,
    )

    db.commit()
    db.refresh(new_test)

    return {
        "message": "Test saved successfully",
        "topic": topic,
        "analytics_updated": True,
        "session": format_test_session(
            new_test,
            profile_learning_context(db, test.user_id).get("class_level", ""),
        ),
    }


@router.get("/sessions/{user_id}")
def get_sessions(
    user_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    class_level = profile_learning_context(db, user_id).get("class_level", "")

    base = db.query(TestHistory).filter(TestHistory.user_id == user_id)
    total = base.count()
    tests = (
        base.order_by(TestHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "user_id": user_id,
        "sessions": [format_test_session(test, class_level) for test in tests],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/test-history/{user_id}", response_model=List[TestHistoryResponse])
def get_test_history(
    user_id: str,
    limit: int = Query(default=500, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)

    tests = (
        db.query(TestHistory)
        .filter(TestHistory.user_id == user_id)
        .order_by(TestHistory.date.asc(), TestHistory.id.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return tests


@router.get("/session-replay/{test_id}")
def get_session_replay(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    test = db.query(TestHistory).filter(TestHistory.id == test_id).first()

    if not test:
        raise HTTPException(status_code=404, detail="Session not found")

    require_same_user_or_admin(test.user_id, current_user)

    replay = test.details.replay_data if test.details else {}

    return {
        "id": test.id,
        "topic": test.topic,
        "date": test.date.isoformat() if test.date else None,
        "replay_data": replay,
    }


@router.get("/public/pulse")
def public_pulse(db: Session = Depends(get_db)):
    """Anonymized platform aggregates for logged-out surfaces (login page).

    Deliberately public: returns counts and totals only — never user ids,
    names, emails, phones, or any per-student record.
    """

    def build() -> Dict[str, Any]:
        students, total_xp, top_streak = db.query(
            func.count(UserProgress.id),
            func.coalesce(func.sum(UserProgress.xp), 0),
            func.coalesce(func.max(UserProgress.streak), 0),
        ).one()
        week_ago = date.today() - timedelta(days=7)
        sessions_7d = (
            db.query(func.count(TestHistory.id)).filter(TestHistory.date >= week_ago).scalar() or 0
        )
        return {
            "students": int(students or 0),
            "total_xp": int(total_xp or 0),
            "top_streak": int(top_streak or 0),
            "sessions_7d": int(sessions_7d),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    return _pulse_cache.get_or_build("pulse", 60.0, build)


@router.get("/leaderboard")
def leaderboard(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    return build_leaderboard(db, current_user)


@router.get("/revision/queue/{user_id}")
def revision_queue(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=25),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Spaced-repetition revision queue: which topics to review now and why."""
    require_same_user_or_admin(user_id, current_user)

    topics = db.query(TopicPerformance).filter(TopicPerformance.user_id == user_id).all()
    queue = build_revision_queue(topics, limit=limit)

    return {
        "user_id": user_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize_queue(queue),
        "queue": queue,
    }


@router.get("/analytics/{user_id}")
def analytics(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    return get_user_analytics(db, user_id)


@router.get("/dashboard/{user_id}")
def get_dashboard_data(
    user_id: str,
    include_analytics: bool = True,
    include_leaderboard: bool = True,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)

    user = get_or_create_progress(db, user_id)
    class_level = profile_learning_context(db, user_id).get("class_level", "")

    # The compact dashboard still computes/persists every metric its UI uses;
    # only unused evidence assembly and the global leaderboard are skipped.
    metrics_changed = False
    if not include_analytics:
        metrics = calculate_cognitive_metrics(db, user_id, persist=False, user=user)
        values = (metrics['focus_score'], metrics['consistency_index'], metrics['learning_efficiency'])
        metrics_changed = (user.focus_score, user.consistency_index, user.learning_efficiency) != values
        user.focus_score, user.consistency_index, user.learning_efficiency = values
    tests = (
        db.query(TestHistory)
        .filter(TestHistory.user_id == user_id)
        .order_by(TestHistory.id.desc())
        .limit(50)
        .all()
    )

    result = {
        "progress": progress_payload(user),
        "sessions": [format_test_session(test, class_level) for test in tests],
        "analytics": get_user_analytics(db, user_id) if include_analytics else None,
        "leaderboard": build_leaderboard(db, current_user) if include_leaderboard else None,
    }

    if metrics_changed:
        db.commit()
    return result
