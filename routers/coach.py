"""Personal AI Coach API: bootstrap, chat (sync + SSE), conversations, cycles."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, Generator, Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.request_models import CoachConversationPatch
from app.security import (
    enforce_user_quota,
    is_backend_admin,
    require_same_user_or_admin,
    session_id_belongs_to_user,
    verify_firebase_user,
)
from app.serializers import (
    apply_conversation_patch,
    group_conversation_rows,
    interaction_session_id,
    serialize_coach_conversation,
    serialize_coach_memory,
    serialize_coach_profile,
    serialize_daily_signal,
    session_id_from_conversation_id,
)
from database import SessionLocal, get_db
from Logic.agents.coach_agent import (
    coach_agent,
    coach_agent_stream,
    get_or_create_coach,
    run_daily_learning_cycle,
)
from Logic.analytics_engine import get_user_analytics
from Logic.autonomous_study_loop import (
    PlanningChapterNotFoundError,
    run_autonomous_study_loop,
)
from Logic.planning.curriculum_registry import resolve_planning_curriculum
from models import AICoachDailySignal, AICoachMemory
from schemas import (
    AutonomousStudyRequest,
    AutonomousStudyResponse,
    CoachBootstrapRequest,
    CoachChatRequest,
    CoachDailySignalResponse,
    CoachDashboardResponse,
    CoachMemoryResponse,
    CoachProfileResponse,
)
from services.coach_service import conversation_rows_for_user
from services.profile_service import profile_learning_context

router = APIRouter(tags=["coach"])

logger = logging.getLogger("ai_educator.routers.coach")


async def _run_sse_on_single_thread(
    make_generator: Callable[[], Iterator[str]],
) -> Generator[str, None, None]:
    """Drive a sync SSE generator from one dedicated worker thread.

    Starlette iterates sync generators across its threadpool, so successive
    ``next()`` calls can run on different threads. The coach turn tracks
    per-turn cost, budget, and observability via thread-local state in the
    model and tool gateways; running the whole turn on a single thread keeps
    that state coherent and keeps the request-scoped DB session on one thread.

    If the client disconnects mid-stream the worker simply runs the turn to
    completion (persisting normally) — one extra turn of compute, no lost data.
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    done = object()

    def worker() -> None:
        try:
            for frame in make_generator():
                loop.call_soon_threadsafe(queue.put_nowait, frame)
        except BaseException as exc:  # surface to the response, don't swallow
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, name="coach-sse-turn", daemon=True).start()

    while True:
        item = await queue.get()
        if item is done:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


class CoachTurnRequest:
    """Adapter exposing a CoachChatRequest as the attribute bag the agent expects."""

    def __init__(self, payload: CoachChatRequest, learner_profile: Dict[str, str] | None = None):
        self.user_id = payload.user_id
        self.question = (payload.original_message or payload.message).strip()
        self.raw_message = payload.message
        self.original_message = payload.original_message
        self.grounding_context_prompt = payload.grounding_context_prompt
        self.section_id = payload.section_id or payload.topic or payload.subject or "general"
        self.session_id = payload.session_id or f"coach-{payload.user_id}"
        self.mode = "coach"
        self.intent = payload.intent
        self.difficulty = "medium"
        self.subject = payload.subject
        self.chapter = payload.chapter
        self.topic = payload.topic
        self.mentor_directive = payload.mentor_directive
        self.system_guardrail = payload.system_guardrail
        self.strict_grounding = payload.strict_grounding
        self.retrieval_required = payload.retrieval_required
        self.fallback_to_general_knowledge = payload.fallback_to_general_knowledge
        self.required_not_found_response = payload.required_not_found_response
        self.student_state = payload.student_state
        self.adaptive_strategy = payload.adaptive_strategy
        self.learning_context = {
            **payload.learning_context,
            **(learner_profile or {}),
        }
        if (
            str(payload.learning_context.get("catalog_source") or "").strip().lower()
            == "planning_manifest"
            and str(payload.learning_context.get("class_level") or "").strip()
        ):
            # A registered Planning handoff carries an exact curriculum class;
            # retain it while keeping profile-owned identity fields authoritative.
            self.learning_context["class_level"] = str(
                payload.learning_context["class_level"]
            ).strip()
        self.attachments = [item.model_dump() for item in payload.attachments]
        self.direct_answer = payload.direct_answer
        self.socratic_mode = payload.socratic_mode


@router.post("/coach/bootstrap", response_model=CoachProfileResponse)
def coach_bootstrap(
    payload: CoachBootstrapRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(payload.user_id, current_user)

    coach = get_or_create_coach(
        db=db,
        user_id=payload.user_id,
        student_display_name=payload.student_display_name,
        preferred_subjects=payload.preferred_subjects,
        target_exam=payload.target_exam,
        target_exam_date=payload.target_exam_date,
    )

    return CoachProfileResponse(**serialize_coach_profile(coach))


@router.get("/coach/conversations/{user_id}")
def coach_conversations(
    user_id: str,
    include_archived: bool = Query(default=True),
    limit: int = Query(default=40, ge=1, le=80),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    coach = get_or_create_coach(db=db, user_id=user_id)
    rows = conversation_rows_for_user(db, coach, user_id)
    grouped = group_conversation_rows(rows)
    conversations = [
        serialize_coach_conversation(session_id, session_rows)
        for session_id, session_rows in grouped.items()
    ]
    if not include_archived:
        conversations = [item for item in conversations if not item.get("archived")]
    conversations.sort(key=lambda item: (bool(item.get("pinned")), item.get("updatedAt") or ""), reverse=True)
    return {
        "user_id": user_id,
        "conversations": conversations[:limit],
    }


@router.get("/coach/conversations/{user_id}/{conversation_id}")
def coach_conversation_detail(
    user_id: str,
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    session_id = session_id_from_conversation_id(user_id, conversation_id)
    if not is_backend_admin(current_user) and not session_id_belongs_to_user(session_id, user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed for this conversation")

    coach = get_or_create_coach(db=db, user_id=user_id)
    rows = [
        row for row in conversation_rows_for_user(db, coach, user_id, limit=800)
        if interaction_session_id(row) == session_id
    ]
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    return {
        "user_id": user_id,
        "conversation": serialize_coach_conversation(session_id, rows),
    }


@router.patch("/coach/conversations/{user_id}/{conversation_id}")
def update_coach_conversation(
    user_id: str,
    conversation_id: str,
    patch: CoachConversationPatch,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    session_id = session_id_from_conversation_id(user_id, conversation_id)
    if not is_backend_admin(current_user) and not session_id_belongs_to_user(session_id, user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed for this conversation")

    coach = get_or_create_coach(db=db, user_id=user_id)
    rows = [
        row for row in conversation_rows_for_user(db, coach, user_id, limit=800)
        if interaction_session_id(row) == session_id
    ]
    if not rows:
        return {
            "user_id": user_id,
            "conversation": None,
            "status": "no_saved_messages",
        }
    for row in rows:
        apply_conversation_patch(row, patch)
    db.commit()
    return {
        "user_id": user_id,
        "conversation": serialize_coach_conversation(session_id, rows),
        "status": "updated",
    }


@router.delete("/coach/conversations/{user_id}/{conversation_id}")
def delete_coach_conversation(
    user_id: str,
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    session_id = session_id_from_conversation_id(user_id, conversation_id)
    if not is_backend_admin(current_user) and not session_id_belongs_to_user(session_id, user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed for this conversation")

    coach = get_or_create_coach(db=db, user_id=user_id)
    rows = [
        row for row in conversation_rows_for_user(db, coach, user_id, limit=800)
        if interaction_session_id(row) == session_id
    ]
    deleted = len(rows)
    for row in rows:
        db.delete(row)
    db.commit()
    return {
        "user_id": user_id,
        "session_id": session_id,
        "deleted": deleted,
        "status": "deleted",
    }


@router.get("/coach/{user_id}", response_model=CoachDashboardResponse)
def coach_dashboard(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)

    coach = get_or_create_coach(db=db, user_id=user_id)

    memories = (
        db.query(AICoachMemory)
        .filter(AICoachMemory.coach_id == coach.coach_id)
        .order_by(AICoachMemory.importance.desc(), AICoachMemory.updated_at.desc())
        .limit(8)
        .all()
    )

    daily_signal = (
        db.query(AICoachDailySignal)
        .filter(AICoachDailySignal.coach_id == coach.coach_id)
        .order_by(AICoachDailySignal.id.desc())
        .first()
    )

    try:
        analytics_snapshot = get_user_analytics(db, user_id)
    except Exception:
        analytics_snapshot = {}

    return CoachDashboardResponse(
        profile=CoachProfileResponse(**serialize_coach_profile(coach)),
        memories=[CoachMemoryResponse(**serialize_coach_memory(memory)) for memory in memories],
        daily_signal=(
            CoachDailySignalResponse(**serialize_daily_signal(daily_signal))
            if daily_signal
            else None
        ),
        analytics_snapshot=analytics_snapshot,
    )


@router.post("/coach/chat")
def coach_chat(
    payload: CoachChatRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(payload.user_id, current_user)
    enforce_user_quota(payload.user_id, "coach")

    learner_profile = profile_learning_context(db, payload.user_id)
    result = coach_agent(CoachTurnRequest(payload, learner_profile), db=db)
    return result


@router.post("/coach/chat/stream")
async def coach_chat_stream(
    payload: CoachChatRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """
    Stream the coach's reply via Server-Sent Events (SSE).
    The frontend will receive tokens one by one and display them
    as they arrive, creating a real-time typing effect.
    """
    require_same_user_or_admin(payload.user_id, current_user)
    enforce_user_quota(payload.user_id, "coach")

    learner_profile = profile_learning_context(db, payload.user_id)
    coach_request = CoachTurnRequest(payload, learner_profile)

    def make_event_stream() -> Iterator[str]:
        # The generator owns its session: it is opened only once streaming
        # starts and is guaranteed to close even if the client disconnects
        # mid-stream, so long turns cannot pin request-scoped pool slots. It is
        # created and consumed entirely on the single worker thread below, so
        # the session is never touched from more than one thread.
        db = SessionLocal()

        def event_stream() -> Iterator[str]:
            try:
                for token in coach_agent_stream(coach_request, db=db):
                    # coach_agent_stream already returns complete SSE frames.
                    yield token
            finally:
                try:
                    db.rollback()
                except Exception:
                    pass
                db.close()

        return event_stream()

    return StreamingResponse(
        _run_sse_on_single_thread(make_event_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/coach/daily-learning/{user_id}", response_model=CoachDailySignalResponse)
def coach_daily_learning(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)

    signal = run_daily_learning_cycle(db=db, user_id=user_id)
    return CoachDailySignalResponse(**serialize_daily_signal(signal))


@router.post("/coach/autonomous-study/{user_id}", response_model=AutonomousStudyResponse)
def coach_autonomous_study(
    user_id: str,
    payload: AutonomousStudyRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    require_same_user_or_admin(user_id, current_user)
    registered_curriculum = resolve_planning_curriculum(
        chapter_ref=payload.current_chapter,
        subject=payload.subject or None,
        class_level=payload.class_level or None,
    )
    if registered_curriculum is None:
        enforce_user_quota(user_id, "coach")

    try:
        mission = run_autonomous_study_loop(
            db=db,
            user_id=user_id,
            current_chapter=payload.current_chapter,
            subject=payload.subject,
            chapter_proficiency=payload.chapter_proficiency,
            class_level=payload.class_level,
            session_duration_minutes=payload.session_duration_minutes,
        )
    except PlanningChapterNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return AutonomousStudyResponse(**mission)
