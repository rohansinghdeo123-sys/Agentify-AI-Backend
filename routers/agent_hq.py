"""Authenticated executive controls and sanitized Agent HQ read models."""
from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from datetime import date, timedelta
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, or_, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.background import BackgroundTask

from database import SessionLocal, get_db
from models import AgentHQRecommendationDecision, AgentHQRun, AgentHQSchedule
from services.admin_service import record_admin_audit
from services.agent_hq import (
    ACTIVE_STATUSES, MAX_PARTICIPANTS, add_run, event_version, iso, job_queue,
    load_snapshot, next_schedule_run, run_detail, serialize_run, utcnow,
)
from services.agent_hq_access import list_agent_hq_executives, require_agent_hq_executive
from services.agent_hq_evidence import registry

router = APIRouter(prefix="/admin/agent-hq", tags=["agent-hq"],
                   dependencies=[Depends(require_agent_hq_executive)])
_streams: dict[str, int] = {}


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: UUID
    kind: Literal["direct", "meeting", "standup", "report"]
    topic: str = Field(default="", max_length=600)
    participant_ids: list[str] = Field(default_factory=list, max_length=MAX_PARTICIPANTS)
    days: int = Field(default=1, ge=1, le=90)
    ending_on: date | None = None
    parent_run_id: UUID | None = None


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: Literal["accepted", "deferred", "dismissed"]
    note: str = Field(default="", max_length=500)
    days: int = Field(default=1, ge=1, le=90)


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    enabled: bool
    timezone: str = Field(min_length=1, max_length=64)
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)


def _lock_updates(db):
    if db.get_bind().dialect.name == "postgresql":
        db.execute(text("SELECT pg_advisory_xact_lock(874119201)"))


@router.get("/snapshot")
def get_snapshot(days: int = Query(1, ge=1, le=90), db: Session = Depends(get_db)):
    data = deepcopy(load_snapshot(db, days))
    ids = [item["id"] for item in data["recommendations"]]
    decisions = db.query(AgentHQRecommendationDecision).filter(
        AgentHQRecommendationDecision.recommendation_id.in_(ids)
    ).all() if ids else []
    by_id = {item.recommendation_id: item for item in decisions}
    for item in data["recommendations"]:
        decision = by_id.get(item["id"])
        item["status"] = decision.status if decision else "pending"
    active = db.query(AgentHQRun).filter(AgentHQRun.status.in_(ACTIVE_STATUSES)).order_by(AgentHQRun.created_at).limit(10).all()
    data["active_runs"] = [serialize_run(run) for run in active]
    data["executives"] = list_agent_hq_executives()
    return data


@router.get("/runs")
def list_runs(
    limit: int = Query(30, ge=1, le=100),
    before: UUID | None = None,
    kind: Literal["direct", "meeting", "standup", "report"] | None = None,
    search: str | None = Query(None, max_length=120),
    db: Session = Depends(get_db),
):
    query = db.query(AgentHQRun)
    if kind:
        query = query.filter_by(kind=kind)
    if search:
        query = query.filter(AgentHQRun.topic.icontains(search, autoescape=True))
    if before:
        cursor = db.get(AgentHQRun, str(before))
        if not cursor:
            raise HTTPException(404, "Run cursor expired")
        query = query.filter(or_(
            AgentHQRun.created_at < cursor.created_at,
            and_(AgentHQRun.created_at == cursor.created_at, AgentHQRun.id < cursor.id),
        ))
    rows = query.order_by(AgentHQRun.created_at.desc(), AgentHQRun.id.desc()).limit(limit + 1).all()
    return {
        "runs": [serialize_run(run) for run in rows[:limit]],
        "next_cursor": rows[limit - 1].id if len(rows) > limit else None,
    }


@router.get("/traces/{activity_id}")
def get_trace(activity_id: str, db: Session = Depends(get_db)):
    from services.agent_hq_evidence import trace_detail

    detail = trace_detail(db, activity_id) if len(activity_id) <= 64 else None
    if not detail:
        raise HTTPException(404, "Recorded activity not found")
    return detail


@router.get("/runs/{run_id}")
def get_run(run_id: UUID, db: Session = Depends(get_db)):
    run = db.get(AgentHQRun, str(run_id))
    if not run:
        raise HTTPException(404, "Run not found")
    return run_detail(db, run)


@router.post("/runs", status_code=202)
def create_run(payload: RunRequest, request: Request, db: Session = Depends(get_db),
               executive=Depends(require_agent_hq_executive)):
    ids = [item["id"] for item in registry()]
    participants = list(dict.fromkeys(payload.participant_ids))
    if any(item not in ids for item in participants):
        raise HTTPException(422, "Select registered Agent HQ roles")
    if payload.kind == "direct" and len(participants) != 1:
        raise HTTPException(422, "Direct requests require one agent")
    if payload.kind in {"direct", "meeting"} and not payload.topic:
        raise HTTPException(422, "Enter a request or meeting topic")
    if payload.ending_on and (payload.kind != "report" or payload.ending_on > utcnow().date()
                              or payload.ending_on < date(2020, 1, 1)):
        raise HTTPException(422, "Choose a past or current report date on or after January 1, 2020")
    parent_id = str(payload.parent_run_id) if payload.parent_run_id else None
    if parent_id:
        parent = db.get(AgentHQRun, parent_id)
        if not parent or parent.kind != "direct" or payload.kind != "direct" or participants != parent.participant_ids:
            raise HTTPException(422, "A follow-up must address the same agent as the parent request")
        if parent.status not in {"completed", "failed"}:
            raise HTTPException(409, "Wait for the preceding response")
    if not participants:
        participants = ids[:MAX_PARTICIPANTS]
    topic = payload.topic or {"standup": "Daily standup", "report": "Executive report"}[payload.kind]
    uid = executive["uid"]
    request_id = str(payload.request_id)
    _lock_updates(db)
    existing = db.query(AgentHQRun).filter_by(actor_uid=uid, request_id=request_id).first()
    if existing:
        if (existing.kind, existing.topic, existing.participant_ids, existing.days, existing.parent_run_id,
            existing.ending_on) != (
            payload.kind, topic, participants, payload.days, parent_id, payload.ending_on
        ):
            raise HTTPException(409, "Request ID already belongs to a different operation")
        return run_detail(db, existing)
    recent = db.query(AgentHQRun).filter(AgentHQRun.actor_uid == uid, AgentHQRun.created_at >= utcnow() - timedelta(days=1)).count()
    pending = db.query(AgentHQRun).filter(AgentHQRun.actor_uid == uid, AgentHQRun.status.in_(ACTIVE_STATUSES)).count()
    if recent >= 40 or pending >= 2:
        raise HTTPException(429, "Agent HQ request capacity reached. Wait for current work or try later.")
    try:
        run = add_run(db, actor_uid=uid, request_id=request_id, kind=payload.kind, topic=topic,
                      participant_ids=participants, days=payload.days, parent_run_id=parent_id,
                      ending_on=payload.ending_on)
        record_admin_audit(db, current_admin=executive, action="agent_hq_run", target_type="agent_hq_run",
                           target_id=run.id, metadata={"kind": run.kind, "participant_ids": participants}, request=request)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "This request was submitted concurrently. Retry with the same request ID.") from None
    job_queue.wake()
    return run_detail(db, run)


@router.post("/recommendations/{recommendation_id}/decision")
def decide(recommendation_id: str, payload: RecommendationRequest, request: Request,
           db: Session = Depends(get_db), executive=Depends(require_agent_hq_executive)):
    if len(recommendation_id) > 128 or not any(
        item["id"] == recommendation_id for item in load_snapshot(db, payload.days)["recommendations"]
    ):
        raise HTTPException(404, "Recommendation no longer appears in this reporting period")
    _lock_updates(db)
    item = db.get(AgentHQRecommendationDecision, recommendation_id)
    if not item:
        item = AgentHQRecommendationDecision(recommendation_id=recommendation_id)
        db.add(item)
    item.status, item.note, item.actor_uid = payload.status, payload.note, executive["uid"]
    item.updated_at = utcnow()
    record_admin_audit(db, current_admin=executive, action="agent_hq_recommendation_decision",
                       target_type="agent_hq_recommendation", target_id=recommendation_id,
                       metadata={"decision": payload.status}, request=request)
    return {"id": recommendation_id, "status": payload.status}


def _schedule_data(schedule):
    if not schedule:
        return {"enabled": False, "timezone": "Asia/Kolkata", "hour": 9, "minute": 0, "next_run_at": None}
    return {"enabled": schedule.enabled, "timezone": schedule.timezone, "hour": schedule.hour,
            "minute": schedule.minute, "next_run_at": iso(schedule.next_run_at)}


@router.get("/schedule")
def get_schedule(db: Session = Depends(get_db)):
    return _schedule_data(db.get(AgentHQSchedule, 1))


@router.put("/schedule")
def save_schedule(payload: ScheduleRequest, request: Request, db: Session = Depends(get_db),
                  executive=Depends(require_agent_hq_executive)):
    try:
        due = next_schedule_run(payload.timezone, payload.hour, payload.minute, utcnow())
    except (ValueError, ZoneInfoNotFoundError):
        raise HTTPException(422, "Select a valid IANA timezone") from None
    _lock_updates(db)
    schedule = db.get(AgentHQSchedule, 1)
    if not schedule:
        schedule = AgentHQSchedule(id=1)
        db.add(schedule)
    for key, value in payload.model_dump().items():
        setattr(schedule, key, value)
    schedule.next_run_at = due if payload.enabled else None
    schedule.actor_uid = executive["uid"]
    schedule.updated_at = utcnow()
    record_admin_audit(db, current_admin=executive, action="agent_hq_schedule", target_type="agent_hq_schedule",
                       target_id="1", metadata=payload.model_dump(), request=request)
    return _schedule_data(schedule)


def _read_version():
    with SessionLocal() as db:
        return event_version(db)


@router.get("/events")
async def stream_events(request: Request, executive=Depends(require_agent_hq_executive)):
    uid = executive["uid"]
    if _streams.get(uid, 0) >= 2:
        raise HTTPException(429, "Close an existing live connection and retry")
    _streams[uid] = _streams.get(uid, 0) + 1
    released = False

    def release():
        nonlocal released
        if released:
            return
        released = True
        remaining = _streams.get(uid, 1) - 1
        if remaining:
            _streams[uid] = remaining
        else:
            _streams.pop(uid, None)

    async def release_connection():
        release()

    async def updates():
        expires = time.monotonic() + 55
        previous = None
        try:
            while time.monotonic() < expires and not await request.is_disconnected():
                version = await run_in_threadpool(_read_version)
                if version != previous:
                    yield f"event: update\ndata: {json.dumps({'version': version})}\n\n"
                    previous = version
                else:
                    yield ": connected\n\n"
                await asyncio.sleep(3)
        finally:
            release()

    return StreamingResponse(updates(), media_type="text/event-stream", background=BackgroundTask(release_connection),
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
