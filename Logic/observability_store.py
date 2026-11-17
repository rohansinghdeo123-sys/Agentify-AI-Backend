"""Durable observability persistence for Ops and coach traces."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal
from models import ModelToolTrace, ObservabilityEvent

logger = logging.getLogger("ai_educator.observability_store")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


def _event_summary(data: Dict[str, Any]) -> str:
    for key in ("message", "title", "detail", "step", "task", "status"):
        value = str(data.get(key) or "").strip()
        if value:
            return value[:500]
    return ""


def _event_to_row(event: Any) -> Dict[str, Any]:
    if hasattr(event, "to_dict"):
        payload = event.to_dict()
    elif isinstance(event, dict):
        payload = event
    else:
        payload = {}

    data = _safe_dict(payload.get("data"))
    return {
        "event_version": _as_int(payload.get("version")),
        "created_at": _parse_timestamp(payload.get("timestamp")),
        "agent_id": str(payload.get("agent_id") or "unknown"),
        "event_type": str(payload.get("event_type") or "event"),
        "severity": str(payload.get("severity") or "info"),
        "session_id": str(payload.get("session_id") or ""),
        "source": "event_bus",
        "summary": _event_summary(data),
        "latency_ms": _as_int(data.get("latency_ms")),
        "estimated_cost_usd": _as_float(
            data.get("estimated_cost_usd")
            or data.get("total_cost_usd")
            or data.get("cost_usd")
        ),
        "data_json": data,
    }


def persist_observability_event(db: Session, event: Any) -> Optional[ObservabilityEvent]:
    row = _event_to_row(event)
    if row["event_version"] > 0:
        existing = (
            db.query(ObservabilityEvent)
            .filter(ObservabilityEvent.event_version == row["event_version"])
            .first()
        )
        if existing:
            return existing
    stored = ObservabilityEvent(**row)
    db.add(stored)
    db.commit()
    db.refresh(stored)
    return stored


def persist_event_from_bus(event: Any) -> None:
    db = SessionLocal()
    try:
        persist_observability_event(db, event)
    except Exception as exc:
        db.rollback()
        logger.warning("Could not persist observability event: %s", exc)
    finally:
        db.close()


def persist_coach_trace(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    turn_id: str,
    observability: Dict[str, Any],
) -> Dict[str, Any]:
    model_calls = list(observability.get("model_calls") or [])
    trace = _safe_dict(observability.get("trace"))
    tools = list(trace.get("tools") or [])
    quality = _safe_dict(observability.get("quality"))

    total_input_tokens = sum(_as_int(call.get("estimated_input_tokens")) for call in model_calls)
    total_output_tokens = sum(_as_int(call.get("estimated_output_tokens")) for call in model_calls)
    total_cost = round(sum(_as_float(call.get("estimated_cost_usd")) for call in model_calls), 8)
    total_latency = _as_int(trace.get("phases_ms", {}).get("delivering")) or sum(
        _as_int(call.get("latency_ms")) for call in model_calls
    )

    db.add(
        ModelToolTrace(
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_type="turn",
            name="coach_turn",
            status="success" if quality.get("passed", True) else "needs_review",
            latency_ms=_as_int(observability.get("latency_ms")) or _as_int(trace.get("latency_ms")) or total_latency,
            estimated_input_tokens=total_input_tokens,
            estimated_output_tokens=total_output_tokens,
            estimated_cost_usd=total_cost,
            metadata_json={
                "query": observability.get("query") or {},
                "retrieval": observability.get("retrieval") or {},
                "plan": observability.get("plan") or {},
                "quality": quality,
                "phases_ms": trace.get("phases_ms") or {},
                "fallbacks": trace.get("fallbacks") or [],
                "memory_layers": trace.get("memory_layers") or [],
                "mastery_signal": observability.get("mastery_signal") or {},
                "growth": observability.get("growth") or {},
            },
        )
    )

    for call in model_calls:
        call_status = str(call.get("status") or "success")
        db.add(
            ModelToolTrace(
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                trace_type="model" if call_status != "skipped" else "route",
                name=str(call.get("role") or "model_call"),
                provider=str(call.get("provider") or ""),
                model=str(call.get("model") or ""),
                status=call_status,
                latency_ms=_as_int(call.get("latency_ms")),
                estimated_input_tokens=_as_int(call.get("estimated_input_tokens")),
                estimated_output_tokens=_as_int(call.get("estimated_output_tokens")),
                estimated_cost_usd=_as_float(call.get("estimated_cost_usd")),
                metadata_json={
                    key: value
                    for key, value in call.items()
                    if key
                    not in {
                        "role",
                        "provider",
                        "model",
                        "status",
                        "latency_ms",
                        "estimated_input_tokens",
                        "estimated_output_tokens",
                        "estimated_cost_usd",
                    }
                },
            )
        )

    for tool in tools:
        db.add(
            ModelToolTrace(
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                trace_type="tool",
                name=str(tool.get("name") or "tool"),
                status="success",
                latency_ms=_as_int(tool.get("latency_ms")),
                metadata_json={key: value for key, value in tool.items() if key != "name"},
            )
        )

    db.commit()

    # Mirror the completed turn into OpenTelemetry (no-op unless configured).
    try:
        from app.telemetry import emit_coach_turn_trace

        emit_coach_turn_trace(
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            observability=observability,
        )
    except Exception as exc:
        logger.warning("Telemetry mirror skipped: %s", exc)

    return {
        "model_calls": len(model_calls),
        "tool_calls": len(tools),
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": total_cost,
    }


_MODEL_CALL_BASE_KEYS = {
    "role",
    "provider",
    "model",
    "status",
    "latency_ms",
    "estimated_input_tokens",
    "estimated_output_tokens",
    "estimated_cost_usd",
}


def persist_agent_trace(
    *,
    agent: str,
    turn_id: str,
    session_id: str = "",
    user_id: str = "",
    status: str = "success",
    latency_ms: int = 0,
    quality: Optional[Dict[str, Any]] = None,
    model_calls: Optional[Iterable[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_type: str = "turn",
    record_dispatch: bool = False,
) -> Dict[str, Any]:
    """Durably trace one legacy-agent run (revision / exam / tutor / planner).

    Gives these surfaces the same cost + quality visibility the coach pipeline
    already has via persist_coach_trace, but without the full agent-runtime
    machinery: one "turn" row plus one row per model call. Opens its own short
    session and never raises, so telemetry can never break a student response.
    """
    calls = list(model_calls or [])
    total_input = sum(_as_int(call.get("estimated_input_tokens")) for call in calls)
    total_output = sum(_as_int(call.get("estimated_output_tokens")) for call in calls)
    total_cost = round(sum(_as_float(call.get("estimated_cost_usd")) for call in calls), 8)

    db = None
    try:
        db = SessionLocal()
        turn_meta = dict(metadata or {})
        if quality is not None:
            turn_meta["quality"] = quality
        db.add(
            ModelToolTrace(
                user_id=user_id or None,
                session_id=session_id or None,
                turn_id=turn_id,
                trace_type=trace_type,
                name=f"{agent}_{trace_type}",
                status=status,
                latency_ms=_as_int(latency_ms),
                estimated_input_tokens=total_input,
                estimated_output_tokens=total_output,
                estimated_cost_usd=total_cost,
                metadata_json=turn_meta,
            )
        )
        if record_dispatch:
            db.add(
                ModelToolTrace(
                    turn_id=turn_id,
                    trace_type="dispatch",
                    name="orchestrator_dispatch",
                    status=status,
                    latency_ms=_as_int(latency_ms),
                )
            )
        for call in calls:
            call_status = str(call.get("status") or "success")
            db.add(
                ModelToolTrace(
                    user_id=user_id or None,
                    session_id=session_id or None,
                    turn_id=turn_id,
                    trace_type="model" if call_status != "skipped" else "route",
                    name=str(call.get("role") or "model_call"),
                    provider=str(call.get("provider") or ""),
                    model=str(call.get("model") or ""),
                    status=call_status,
                    latency_ms=_as_int(call.get("latency_ms")),
                    estimated_input_tokens=_as_int(call.get("estimated_input_tokens")),
                    estimated_output_tokens=_as_int(call.get("estimated_output_tokens")),
                    estimated_cost_usd=_as_float(call.get("estimated_cost_usd")),
                    metadata_json={key: value for key, value in call.items() if key not in _MODEL_CALL_BASE_KEYS},
                )
            )
        db.commit()
        return {"model_calls": len(calls), "estimated_cost_usd": total_cost}
    except Exception as exc:
        if db is not None:
            db.rollback()
        logger.warning("Could not persist agent trace (agent=%s): %s", agent, exc)
        return {}
    finally:
        if db is not None:
            db.close()


def _format_event(event: ObservabilityEvent) -> Dict[str, Any]:
    return {
        "id": event.id,
        "version": event.event_version,
        "timestamp": event.created_at.isoformat() if event.created_at else "",
        "agent_id": event.agent_id,
        "event_type": event.event_type,
        "data": event.data_json or {},
        "session_id": event.session_id or "",
        "severity": event.severity or "info",
        "durable": True,
        "summary": event.summary or "",
        "latency_ms": event.latency_ms or 0,
        "estimated_cost_usd": event.estimated_cost_usd or 0.0,
    }


def get_recent_observability_events(
    db: Session,
    *,
    limit: int = 50,
    agent_id: Optional[str] = None,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = db.query(ObservabilityEvent)
    if agent_id:
        query = query.filter(ObservabilityEvent.agent_id == agent_id)
    if severity:
        query = query.filter(ObservabilityEvent.severity == severity)
    if event_type:
        query = query.filter(ObservabilityEvent.event_type == event_type)
    rows = query.order_by(ObservabilityEvent.event_version.desc(), ObservabilityEvent.id.desc()).limit(limit).all()
    return [_format_event(row) for row in rows]


def get_observability_events_since(db: Session, since_version: int) -> List[Dict[str, Any]]:
    rows = (
        db.query(ObservabilityEvent)
        .filter(ObservabilityEvent.event_version > since_version)
        .order_by(ObservabilityEvent.event_version.asc(), ObservabilityEvent.id.asc())
        .limit(500)
        .all()
    )
    return [_format_event(row) for row in rows]


def get_latest_observability_version(db: Session) -> int:
    return _as_int(db.query(func.max(ObservabilityEvent.event_version)).scalar())


def get_observability_summary(db: Session, *, hours: int = 24) -> Dict[str, Any]:
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    event_query = db.query(ObservabilityEvent).filter(ObservabilityEvent.created_at >= cutoff)
    trace_query = db.query(ModelToolTrace).filter(ModelToolTrace.created_at >= cutoff)

    return {
        "durable": True,
        "window_hours": hours,
        "events": _as_int(event_query.count()),
        "errors": _as_int(event_query.filter(ObservabilityEvent.severity.in_(["error", "critical"])).count()),
        "model_calls": _as_int(trace_query.filter(ModelToolTrace.trace_type == "model").count()),
        "tool_calls": _as_int(trace_query.filter(ModelToolTrace.trace_type == "tool").count()),
        "turns": _as_int(trace_query.filter(ModelToolTrace.trace_type == "turn").count()),
        "estimated_cost_usd": round(
            _as_float(trace_query.with_entities(func.sum(ModelToolTrace.estimated_cost_usd)).scalar()),
            8,
        ),
        "avg_model_latency_ms": round(
            _as_float(
                trace_query.filter(ModelToolTrace.trace_type == "model")
                .with_entities(func.avg(ModelToolTrace.latency_ms))
                .scalar()
            ),
            1,
        ),
        "avg_turn_latency_ms": round(
            _as_float(
                trace_query.filter(ModelToolTrace.trace_type == "turn")
                .with_entities(func.avg(ModelToolTrace.latency_ms))
                .scalar()
            ),
            1,
        ),
    }


def persist_many_events(db: Session, events: Iterable[Any]) -> int:
    stored = 0
    for event in events:
        if persist_observability_event(db, event):
            stored += 1
    return stored
