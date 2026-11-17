"""Agent HQ's read-only view of recorded AgentifyAI execution.

Use separate units of measurement for platform turns and internal runtime
steps. Aggregate approved fields only. Student prompts, answers, identifiers,
documents, exception strings and raw trace payloads never cross this boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import ceil
import re

from sqlalchemy import case, func, literal, select, union_all
from sqlalchemy.orm import Session

from Logic.agent_event_bus import event_bus
from Logic.agent_runtime.contracts import AGENT_ROLES
from Logic.agentic_orchestrator import get_agent_registry
from models import (
    AgentRuntimeHandoff,
    AgentRuntimeMessage,
    AgentRuntimeRun,
    AgentRuntimeStep,
    ModelToolTrace,
)


_STALE_AFTER = timedelta(minutes=30)
_ACTIVE_FOR = timedelta(minutes=5)
_LATENCY_SAMPLE_LIMIT = 1500
_ACTIVITY_LIMIT = 50
_SUCCESS = ("success", "completed", "done")
_FAILURE = ("failed", "error", "timeout", "timed_out")
_REVIEW = ("needs_review", "fallback")
_ACTIVE = ("running", "active", "queued", "waiting")
_KNOWN_STATES = set(_SUCCESS + _FAILURE + _REVIEW + _ACTIVE + ("skipped", "cancelled", "requested", "accepted"))
_TOP_LEVEL_PROFILES = {
    "orchestrator": ("Operations", "Concise, with priorities and cross-team dependencies."),
    "coach": ("Learning", "Practical and focused on study workflows."),
    "tutor": ("Learning", "Clear and attentive to student-facing explanations."),
    "planner": ("Learning", "Sequenced and explicit about prerequisites."),
    "revision": ("Learning", "Compact and organized around recall."),
    "exam": ("Assessment", "Precise about question generation and checks."),
}
# These roles execute in the durable coach pipeline. The contract also
# declares reserved roles whose declarations alone do not make them agents.
_COACH_ROLES = {
    "lead_coach_orchestrator": ("Coach Orchestrator", "Operations", "Coordinates the coach pipeline and its handoffs."),
    "intent_profiler": ("Intent Profiler", "Learning", "Identifies the learner's intent and required response depth."),
    "conversation_responder": ("Conversation Responder", "Learning", "Handles light conversational turns in the coach pipeline."),
    "response_planner": ("Response Planner", "Learning", "Selects a response structure and teaching strategy."),
    "context_retriever": ("Context Retriever", "Knowledge", "Retrieves approved evidence and checks material coverage."),
    "tool_gateway": ("Tool Gateway", "Operations", "Routes the coach's supported tool calls."),
    "tutor_model": ("Tutor Model", "Learning", "Drafts teaching answers from the selected evidence."),
    "answer_reviewer": ("Answer Reviewer", "Quality", "Reviews draft answers and records corrections."),
    "quality_verifier": ("Quality Verifier", "Quality", "Checks grounding, response quality and repair decisions."),
    "memory_mastery_engine": ("Memory & Mastery Engine", "Learning", "Prepares scoped learning-memory and mastery signals."),
}
_OPERATIONS = {
    "study_coach_turn": "Coach workflow",
    "request_received": "Receive coach request",
    "intent_profiled": "Profile intent",
    "response_planned": "Plan response",
    "context_prepared": "Prepare evidence",
    "orchestration_planned": "Plan handoffs",
    "tools_selected": "Select tools",
    "conversation_answered": "Prepare conversational reply",
    "answer_drafted": "Draft answer",
    "answer_reviewed": "Review answer",
    "answer_verified": "Verify answer",
    "answer_repair_checked": "Check answer repair",
    "memory_signal_prepared": "Prepare mastery signal",
    "stream_failed": "Coach stream failed",
}
_MESSAGE_TYPES = {
    "profile_result": "Intent classification",
    "conversation_result": "Conversation result",
    "response_plan": "Response plan",
    "context_result": "Retrieval finding",
    "tool_results": "Tool results",
    "draft_result": "Draft result",
    "review_result": "Review finding",
    "verification_result": "Verification result",
    "memory_update": "Memory update",
    "handoff_request": "Delegation request",
    "handoff_result": "Delegation result",
    "status": "Status update",
    "error": "Failure",
}


def registry() -> list[dict]:
    """Discover public entry points from the actual orchestration registry."""
    agents = []
    for entry in get_agent_registry():
        agent_id = entry["agent_id"]
        team, style = _TOP_LEVEL_PROFILES.get(agent_id, ("Operations", "Evidence-led and direct."))
        agents.append({
            "id": agent_id,
            "name": entry["display_name"],
            "role": entry["role"],
            "team": team,
            "manager_id": None if agent_id == "orchestrator" else "orchestrator",
            "description": entry["responsibility"],
            "capabilities": entry["intents"],
            "tools": entry["tools"],
            "style": style,
            "kind": "registered",
            "version": "agentic-control-plane-v1",
        })
    if any(agent["id"] == "coach" for agent in agents):
        for agent_id, (name, team, description) in _COACH_ROLES.items():
            if agent_id in AGENT_ROLES:
                agents.append({
                    "id": agent_id,
                    "name": name,
                    "role": "Coach runtime specialist",
                    "team": team,
                    "manager_id": "coach" if agent_id == "lead_coach_orchestrator" else "lead_coach_orchestrator",
                    "description": description,
                    "capabilities": [description],
                    "tools": [],
                    "style": "Evidence-led, with findings scoped to this workflow step.",
                    "kind": "runtime_role",
                    "version": "agent_runtime_v1",
                })
    return agents


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc).isoformat()


def _trace_ref(value: str | None) -> str | None:
    # These prefixes/lengths are generated by the platform's run factories.
    if isinstance(value, str) and re.fullmatch(r"(run|turn)_[a-f0-9]{12}", value):
        return value
    return None


def _state(value: str | None) -> str:
    return value if isinstance(value, str) and value in _KNOWN_STATES else "unknown"


def _records(registry_rows: list[dict], start: datetime, end: datetime):
    # This projection is also the data-minimization boundary. Keep summaries,
    # request bodies, error text, and actor/session identifiers out of it.
    trace = ModelToolTrace
    run = AgentRuntimeRun
    step = AgentRuntimeStep
    turn_agents = {f'{agent["id"]}_turn': agent["id"] for agent in registry_rows if agent["kind"] == "registered"}
    role_ids = [agent["id"] for agent in registry_rows if agent["kind"] == "runtime_role"]
    turns = select(
        literal("turn").label("kind"),
        trace.id.label("id"),
        case(turn_agents, value=trace.name).label("agent_id"),
        trace.name.label("operation"),
        trace.status.label("status"),
        trace.created_at.label("created_at"),
        trace.latency_ms.label("latency_ms"),
        trace.turn_id.label("run_id"),
        (func.coalesce(trace.estimated_input_tokens, 0) + func.coalesce(trace.estimated_output_tokens, 0)).label("tokens"),
        trace.estimated_cost_usd.label("cost"),
    ).where(trace.trace_type == "turn", trace.name.in_(turn_agents), trace.created_at >= start, trace.created_at <= end)
    dispatch = select(
        literal("dispatch"), trace.id, literal("orchestrator"), literal("dispatch"),
        trace.status, trace.created_at, trace.latency_ms, trace.turn_id, literal(0), literal(0.0),
    ).where(trace.trace_type == "dispatch", trace.name == "orchestrator_dispatch", trace.created_at >= start, trace.created_at <= end)
    workflows = select(
        literal("runtime_run"), run.id, run.lead_agent, run.workflow_name,
        run.status, run.started_at, run.latency_ms, run.run_id, literal(0), literal(0.0),
    ).where(run.lead_agent.in_(role_ids), run.started_at >= start, run.started_at <= end)
    steps = select(
        literal("runtime_step"), step.id, step.agent_name, step.step_name,
        step.status, step.started_at, step.latency_ms, step.run_id, literal(0), literal(0.0),
    ).where(
        step.agent_name.in_([role_id for role_id in role_ids if role_id != "lead_coach_orchestrator"]),
        step.started_at >= start, step.started_at <= end,
    )
    return union_all(turns, dispatch, workflows, steps).subquery()


def _percentile(sorted_values: list[int], fraction: float) -> int | None:
    return sorted_values[ceil(len(sorted_values) * fraction) - 1] if sorted_values else None


def _measure(rows, latencies: list[int], *, active: int = 0) -> dict:
    completed = sum(row.count for row in rows if row.status in _SUCCESS)
    failed = sum(row.count for row in rows if row.status in _FAILURE)
    needs_review = sum(row.count for row in rows if row.status in _REVIEW)
    count = sum(row.count for row in rows)
    tokens = sum(row.tokens or 0 for row in rows)
    cost = sum(row.cost or 0 for row in rows)
    latency_count = sum(row.latency_count for row in rows)
    latency_total = sum(row.latency_total or 0 for row in rows)
    sample = sorted(latencies)
    return {
        "recorded": count,
        "completed": completed,
        "failed": failed,
        "needs_review": needs_review,
        "active": active,
        "avg_latency_ms": round(latency_total / latency_count, 1) if latency_count else None,
        "p50_latency_ms": _percentile(sample, 0.50),
        "p95_latency_ms": _percentile(sample, 0.95),
        "p99_latency_ms": _percentile(sample, 0.99),
        "latency_sample_size": len(sample),
        "estimated_tokens": tokens or None,
        "estimated_cost_usd": round(cost, 8) if cost > 0 else None,
        "success_rate": round(completed / (completed + failed + needs_review) * 100, 1)
        if completed + failed + needs_review else None,
    }


def _operation(kind: str, code: str) -> str:
    if kind == "turn":
        return "Platform turn"
    if kind == "dispatch":
        return "Dispatch request"
    return _OPERATIONS.get(code, "Runtime operation")


def _activity(row) -> dict:
    return {
        "id": f"{row.kind}-{row.id}",
        "agent_id": row.agent_id,
        "operation": _operation(row.kind, row.operation),
        "status": _state(row.status),
        "created_at": _iso(row.created_at),
        "latency_ms": row.latency_ms if row.latency_ms and row.latency_ms > 0 else None,
        "run_id": _trace_ref(row.run_id),
        "source": row.kind,
        "error_category": "timeout" if row.status in ("timeout", "timed_out")
        else "execution" if row.status in _FAILURE else None,
    }


def _live_status(agent_ids: set[str], now: datetime) -> dict[str, dict]:
    """In-process event data supplements durable evidence for live activity."""
    current = {}
    observed_runs = set()
    for event in event_bus.get_recent_events(limit=1000):
        agent_id = event["agent_id"]
        kind = event["event_type"]
        if agent_id not in agent_ids or kind not in {"task_start", "task_complete", "error"}:
            continue
        try:
            observed = datetime.fromisoformat(event["timestamp"])
        except (TypeError, ValueError):
            continue
        # The legacy bus emits naive local timestamps. Interpret those using
        # host timezone then convert to the persistence layer's naive UTC.
        observed = observed.astimezone(timezone.utc).replace(tzinfo=None)
        if observed > now or now - observed > _ACTIVE_FOR:
            continue
        payload = event.get("data") or {}
        status = "running" if kind == "task_start" else "error" if kind == "error" else _state(payload.get("status") or "completed")
        item = current.setdefault(agent_id, {"status": status, "timestamp": observed, "active": 0})
        run_ref = _trace_ref(payload.get("run_id"))
        if run_ref and (agent_id, run_ref) not in observed_runs:
            observed_runs.add((agent_id, run_ref))
            item["active"] += kind == "task_start"
    return current


def _recommendation(agent_id: str, metrics: dict, days: int) -> dict | None:
    if metrics["failed"] >= 3:
        topic, title = "failures", "Investigate recurring execution failures"
        evidence = f'{metrics["failed"]} recorded failures across {metrics["recorded"]} observations in the last {days} days.'
        action = "Inspect failed execution records and compare the affected workflow's status and timing."
        benefit = "Identify repeatable failure conditions and prioritize reliability work."
    elif metrics["needs_review"] >= 3:
        topic, title = "review", "Review repeated quality flags or fallbacks"
        evidence = f'{metrics["needs_review"]} operations were flagged for review or used a fallback in the last {days} days.'
        action = "Compare the recorded workflow outcomes and find which stage most often requires a fallback or review."
        benefit = "Identify where additional source coverage or quality checks are warranted."
    elif metrics["latency_sample_size"] >= 10 and metrics["p95_latency_ms"] >= 5000:
        topic, title = "latency", "Inspect slow executions"
        evidence = f'p95 was {metrics["p95_latency_ms"]:,} ms across {metrics["latency_sample_size"]} recent measured executions.'
        action = "Review the slowest traces and compare model and tool spans before changing the workflow."
        benefit = "Find the stage contributing most to slow responses."
    else:
        return None
    return {
        "id": f"{agent_id}-{topic}",
        "agent_id": agent_id,
        "title": title,
        "summary": evidence,
        "evidence": [evidence],
        "proposed_action": action,
        "expected_benefit": benefit,
        "confidence": "Observed operational signal",
        "risks": "Recorded events show the symptom. Its cause requires trace review.",
        "estimated_effort": None,
    }


def _collaboration(db: Session, ids: set[str], start: datetime, end: datetime, *, run_id: str | None = None):
    handoff = AgentRuntimeHandoff
    messages = AgentRuntimeMessage
    recorded_handoffs = select(
        literal("handoff").label("kind"), handoff.id.label("id"), handoff.run_id.label("run_id"),
        handoff.created_at.label("created_at"), handoff.from_agent.label("from_agent"),
        handoff.to_agent.label("to_agent"), handoff.status.label("outcome"),
    ).where(
        handoff.created_at >= start, handoff.created_at <= end,
        handoff.from_agent.in_(ids), handoff.to_agent.in_(ids), handoff.from_agent != handoff.to_agent,
    )
    recorded_messages = select(
        literal("message"), messages.id, messages.run_id, messages.created_at, messages.sender_agent,
        messages.receiver_agent, messages.message_type,
    ).where(
        messages.created_at >= start, messages.created_at <= end,
        messages.sender_agent.in_(ids), messages.receiver_agent.in_(ids), messages.sender_agent != messages.receiver_agent,
    )
    if run_id is not None:
        recorded_handoffs = recorded_handoffs.where(handoff.run_id == run_id)
        recorded_messages = recorded_messages.where(messages.run_id == run_id)
    records = union_all(recorded_handoffs, recorded_messages).subquery()
    recent = db.execute(select(records).order_by(records.c.created_at.desc(), records.c.id.desc()).limit(_ACTIVITY_LIMIT))
    items = [
        {
            "id": f"{row.kind}-{row.id}",
            "kind": row.kind,
            "run_id": _trace_ref(row.run_id),
            "created_at": _iso(row.created_at),
            "from_agent": row.from_agent,
            "to_agent": row.to_agent,
            "status": _state(row.outcome) if row.kind == "handoff" else "recorded",
            "operation": "Runtime handoff" if row.kind == "handoff" else _MESSAGE_TYPES.get(row.outcome, "Runtime message"),
        }
        for row in recent
    ]
    totals = dict(db.execute(select(records.c.kind, func.count()).group_by(records.c.kind)).tuples().all())
    return items, {"messages": totals.get("message", 0), "handoffs": totals.get("handoff", 0)}


def _usage(db: Session, start: datetime, end: datetime, *, run_id: str | None = None) -> list[dict]:
    trace = ModelToolTrace
    query = select(
        trace.provider, trace.model,
        func.count().label("calls"),
        func.sum(case((trace.status.in_(_FAILURE), 1), else_=0)).label("failures"),
        func.avg(case((trace.latency_ms > 0, trace.latency_ms))).label("average"),
        func.sum(func.coalesce(trace.estimated_input_tokens, 0) + func.coalesce(trace.estimated_output_tokens, 0)).label("tokens"),
        func.sum(trace.estimated_cost_usd).label("cost"),
    ).where(
        trace.trace_type == "model", trace.created_at >= start, trace.created_at <= end,
    )
    if run_id is not None:
        query = query.where(trace.turn_id == run_id)
    rows = db.execute(query.group_by(trace.provider, trace.model).order_by(func.count().desc()).limit(20))
    return [
        {
            "provider": row.provider if row.provider and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", row.provider) else None,
            "model": row.model if row.model and re.fullmatch(r"[a-zA-Z0-9_./:-]{1,120}", row.model) else None,
            "calls": row.calls,
            "failures": row.failures or 0,
            "avg_latency_ms": round(float(row.average), 1) if row.average is not None else None,
            "estimated_tokens": row.tokens or None,
            "estimated_cost_usd": round(row.cost, 8) if row.cost and row.cost > 0 else None,
        }
        for row in rows
    ]


def snapshot(
    db: Session,
    *,
    days: int,
    ending_at: datetime | None = None,
    include_live: bool = True,
) -> dict:
    """Current or historical HQ evidence, with explicit sampling and gaps."""
    if not 1 <= days <= 90:
        raise ValueError("Select a reporting period between 1 and 90 days")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    end = ending_at or now
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    agents = registry()
    ids = {agent["id"] for agent in agents}
    records = _records(agents, start, end)
    data = records.c
    latency = case((data.latency_ms > 0, data.latency_ms))
    grouped = db.execute(select(
        data.kind, data.agent_id, data.status,
        func.count().label("count"),
        func.count(latency).label("latency_count"),
        func.sum(latency).label("latency_total"),
        func.sum(data.tokens).label("tokens"),
        func.sum(data.cost).label("cost"),
        func.min(data.created_at).label("first"),
        func.max(data.created_at).label("last"),
    ).group_by(data.kind, data.agent_id, data.status)).all()
    recent = db.execute(select(records).order_by(data.created_at.desc(), data.id.desc(), data.kind).limit(_LATENCY_SAMPLE_LIMIT)).all()
    by_agent = defaultdict(list)
    samples = defaultdict(list)
    activity_by_agent = defaultdict(list)
    incidents_by_agent = defaultdict(list)
    latest = {}
    for row in grouped:
        by_agent[row.agent_id].append(row)
    for row in recent:
        latest.setdefault(row.agent_id, row)
        if row.latency_ms and row.latency_ms > 0:
            samples[row.agent_id].append(row.latency_ms)
        if len(activity_by_agent[row.agent_id]) < 5:
            activity_by_agent[row.agent_id].append(_activity(row))
        if row.status in _FAILURE and len(incidents_by_agent[row.agent_id]) < 5:
            incidents_by_agent[row.agent_id].append(_activity(row))

    live = _live_status(ids, now) if include_live and ending_at is None else {}
    active_by_agent = dict(db.execute(select(data.agent_id, func.count()).where(
        data.status.in_(_ACTIVE), data.created_at >= end - _ACTIVE_FOR,
    ).group_by(data.agent_id)).tuples().all()) if include_live and ending_at is None else {}
    recommendations = []
    for agent in agents:
        agent_id = agent["id"]
        rows = by_agent[agent_id]
        observed = live.get(agent_id)
        last = max((row.last for row in rows), default=None)
        last_status = latest[agent_id].status if agent_id in latest else None
        if observed and (last is None or observed["timestamp"] >= last):
            last = observed["timestamp"]
            last_status = observed["status"]
        active = max(active_by_agent.get(agent_id, 0), observed["active"] if observed else 0)
        status = "unknown"
        if active:
            status = "working"
        elif include_live and ending_at is None and last and now - last <= _STALE_AFTER:
            if last_status in _FAILURE:
                status = "error"
            elif last_status in _REVIEW:
                status = "degraded"
            elif last_status in _SUCCESS:
                status = "available"
        metrics = _measure(rows, samples[agent_id], active=active)
        agent.update(
            status=status,
            last_activity=_iso(last),
            current_task="Platform request" if status == "working" else None,
            metrics=metrics,
        )
        item = _recommendation(agent_id, metrics, days)
        if item:
            recommendations.append(item)

    turn_rows = [row for row in grouped if row.kind == "turn"]
    turn_samples = [row.latency_ms for row in recent if row.kind == "turn" and row.latency_ms and row.latency_ms > 0]
    metrics = _measure(turn_rows, turn_samples, active=sum(agent["metrics"]["active"] for agent in agents if agent["kind"] == "registered"))
    handoffs, collaboration = _collaboration(db, ids, start, end)
    age = max((row.last for row in grouped), default=None)
    date = func.date(data.created_at)
    trends = db.execute(select(
        date.label("date"), func.count().label("recorded"),
        func.sum(case((data.status.in_(_SUCCESS), 1), else_=0)).label("completed"),
        func.sum(case((data.status.in_(_FAILURE), 1), else_=0)).label("failed"),
    ).where(data.kind == "turn").group_by(date).order_by(date)).all()

    return {
        "generated_at": _iso(now),
        "days": days,
        "starts_at": _iso(start),
        "ends_at": _iso(end),
        "agents": agents,
        "metrics": metrics,
        "activity": [_activity(row) for row in recent[:_ACTIVITY_LIMIT]],
        "incidents": [_activity(row) for row in recent if row.status in _FAILURE][:_ACTIVITY_LIMIT],
        "activity_by_agent": dict(activity_by_agent),
        "incidents_by_agent": dict(incidents_by_agent),
        "handoffs": handoffs,
        "collaboration": collaboration,
        "recommendations": recommendations,
        "models": _usage(db, start, end),
        "trends": [{"date": str(row.date), "recorded": row.recorded, "completed": row.completed, "failed": row.failed} for row in trends],
        "coverage": {
            "sources": ["ModelToolTrace", "AgentRuntimeRun", "AgentRuntimeStep", "AgentRuntimeMessage", "AgentRuntimeHandoff"]
            + (["local event bus"] if include_live and ending_at is None else []),
            "first_observation": _iso(min((row.first for row in grouped), default=None)),
            "last_observation": _iso(age),
            "latency_sample_size": len(turn_samples),
            "latency_sample_limit": _LATENCY_SAMPLE_LIMIT,
            "unavailable": ["Queue latency", "Time to first token", "Exact billing", "CPU and memory utilization"],
            "stale": age is None or now - age > _STALE_AFTER,
            "notes": [
                "Organization totals count platform turns. Agent totals may count dispatches, workflows or workflow steps.",
                "Latency percentiles use the recent sample. Means and counts use all matching stored records.",
                "Live worker observations combine persisted coach runs with events from this API process.",
                "Token and cost figures are estimates from recorded model usage.",
            ],
        },
    }


def trace_detail(db: Session, activity_id: str) -> dict | None:
    """Inspect one recorded operation and its associated sanitized evidence."""
    identity = re.fullmatch(r"(turn|dispatch|runtime_run|runtime_step)-([1-9]\d{0,17})", activity_id)
    if not identity:
        return None
    agents = registry()
    start = datetime(2000, 1, 1)
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    records = _records(agents, start, end)
    data = records.c
    row = db.execute(select(records).where(data.kind == identity[1], data.id == int(identity[2]))).first()
    if row is None:
        return None
    result = {"activity": _activity(row), "related_activity": [], "handoffs": [], "models": []}
    run_id = _trace_ref(row.run_id)
    if run_id:
        related = db.execute(select(records).where(
            data.run_id == run_id,
        ).order_by(data.created_at, data.id, data.kind).limit(_ACTIVITY_LIMIT))
        result["related_activity"] = [_activity(item) for item in related]
        result["handoffs"], _ = _collaboration(db, {agent["id"] for agent in agents}, start, end, run_id=run_id)
        result["models"] = _usage(db, start, end, run_id=run_id)
    return result
