import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database import Base
from Logic import agentic_orchestrator, observability_store
from Logic.agentic_orchestrator import get_agent_registry
from models import AgentRuntimeHandoff, AgentRuntimeMessage, AgentRuntimeRun, AgentRuntimeStep, ModelToolTrace
from services import agent_hq_evidence as evidence


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_registry_discovers_real_entry_points_and_has_a_valid_hierarchy():
    agents = evidence.registry()
    profiles = {agent["id"]: agent for agent in agents}
    assert {agent["id"] for agent in agents if agent["kind"] == "registered"} == {
        agent["agent_id"] for agent in get_agent_registry()
    }
    assert "context_retriever" in profiles
    assert "answer_reviewer" in profiles
    assert "unknown_agent" not in profiles
    for agent in agents:
        hierarchy = []
        while agent["manager_id"]:
            assert agent["id"] not in hierarchy
            hierarchy.append(agent["id"])
            agent = profiles[agent["manager_id"]]
        assert agent["id"] == "orchestrator"


def test_zero_activity_keeps_missing_metrics_and_current_status_unknown(db):
    result = evidence.snapshot(db, days=90, include_live=False)
    assert result["metrics"]["recorded"] == 0
    assert result["metrics"]["p95_latency_ms"] is None
    assert result["metrics"]["success_rate"] is None
    assert result["metrics"]["estimated_cost_usd"] is None
    assert result["coverage"]["first_observation"] is None
    assert result["coverage"]["stale"]
    assert {agent["status"] for agent in result["agents"]} == {"unknown"}
    assert result["recommendations"] == []
    assert result["activity"] == []


def test_snapshot_and_trace_detail_are_sanitized_and_do_not_double_count_substeps(db):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    run_id = "turn_0123456789ab"
    sensitive = "private learner identity@example.com"
    turn = ModelToolTrace(
        name="coach_turn", trace_type="turn", status="success", created_at=now - timedelta(minutes=2),
        latency_ms=800, estimated_input_tokens=300, estimated_output_tokens=200, estimated_cost_usd=0.0009,
        user_id=sensitive, session_id=sensitive, turn_id=run_id, metadata_json={"question": sensitive},
    )
    db.add_all([
        turn,
        ModelToolTrace(name="op", trace_type="model", status="success", turn_id=run_id, provider="openai",
                       model="example-model", latency_ms=500, estimated_input_tokens=300),
        ModelToolTrace(name="orchestrator_dispatch", trace_type="dispatch", status="success", latency_ms=850),
        ModelToolTrace(name="tutor_turn", trace_type="turn", status="failed", latency_ms=700,
                       created_at=now - timedelta(days=2)),
        AgentRuntimeRun(run_id=run_id, lead_agent="lead_coach_orchestrator", status="success",
                        latency_ms=800, state_json={"excerpt": sensitive}, user_id=sensitive),
        AgentRuntimeStep(run_id=run_id, step_name=sensitive, agent_name="context_retriever", status="success",
                         latency_ms=100, input_json={"context": sensitive}, error=sensitive),
        AgentRuntimeMessage(run_id=run_id, sender_agent="context_retriever", receiver_agent="lead_coach_orchestrator",
                            message_type="context_result", result_json={"content": sensitive}),
        AgentRuntimeHandoff(run_id=run_id, from_agent="lead_coach_orchestrator", to_agent="answer_reviewer",
                            status="completed", reason=sensitive),
        ModelToolTrace(name="tutor_turn", trace_type="turn", status="failed", latency_ms=200),
        ModelToolTrace(name="tutor_turn", trace_type="turn", status="failed", latency_ms=300),
        ModelToolTrace(name="tutor_turn", trace_type="turn", status="failed", latency_ms=400),
    ])
    db.commit()
    result = evidence.snapshot(db, days=1, include_live=False)
    assert result["metrics"]["recorded"] == 4
    assert result["metrics"]["completed"] == 1
    assert result["metrics"]["failed"] == 3
    assert result["metrics"]["p50_latency_ms"] == 300
    assert result["metrics"]["p95_latency_ms"] == 800
    assert result["metrics"]["estimated_tokens"] == 500
    assert result["metrics"]["estimated_cost_usd"] == 0.0009
    assert result["recommendations"][0]["agent_id"] == "tutor"
    assert result["collaboration"] == {"messages": 1, "handoffs": 1}
    assert result["models"][0]["calls"] == 1
    assert sensitive not in json.dumps(result)
    detail = evidence.trace_detail(db, f"turn-{turn.id}")
    assert detail["activity"]["run_id"] == run_id
    assert len(detail["related_activity"]) == 3
    assert len(detail["handoffs"]) == 2
    assert detail["models"][0]["provider"] == "openai"
    assert sensitive not in json.dumps(detail)
    assert evidence.trace_detail(db, "turn-untrusted") is None
    assert evidence.trace_detail(db, "turn-9999999") is None


def test_live_events_track_overlapping_runs_without_exposing_task_text(db, monkeypatch):
    now = datetime.now().astimezone()
    long_run, short_run = "run_000000000001", "run_000000000002"
    events = [
        {"agent_id": "tutor", "event_type": kind, "data": {"run_id": run_id, "task": "private text"},
         "timestamp": (now - timedelta(seconds=seconds)).isoformat()}
        for kind, run_id, seconds in [
            ("task_complete", short_run, 1),
            ("task_start", short_run, 2),
            ("task_start", long_run, 3),
        ]
    ]
    monkeypatch.setattr(evidence.event_bus, "get_recent_events", lambda **kwargs: events)
    result = evidence.snapshot(db, days=1)
    tutor = next(agent for agent in result["agents"] if agent["id"] == "tutor")
    assert tutor["status"] == "working"
    assert tutor["metrics"]["active"] == 1
    assert "private text" not in json.dumps(result)


def test_dispatch_failure_is_traced_without_changing_response(db, monkeypatch):
    monkeypatch.setattr(observability_store, "SessionLocal", lambda: Session(db.get_bind()))
    monkeypatch.setattr(agentic_orchestrator.event_bus, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(agentic_orchestrator, "_agent_executor", lambda route: lambda req, db: {"type": "error", "answer": "Error"})
    request = SimpleNamespace(question="Audit test", session_id="test", mode="tutor")
    result = agentic_orchestrator.execute_agentic_request(request)
    assert result["type"] == "error"
    traces = db.query(ModelToolTrace).order_by(ModelToolTrace.id).all()
    assert {(row.trace_type, row.status) for row in traces} == {("turn", "failed"), ("dispatch", "failed")}
