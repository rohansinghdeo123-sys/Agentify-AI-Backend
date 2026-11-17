"""Agent HQ integration over the protected API, real queue, and durable SQLite state."""

from datetime import datetime, timedelta
from itertools import count
import json
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config
from app.security import verify_firebase_user
from database import Base, get_db
from models import AdminAuditLog, AgentHQRun, ContentIngestionJob, ModelToolTrace
from routers import agent_hq as routes
from services import agent_hq, agent_hq_evidence
from services.job_queue import DbJobQueue
from services.ttl_cache import TTLCache

PREFIX = "/admin/agent-hq"
AMIT = {"Authorization": "Bearer amit"}


class HQHarness:
    def __init__(self, client, queue, sessions):
        self.client = client
        self.queue = queue
        self.sessions = sessions

    def post_run(self, **payload):
        return self.client.post(f"{PREFIX}/runs", json={
            "request_id": str(uuid4()), "kind": "direct",
            "topic": "What are your responsibilities?", "participant_ids": ["tutor"],
            **payload,
        })

    def process_next(self):
        job_id = self.queue._claim_next()
        assert job_id is not None
        self.queue._execute(job_id)

    def detail(self, run_id):
        response = self.client.get(f"{PREFIX}/runs/{run_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def version(self):
        with self.sessions() as db:
            return agent_hq.event_version(db)


@pytest.fixture
def hq(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_HQ_MODEL_ANALYSIS", "0")
    engine = create_engine(f"sqlite:///{(tmp_path / 'hq.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False)
    queue = DbJobQueue(session_factory=sessions, poll_seconds=0.02)
    monkeypatch.setattr(routes, "job_queue", queue)
    monkeypatch.setattr(agent_hq, "job_queue", queue)
    monkeypatch.setattr(agent_hq, "_snapshots", TTLCache(max_entries=16))
    monkeypatch.setattr(config, "AGENT_HQ_EXECUTIVE_UIDS", {"rohan": "", "amit": ""})
    monkeypatch.setattr(agent_hq_evidence.event_bus, "get_recent_events", lambda **_: [])
    agent_hq.register_agent_hq()

    def session_dependency():
        with sessions() as db:
            yield db

    def verify_test_token(authorization: str | None = Header(default=None)):
        token = (authorization or "").removeprefix("Bearer ")
        if token == "admin":
            return {"uid": "generic-admin", "admin": True}
        if token in config.BUILTIN_FOUNDER_IDENTITIES:
            return {
                "uid": token,
                "email": config.BUILTIN_FOUNDER_IDENTITIES[token],
                "email_verified": True,
            }
        raise HTTPException(401, detail="Invalid test token")

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_db] = session_dependency
    app.dependency_overrides[verify_firebase_user] = verify_test_token
    try:
        with TestClient(app, headers={"Authorization": "Bearer rohan"}) as client:
            yield HQHarness(client, queue, sessions)
    finally:
        queue.stop()
        engine.dispose()


def test_both_executives_can_inspect_real_registry_and_generic_admin_cannot(hq):
    url = f"{PREFIX}/snapshot"
    assert hq.client.get(url, headers={"Authorization": "invalid"}).status_code == 401
    assert hq.client.get(url, headers={"Authorization": "Bearer admin"}).status_code == 404
    rohan = hq.client.get(url)
    amit = hq.client.get(url, headers=AMIT)
    assert rohan.status_code == amit.status_code == 200
    data = rohan.json()
    assert {a["id"] for a in data["executives"]} == {"rohan", "amit"}
    assert {a["id"] for a in data["agents"]} == {a["id"] for a in agent_hq_evidence.registry()}
    assert data["metrics"]["recorded"] == 0
    assert data["metrics"]["p95_latency_ms"] is None
    assert not data["activity"]
    assert "Time to first token" in data["coverage"]["unavailable"]
    assert not hq.client.get(f"{PREFIX}/runs").json()["runs"]
    assert hq.client.get(url, params={"days": 90}).status_code == 200
    assert hq.client.get(url, params={"days": 91}).status_code == 422


def test_requests_are_idempotent_bounded_and_followups_keep_the_same_role(hq):
    request_id = str(uuid4())
    initial = hq.version()
    response = hq.post_run(request_id=request_id)
    assert response.status_code == 202, response.text
    first = response.json()["id"]
    assert hq.version() != initial
    assert hq.post_run(request_id=request_id).json()["id"] == first
    assert hq.post_run(request_id=request_id, topic="different").status_code == 409
    assert hq.post_run(parent_run_id=first).status_code == 409
    assert hq.post_run(kind="meeting", participant_ids=["tutor"] * (agent_hq.MAX_PARTICIPANTS + 1)).status_code == 422
    assert hq.post_run(kind="meeting", participant_ids=["invented-role"]).status_code == 422
    assert hq.post_run(topic="Second active request").status_code == 202
    assert hq.post_run(topic="Third active request").status_code == 429

    with hq.sessions() as db:
        assert db.query(AgentHQRun).count() == 2
        assert db.query(ContentIngestionJob).count() == 2
        assert db.query(AdminAuditLog).filter_by(action="agent_hq_run").count() == 2

    queued_version = hq.version()
    hq.process_next()
    assert hq.version() != queued_version
    assert hq.detail(first)["status"] == "completed"
    assert hq.post_run(parent_run_id=first, participant_ids=["coach"]).status_code == 422
    followup = hq.post_run(parent_run_id=first, topic="What did you do today?")
    assert followup.status_code == 202, followup.text
    assert followup.json()["parent_run_id"] == first
    assert hq.client.get(f"{PREFIX}/runs/{first}", headers=AMIT).status_code == 200


def _record_tutor_failures(sessions):
    with sessions() as db:
        for _ in range(3):
            db.add(ModelToolTrace(
                trace_type="turn", name="tutor_turn", status="failed", latency_ms=1900,
                turn_id="turn_0123456789ab", user_id="private-learner-id",
                metadata_json={"prompt": "private-learner-prompt"},
            ))
        db.commit()


def test_meeting_worker_persists_delegation_findings_and_safe_evidence(hq):
    _record_tutor_failures(hq.sessions)
    response = hq.post_run(kind="meeting", topic="Discuss failures", participant_ids=["tutor", "coach", "tutor"])
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    hq.queue.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = hq.detail(run_id)
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert result["status"] == "completed", result
    assert result["participant_ids"] == ["tutor", "coach"]
    assert [m["kind"] for m in result["messages"]] == ["request", "delegation", "finding", "finding", "summary"]
    findings = {entry["agent_id"]: entry for entry in result["result"]["findings"]}
    assert "3 failed" in findings["tutor"]["text"]
    assert findings["tutor"]["evidence"][0]["run_id"] == "turn_0123456789ab"
    assert findings["coach"]["text"] == "No recorded activity for this period."
    assert "private-learner" not in str(result)
    assert hq.client.get(f"{PREFIX}/runs/{run_id}", headers=AMIT).json() == result


def test_run_pagination_uses_stable_cursor_and_filters(hq):
    for i in range(3):
        response = hq.post_run(kind="report", topic=f"Reliability period {i}", participant_ids=["tutor"])
        assert response.status_code == 202
        hq.process_next()
    response = hq.post_run(topic="Other topic")
    assert response.status_code == 202
    params = {"limit": 2, "kind": "report", "search": "reliability"}
    first = hq.client.get(f"{PREFIX}/runs", params=params).json()
    second = hq.client.get(f"{PREFIX}/runs", params={**params, "before": first["next_cursor"]}).json()
    assert [item["topic"] for item in first["runs"] + second["runs"]] == [
        "Reliability period 2", "Reliability period 1", "Reliability period 0",
    ]
    assert second["next_cursor"] is None
    assert hq.client.get(f"{PREFIX}/runs", params={"before": str(uuid4())}).status_code == 404


def test_report_ending_on_filters_out_later_telemetry(hq):
    observations = [
        (datetime(2025, 11, 11, 12), "turn_0123456789ab"),
        (datetime(2025, 11, 13, 12), "turn_fedcba987654"),
    ]
    with hq.sessions() as db:
        db.add_all(ModelToolTrace(
            trace_type="turn", name="tutor_turn", status="completed", created_at=stamp, turn_id=turn_id,
        ) for stamp, turn_id in observations)
        db.commit()
    response = hq.post_run(kind="report", days=7, ending_on="2025-11-12")
    assert response.status_code == 202, response.text
    hq.process_next()
    report = hq.detail(response.json()["id"])
    assert report["status"] == "completed"
    assert report["ending_on"] == "2025-11-12"
    assert report["result"]["ends_at"].startswith("2025-11-12")
    evidence = report["result"]["findings"][0]["evidence"]
    assert [item["run_id"] for item in evidence] == ["turn_0123456789ab"]


def test_recommendations_require_evidence_and_decisions_update_event_version(hq):
    initial = hq.version()
    _record_tutor_failures(hq.sessions)
    assert hq.version() != initial
    response = hq.client.get(f"{PREFIX}/snapshot")
    assert response.status_code == 200, response.text
    recommendation = next(item for item in response.json()["recommendations"] if item["agent_id"] == "tutor")
    assert recommendation["status"] == "pending"
    assert recommendation["evidence"]
    payload = {"status": "accepted", "note": "Review execution records"}
    assert hq.client.post(f"{PREFIX}/recommendations/nonexistent/decision", json=payload).status_code == 404
    version = hq.version()
    url = f"{PREFIX}/recommendations/{recommendation['id']}/decision"
    accepted = hq.client.post(url, json=payload, headers=AMIT)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["status"] == "accepted"
    assert hq.version() != version
    readback = hq.client.get(f"{PREFIX}/snapshot").json()["recommendations"]
    assert next(item for item in readback if item["id"] == recommendation["id"])["status"] == "accepted"
    with hq.sessions() as db:
        audit = db.query(AdminAuditLog).filter_by(action="agent_hq_recommendation_decision").one()
        assert audit.actor_uid == "amit"
        assert db.query(ContentIngestionJob).count() == 0


def test_scheduled_standup_runs_once_per_window(hq, monkeypatch):
    now = datetime(2026, 11, 17, 3, 0)
    monkeypatch.setattr(routes, "utcnow", lambda: now)
    monkeypatch.setattr(agent_hq, "utcnow", lambda: now)
    url = f"{PREFIX}/schedule"
    assert not hq.client.get(url).json()["enabled"]
    payload = {"enabled": True, "timezone": "Asia/Kolkata", "hour": 9, "minute": 0}
    response = hq.client.put(url, json={**payload, "timezone": "unavailable/zone"})
    assert response.status_code == 422
    version = hq.version()
    response = hq.client.put(url, json=payload, headers=AMIT)
    assert response.status_code == 200, response.text
    assert response.json()["next_run_at"] == "2026-11-17T03:30:00Z"
    assert hq.version() != version
    now += timedelta(minutes=31)
    with hq.sessions() as db:
        agent_hq.maintain(db)
        agent_hq.maintain(db)
        scheduled = db.query(AgentHQRun).filter_by(kind="standup").one()
        run_id = scheduled.id
        assert scheduled.status == "queued"
    hq.process_next()
    assert hq.detail(run_id)["status"] == "completed"
    assert hq.client.get(url).json()["next_run_at"] == "2026-11-18T03:30:00Z"


def test_interrupted_and_failed_jobs_are_recoverable_with_safe_status(hq, monkeypatch):
    run_id = hq.post_run().json()["id"]
    with hq.sessions() as db:
        run = db.get(AgentHQRun, run_id)
        run.status = "running"
        job = db.query(ContentIngestionJob).one()
        job.status = "failed"
        job.error = "private raw exception"
        db.commit()
        agent_hq.maintain(db)
    interrupted = hq.detail(run_id)
    assert interrupted["status"] == "failed"
    assert "raw exception" not in str(interrupted)
    fresh = hq.post_run()
    assert fresh.status_code == 202

    def unavailable(*args):
        raise RuntimeError("private raw exception")

    monkeypatch.setattr(agent_hq, "load_snapshot", unavailable)
    hq.process_next()
    failed = hq.detail(fresh.json()["id"])
    assert failed["status"] == "failed"
    assert failed["messages"][-1]["kind"] == "system"
    assert "raw exception" not in str(failed)


def test_optional_model_timeout_preserves_operational_finding(hq, monkeypatch):
    monkeypatch.setenv("AGENT_HQ_MODEL_ANALYSIS", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    attempted = []

    def unavailable_client(**kwargs):
        attempted.append((kwargs["timeout"], kwargs["max_retries"]))
        raise TimeoutError("private provider diagnostic")

    monkeypatch.setattr("groq.Client", unavailable_client)
    run_id = hq.post_run().json()["id"]
    hq.process_next()
    reply = hq.detail(run_id)
    assert attempted == [(8.0, 0)]
    assert reply["status"] == "completed"
    assert reply["result"]["findings"][0]["agent_id"] == "tutor"
    assert reply["result"]["analysis"] is None
    assert "private provider diagnostic" not in str(reply)


def test_sse_reconnect_gets_the_latest_persisted_version(hq, monkeypatch):
    clock = count(step=30)
    monkeypatch.setattr(routes, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(routes, "SessionLocal", hq.sessions)
    monkeypatch.setattr(routes, "_streams", {})

    async def no_sleep(_):
        return None

    monkeypatch.setattr(routes, "asyncio", SimpleNamespace(sleep=no_sleep))

    def connect():
        with hq.client.stream("GET", f"{PREFIX}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-store"
            lines = list(response.iter_lines())
        assert "event: update" in lines
        payload = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        return json.loads(payload)["version"]

    first = connect()
    assert hq.post_run().status_code == 202
    second = connect()
    assert second == hq.version()
    assert first != second
    assert connect() == second
    assert not routes._streams
