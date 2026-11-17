# Logic/agentic_orchestrator.py

"""
Central agentic control plane for the education platform.

This module keeps the existing specialist agents, but puts a proper registry,
routing policy, workflow description, and telemetry envelope around them. It is
the first step toward a fully autonomous multi-agent learning system.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from Logic.agent_event_bus import event_bus
from Logic.agents.exam_agent import exam_agent
from Logic.agents.planner_agent import planner_agent
from Logic.agents.revision_agent import revision_agent
from Logic.agents.tutor_agent import tutor_agent
from Logic.coach.model_gateway import model_gateway
from Logic.observability_store import persist_agent_trace

try:
    from Logic.agents.coach_agent import coach_agent
except Exception:  # pragma: no cover - defensive import for partial deployments
    coach_agent = None

logger = logging.getLogger("ai_educator.agentic_orchestrator")


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    display_name: str
    role: str
    responsibility: str
    modes: tuple[str, ...]
    intents: tuple[str, ...]
    tools: tuple[str, ...]
    autonomous_priority: int


@dataclass(frozen=True)
class AgentRoute:
    run_id: str
    intent: str
    primary_agent: str
    mode: str
    confidence: float
    reason: str
    workflow: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["workflow"] = list(self.workflow)
        return data


AGENT_REGISTRY: Dict[str, AgentDefinition] = {
    "orchestrator": AgentDefinition(
        agent_id="orchestrator",
        display_name="Supervisor Orchestrator",
        role="Supervisor",
        responsibility="Classifies intent, selects the right specialist, and standardizes run telemetry.",
        modes=("auto", "greeting"),
        intents=("greeting", "fallback"),
        tools=("intent_classifier", "agent_registry", "event_bus"),
        autonomous_priority=100,
    ),
    "coach": AgentDefinition(
        agent_id="coach",
        display_name="Personal AI Coach",
        role="Personal mentor",
        responsibility="Turns learning analytics and memory into personalized advice, plans, and next actions.",
        modes=("coach",),
        intents=("coach", "motivation", "strategy"),
        tools=("coach_memory", "analytics_snapshot", "knowledge_graph"),
        autonomous_priority=90,
    ),
    "tutor": AgentDefinition(
        agent_id="tutor",
        display_name="Subject Tutor",
        role="Concept teacher",
        responsibility="Explains doubts with curriculum context and clear student-friendly structure.",
        modes=("ask", "tutor", "doubt"),
        intents=("doubt", "explanation"),
        tools=("knowledge_search", "knowledge_graph"),
        autonomous_priority=80,
    ),
    "revision": AgentDefinition(
        agent_id="revision",
        display_name="Revision Specialist",
        role="Revision designer",
        responsibility="Creates summaries, key points, and deep explanations for fast recall.",
        modes=("summary", "explain", "key", "keypoints"),
        intents=("revision",),
        tools=("knowledge_search", "knowledge_graph"),
        autonomous_priority=70,
    ),
    "exam": AgentDefinition(
        agent_id="exam",
        display_name="Exam Generator",
        role="Assessment builder",
        responsibility="Creates MCQs, probable questions, distractors, and exam-style practice.",
        modes=("exam", "test", "quiz", "mcq", "probable"),
        intents=("exam",),
        tools=("knowledge_search", "knowledge_graph", "question_normalizer"),
        autonomous_priority=70,
    ),
    "planner": AgentDefinition(
        agent_id="planner",
        display_name="Study Planner",
        role="Learning strategist",
        responsibility="Builds sequenced study plans from weak topics, progress, and exam goals.",
        modes=("plan", "planner", "study_plan"),
        intents=("plan",),
        tools=("analytics_engine", "knowledge_graph"),
        autonomous_priority=75,
    ),
}


INTENT_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "coach": (
        "coach",
        "motivate",
        "motivation",
        "advice",
        "guide me",
        "what should i study",
        "where am i weak",
        "weak area",
        "focus on",
        "next best",
        "strategy",
        "exam strategy",
        "study advice",
        "daily plan",
        "improve",
        "how can i score",
    ),
    "exam": (
        "mcq",
        "quiz",
        "test",
        "exam",
        "generate questions",
        "practice questions",
        "probable",
    ),
    "revision": (
        "summary",
        "summarize",
        "revise",
        "revision",
        "key points",
        "keypoints",
        "deep explain",
    ),
    "plan": (
        "plan",
        "study plan",
        "roadmap",
        "schedule",
        "timetable",
    ),
    "greeting": (
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "thanks",
        "thank you",
    ),
}


def get_agent_registry() -> list[Dict[str, Any]]:
    return [
        {
            **asdict(agent),
            "modes": list(agent.modes),
            "intents": list(agent.intents),
            "tools": list(agent.tools),
        }
        for agent in sorted(
            AGENT_REGISTRY.values(),
            key=lambda item: item.autonomous_priority,
            reverse=True,
        )
    ]


def classify_intent(message: str) -> str:
    msg_lower = (message or "").lower().strip()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in msg_lower for keyword in keywords):
            return intent
    return "doubt"


def _detect_revision_type(message: str) -> str:
    msg_lower = (message or "").lower()
    if any(keyword in msg_lower for keyword in ("summary", "summarize", "smart summary")):
        return "summary"
    if any(keyword in msg_lower for keyword in ("explain", "deep explain", "explanation")):
        return "explain"
    if any(keyword in msg_lower for keyword in ("key point", "keypoint", "key_point", "bullet")):
        return "key"
    return "summary"


def _detect_exam_type(message: str) -> str:
    msg_lower = (message or "").lower()
    if any(keyword in msg_lower for keyword in ("probable", "3 marks", "5 marks", "subjective")):
        return "probable"
    return "mcq"


def _workflow_for(primary_agent: str) -> tuple[str, ...]:
    if primary_agent == "orchestrator":
        return ("classify", "respond")
    return ("classify", f"dispatch:{primary_agent}", "normalize", "deliver")


def resolve_agent_route(request) -> AgentRoute:
    question = getattr(request, "question", "") or ""
    raw_mode = str(getattr(request, "mode", "") or "").lower().strip()
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    explicit_modes: Dict[str, tuple[str, str, str, float, str]] = {
        "coach": ("coach", "coach", "coach", 0.98, "Frontend explicitly requested the personal coach."),
        "summary": ("revision", "revision", "summary", 0.95, "Frontend requested summary revision mode."),
        "explain": ("revision", "revision", "explain", 0.95, "Frontend requested explanation revision mode."),
        "key": ("revision", "revision", "key", 0.95, "Frontend requested key-points revision mode."),
        "keypoints": ("revision", "revision", "key", 0.95, "Frontend requested key-points revision mode."),
        "exam": ("exam", "exam", "mcq", 0.95, "Frontend requested exam practice mode."),
        "test": ("exam", "exam", "mcq", 0.95, "Frontend requested test mode."),
        "quiz": ("exam", "exam", "mcq", 0.95, "Frontend requested quiz mode."),
        "mcq": ("exam", "exam", "mcq", 0.95, "Frontend requested MCQ generation."),
        "probable": ("exam", "exam", "probable", 0.95, "Frontend requested probable questions."),
        "plan": ("planner", "plan", "plan", 0.95, "Frontend requested a study plan."),
        "planner": ("planner", "plan", "plan", 0.95, "Frontend requested the planner."),
        "study_plan": ("planner", "plan", "plan", 0.95, "Frontend requested a study plan."),
        "ask": ("tutor", "doubt", "ask", 0.9, "Frontend requested open tutoring."),
        "tutor": ("tutor", "doubt", "tutor", 0.9, "Frontend requested open tutoring."),
        "doubt": ("tutor", "doubt", "doubt", 0.9, "Frontend requested doubt solving."),
    }

    if raw_mode in explicit_modes:
        primary_agent, intent, mode, confidence, reason = explicit_modes[raw_mode]
    else:
        intent = classify_intent(question)
        confidence = 0.78
        reason = f"Keyword classifier selected intent '{intent}'."

        if intent == "coach":
            primary_agent = "coach"
            mode = "coach"
        elif intent == "exam":
            primary_agent = "exam"
            mode = _detect_exam_type(question)
        elif intent == "revision":
            primary_agent = "revision"
            mode = _detect_revision_type(question)
        elif intent == "plan":
            primary_agent = "planner"
            mode = "plan"
        elif intent == "greeting":
            primary_agent = "orchestrator"
            mode = "greeting"
            confidence = 0.92
        else:
            primary_agent = "tutor"
            mode = "doubt"

    return AgentRoute(
        run_id=run_id,
        intent=intent,
        primary_agent=primary_agent,
        mode=mode,
        confidence=confidence,
        reason=reason,
        workflow=_workflow_for(primary_agent),
    )


def _coach_unavailable_response() -> dict:
    return {
        "type": "coach",
        "answer": "Coach agent is not available yet. Please check Logic/agents/coach_agent.py.",
        "metadata": {"agent": "coach", "status": "unavailable"},
    }


def _greeting_response() -> dict:
    return {
        "type": "greeting",
        "answer": "Ready. Ask any chemistry question, request a study strategy, or choose a focused study mode.",
        "metadata": {"agent": "orchestrator"},
    }


def _planner_unavailable_response() -> dict:
    return {
        "type": "planner",
        "answer": "Planner needs a database session so it can read your learning analytics.",
        "metadata": {"agent": "planner", "status": "needs_db_session"},
    }


def _agent_executor(route: AgentRoute) -> Callable[[Any, Any], dict]:
    if route.primary_agent == "coach":
        return lambda request, db: coach_agent(request, db=db) if coach_agent else _coach_unavailable_response()
    if route.primary_agent == "revision":
        return lambda request, db: revision_agent(request, revision_type=route.mode)
    if route.primary_agent == "exam":
        return lambda request, db: exam_agent(request, exam_type=route.mode)
    if route.primary_agent == "planner":
        return lambda request, db: planner_agent(request, db) if db is not None else _planner_unavailable_response()
    if route.primary_agent == "orchestrator":
        return lambda request, db: _greeting_response()
    return lambda request, db: tutor_agent(request)


def _attach_run_metadata(result: Any, route: AgentRoute, latency_ms: int) -> dict:
    if isinstance(result, dict):
        response = dict(result)
    else:
        response = {"answer": str(result or "")}

    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    metadata["agentic_run"] = {
        **route.to_dict(),
        "latency_ms": latency_ms,
        "registry_version": "agentic-control-plane-v1",
    }
    metadata["agent"] = metadata.get("agent") or route.primary_agent

    response["metadata"] = metadata
    return response


def execute_agentic_request(request, db: Optional[Any] = None) -> dict:
    started_at = time.time()
    route = resolve_agent_route(request)
    question = getattr(request, "question", "") or ""
    session_id = str(getattr(request, "session_id", "") or "")
    # Scope model-call records to this run so the durable trace below captures
    # exactly the calls this agent made (cost, tokens, latency, provider).
    model_gateway.begin_turn(route.run_id)

    event_bus.emit(
        "orchestrator",
        "task_start",
        {
            "task": f"Agentic run {route.run_id}",
            "message": f"Classifying request for autonomous routing: {question[:80]}",
            "run_id": route.run_id,
            "mode": getattr(request, "mode", None) or "auto",
        },
        session_id=session_id,
    )
    event_bus.emit(
        "orchestrator",
        "step",
        {
            "step": "classify",
            "message": route.reason,
            "run_id": route.run_id,
            "intent": route.intent,
            "confidence": route.confidence,
        },
        session_id=session_id,
    )
    event_bus.emit(
        "orchestrator",
        "step",
        {
            "step": "route",
            "message": f"Dispatching to {AGENT_REGISTRY[route.primary_agent].display_name}",
            "run_id": route.run_id,
            "target_agent": route.primary_agent,
            "workflow": list(route.workflow),
        },
        session_id=session_id,
    )

    try:
        event_bus.emit(
            route.primary_agent,
            "task_start",
            {
                "task": f"{route.mode}: {question[:80]}",
                "message": f"{AGENT_REGISTRY[route.primary_agent].display_name} started.",
                "run_id": route.run_id,
                "intent": route.intent,
            },
            session_id=session_id,
        )

        result = _agent_executor(route)(request, db)
        agent_latency_ms = round((time.time() - started_at) * 1000)

        status = _execution_status(result)
        event_bus.emit(
            route.primary_agent,
            "task_complete",
            {
                "status": status,
                "message": f"{AGENT_REGISTRY[route.primary_agent].display_name} completed.",
                "run_id": route.run_id,
                "latency_ms": agent_latency_ms,
            },
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("Agentic request failed")
        event_bus.emit(
            route.primary_agent,
            "error",
            {
                "message": str(exc),
                "run_id": route.run_id,
                "target_agent": route.primary_agent,
            },
            session_id=session_id,
            severity="error",
        )
        result = {
            "type": "error",
            "answer": "The agentic system hit an internal error while processing this request.",
            "metadata": {"agent": route.primary_agent, "error": str(exc)},
        }
        status = "failed"

    latency_ms = round((time.time() - started_at) * 1000)
    response = _attach_run_metadata(result, route, latency_ms)

    # Durably trace non-coach agents (the coach pipeline self-traces via
    # persist_coach_trace). Gives revision/exam/tutor/planner the same cost and
    # quality visibility. persist_agent_trace never raises.
    if route.primary_agent != "coach":
        result_meta = response.get("metadata") if isinstance(response, dict) else {}
        result_meta = result_meta if isinstance(result_meta, dict) else {}
        quality = (
            {"score": result_meta.get("quality_score"), "passed": result_meta.get("quality_passed")}
            if "quality_score" in result_meta
            else None
        )
        persist_agent_trace(
            agent=route.primary_agent,
            turn_id=route.run_id,
            session_id=session_id,
            status=status,
            latency_ms=latency_ms,
            quality=quality,
            model_calls=model_gateway.records(),
            metadata={"mode": route.mode, "intent": route.intent},
            record_dispatch=True,
        )
    else:
        persist_agent_trace(
            agent="orchestrator",
            turn_id=route.run_id,
            status=status,
            latency_ms=latency_ms,
            trace_type="dispatch",
        )
    event_bus.emit(
        "orchestrator",
        "task_complete",
        {
            "status": status,
            "message": f"Delivered by {route.primary_agent}. Total latency: {latency_ms}ms",
            "run_id": route.run_id,
            "target_agent": route.primary_agent,
            "latency_ms": latency_ms,
        },
        session_id=session_id,
    )

    return response


def _execution_status(result: Any) -> str:
    """Classify recorded outcomes without changing the student response."""
    if not isinstance(result, dict) or result.get("type") == "error" or result.get("error"):
        return "failed"
    metadata = result.get("metadata")
    if isinstance(metadata, dict) and metadata.get("quality_passed") is False:
        return "needs_review"
    return "success"
