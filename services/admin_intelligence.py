"""Admin console aggregation: data registry, model registry, traces, overview,
students, system health, agent intelligence, data intake, and console payload.

This module is a faithful extraction of the admin-aggregation logic that
previously lived in ``main.py``. Behavior is unchanged; only the home moved.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import date, datetime, time as datetime_time, timedelta
from hashlib import sha256
from typing import Any, Dict, List, Optional

from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from app import config, security
from app.serializers import (
    format_test_session,
    serialize_audit_log as _serialize_audit_log,
    student_payload as _student_payload,
    trace_payload as _trace_payload,
)
from database import check_db_health
from Logic.agent_event_bus import event_bus
from Logic.coach.settings import coach_settings
from Logic.knowledge_graph import knowledge_graph
from Logic.observability_store import get_observability_summary, get_recent_observability_events
from Logic.tools.artifact_generator import available_artifact_sections
from models import (
    AICoachInteraction,
    AICoachMemory,
    AICoachProfile,
    AdminAuditLog,
    AgentChatMemory,
    ContentChapter,
    ContentChunk,
    ContentIngestionJob,
    DailyQuotaUsage,
    ModelToolTrace,
    ObservabilityEvent,
    TestHistory,
    TopicPerformance,
    UserProfile,
    UserProgress,
)

logger = logging.getLogger("ai_educator.services.admin_intelligence")

ADMIN_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _prompt_fingerprint() -> str:
    try:
        from prompts.registry import prompt_registry

        return prompt_registry.fingerprint()
    except Exception:
        return "agentic-control-plane-v1"


# ===================================================================
# The functions below are extracted verbatim from the original main.py.
# ===================================================================


SUPPORTED_ADMIN_ACTIONS = {
    "clear_temp_cache",
    "export_data_report",
    "refresh_snapshot",
}
CONFIRMED_ADMIN_ACTIONS = {
    "clear_temp_cache",
    "clear_temp_chat_state",
    "disable_user",
    "promote_model_version",
    "refresh_vector_index",
    "reindex_material",
    "reset_user_session",
}


def _admin_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _admin_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _admin_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def _safe_json_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _query_count(query: Any) -> int:
    try:
        return _admin_int(query.count())
    except Exception:
        logger.exception("Admin count query failed")
        return 0


def _sum_scalar(db: Session, model: Any, column: Any, *filters: Any) -> float:
    try:
        query = db.query(func.sum(column))
        for item in filters:
            query = query.filter(item)
        return _admin_float(query.scalar())
    except Exception:
        logger.exception("Admin sum query failed for %s", model)
        return 0.0


def _admin_identity(current_admin: Dict[str, Any]) -> Dict[str, str]:
    return {
        "uid": str(current_admin.get("uid") or ""),
        "email": str(current_admin.get("email") or ""),
        "phone": str(current_admin.get("phone_number") or ""),
    }



def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for suffix in ("B", "KB", "MB", "GB"):
        if size < 1024 or suffix == "GB":
            return f"{size:.1f} {suffix}" if suffix != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _paragraph_chunks(text_value: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r"\n\s*\n+", text_value or "")
        if item.strip()
    ]


def _iter_data_files() -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    if not os.path.isdir(ADMIN_DATA_DIR):
        return files

    for root, _dirs, names in os.walk(ADMIN_DATA_DIR):
        for name in names:
            path = os.path.join(root, name)
            rel_path = os.path.relpath(path, ADMIN_DATA_DIR).replace("\\", "/")
            try:
                stat = os.stat(path)
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
                chunks = _paragraph_chunks(content)
                concepts = 0
                if name.lower().endswith(".json"):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            concepts = len(parsed)
                    except json.JSONDecodeError:
                        pass
                files.append(
                    {
                        "path": rel_path,
                        "bytes": int(stat.st_size),
                        "text_chars": len(content),
                        "chunks": concepts or len(chunks),
                        "low_quality_chunks": sum(1 for item in chunks if len(item) < 40),
                        "concepts": concepts,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "status": "ready",
                    }
                )
            except Exception as exc:
                files.append(
                    {
                        "path": rel_path,
                        "bytes": 0,
                        "text_chars": 0,
                        "chunks": 0,
                        "low_quality_chunks": 0,
                        "concepts": 0,
                        "modified_at": None,
                        "status": "failed",
                        "error": str(exc)[:240],
                    }
                )
    return sorted(files, key=lambda item: item["path"])


def _recent_turn_traces(db: Session, limit: int = 500) -> List[ModelToolTrace]:
    return (
        db.query(ModelToolTrace)
        .filter(ModelToolTrace.trace_type == "turn")
        .order_by(ModelToolTrace.created_at.desc(), ModelToolTrace.id.desc())
        .limit(limit)
        .all()
    )


def _trace_quality_summary(db: Session) -> Dict[str, Any]:
    turns = _recent_turn_traces(db)
    if not turns:
        return {
            "samples": 0,
            "quality_score": None,
            "grounded_answer_rate": None,
            "retrieval_success_rate": None,
        }

    quality_values: List[float] = []
    grounded_total = 0
    grounded_success = 0
    retrieval_total = 0
    retrieval_success = 0

    for row in turns:
        metadata = _safe_json_dict(row.metadata_json)
        quality = _safe_json_dict(metadata.get("quality"))
        retrieval = _safe_json_dict(metadata.get("retrieval"))

        score = quality.get("score") or quality.get("overall_score")
        if score is None and "passed" in quality:
            score = 100 if quality.get("passed") else 0
        if score is not None:
            quality_values.append(_admin_float(score))

        if retrieval.get("policy") or retrieval.get("source") or retrieval.get("section_id"):
            retrieval_total += 1
            if _admin_int(retrieval.get("paragraphs_found")) > 0 or retrieval.get("supported") is True:
                retrieval_success += 1

        if quality or retrieval:
            grounded_total += 1
            if quality.get("passed", True) and (
                not retrieval or _admin_int(retrieval.get("paragraphs_found")) > 0 or retrieval.get("supported") is True
            ):
                grounded_success += 1

    return {
        "samples": len(turns),
        "quality_score": round(sum(quality_values) / len(quality_values), 1) if quality_values else None,
        "grounded_answer_rate": round((grounded_success / grounded_total) * 100, 1) if grounded_total else None,
        "retrieval_success_rate": round((retrieval_success / retrieval_total) * 100, 1) if retrieval_total else None,
    }


def build_admin_data_registry(db: Session) -> Dict[str, Any]:
    files = _iter_data_files()
    total_bytes = sum(_admin_int(item.get("bytes")) for item in files)
    total_text_chars = sum(_admin_int(item.get("text_chars")) for item in files)
    total_chunks = sum(_admin_int(item.get("chunks")) for item in files)
    total_concepts = sum(_admin_int(item.get("concepts")) for item in files)
    failed_files = [item for item in files if item.get("status") == "failed"]
    low_quality_chunks = sum(_admin_int(item.get("low_quality_chunks")) for item in files)
    modified_times = [
        str(item.get("modified_at"))
        for item in files
        if item.get("modified_at")
    ]
    quality = _trace_quality_summary(db)

    source_coverage: Dict[str, int] = defaultdict(int)
    for item in files:
        parts = str(item.get("path") or "").split("/")
        subject = parts[0] if parts else "data"
        source_coverage[subject] += _admin_int(item.get("chunks"))

    extraction_percent = 100 if files and not failed_files else 0 if not files else round(((len(files) - len(failed_files)) / len(files)) * 100, 1)
    chunking_percent = 100 if total_chunks > 0 else 0

    return {
        "summary": {
            "study_materials": len(files),
            "total_file_size_bytes": total_bytes,
            "total_file_size_label": _format_bytes(total_bytes),
            "total_extracted_text_chars": total_text_chars,
            "chapters": len(knowledge_graph.list_chapters()),
            "topics": total_concepts or len(available_artifact_sections()),
            "chunks": total_chunks,
            "embeddings": 0,
            "vector_index_size_bytes": 0,
            "vector_index_size_label": "0 B",
            "last_indexed_time": max(modified_times) if modified_times else None,
            "failed_files": len(failed_files),
            "low_quality_chunks": low_quality_chunks,
        },
        "progress": {
            "extraction_complete": extraction_percent,
            "chunking_complete": chunking_percent,
            "embedding_complete": None,
            "index_health": None,
            "retrieval_success": quality["retrieval_success_rate"],
        },
        "source_coverage": [
            {"source": key, "chunks": value}
            for key, value in sorted(source_coverage.items(), key=lambda item: item[0])
        ],
        "files": files,
        "failed_files": failed_files,
        "vector_index": {
            "available": False,
            "status": "not_configured",
            "todo": "Connect a vector index service before reporting embeddings or index size.",
        },
        "notes": [
            "File sizes, text sizes, chunks, chapters, and topics are computed from backend data files.",
            "Embeddings and vector index size are intentionally not estimated because no vector index backend is present in this repository.",
        ],
    }


def _format_trace_row(row: ModelToolTrace) -> Dict[str, Any]:
    return {
        "id": row.id,
        "created_at": _admin_iso(row.created_at),
        "user_id": row.user_id or "",
        "session_id": row.session_id or "",
        "turn_id": row.turn_id or "",
        "trace_type": row.trace_type or "",
        "name": row.name or "",
        "provider": row.provider or "",
        "model": row.model or "",
        "status": row.status or "unknown",
        "latency_ms": _admin_int(row.latency_ms),
        "estimated_input_tokens": _admin_int(row.estimated_input_tokens),
        "estimated_output_tokens": _admin_int(row.estimated_output_tokens),
        "estimated_cost_usd": round(_admin_float(row.estimated_cost_usd), 8),
        "metadata": row.metadata_json or {},
    }


def _timeline_for_trace_group(rows: List[ModelToolTrace]) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: item.created_at or datetime.utcnow())
    timeline: List[Dict[str, Any]] = []
    turn_row = next((row for row in ordered if row.trace_type == "turn"), None)
    turn_metadata = _safe_json_dict(turn_row.metadata_json if turn_row else {})
    query = _safe_json_dict(turn_metadata.get("query"))
    retrieval = _safe_json_dict(turn_metadata.get("retrieval"))
    quality = _safe_json_dict(turn_metadata.get("quality"))

    if query:
        timeline.append(
            {
                "step": "user_query",
                "label": "User query",
                "status": "success",
                "detail": query.get("normalized") or query.get("question") or query.get("intent") or "Query captured",
            }
        )
    if query.get("intent"):
        timeline.append(
            {
                "step": "intent_detection",
                "label": "Intent detection",
                "status": "success",
                "detail": str(query.get("intent")),
            }
        )
    if retrieval:
        paragraphs = _admin_int(retrieval.get("paragraphs_found"))
        timeline.append(
            {
                "step": "retrieval",
                "label": "Retrieval",
                "status": "success" if paragraphs > 0 else "empty",
                "detail": f"{paragraphs} chunks from {retrieval.get('source') or 'study data'}",
            }
        )
        timeline.append(
            {
                "step": "source_check",
                "label": "Source check",
                "status": "success" if retrieval.get("supported", paragraphs > 0) else "warning",
                "detail": retrieval.get("section_id") or retrieval.get("policy") or "Grounding checked",
            }
        )

    for row in ordered:
        if row.trace_type == "model":
            timeline.append(
                {
                    "step": "model_call",
                    "label": row.name or "Model call",
                    "status": row.status or "unknown",
                    "detail": row.model or row.provider or "Model not recorded",
                    "latency_ms": _admin_int(row.latency_ms),
                }
            )
        elif row.trace_type == "tool":
            timeline.append(
                {
                    "step": "tool_call",
                    "label": row.name or "Tool call",
                    "status": row.status or "unknown",
                    "detail": str(_safe_json_dict(row.metadata_json).get("summary") or "Tool executed"),
                    "latency_ms": _admin_int(row.latency_ms),
                }
            )

    if quality:
        timeline.append(
            {
                "step": "quality_validation",
                "label": "Quality validation",
                "status": "success" if quality.get("passed", True) else "needs_review",
                "detail": ", ".join(str(item) for item in quality.get("issues", [])[:2]) or "Quality checked",
            }
        )

    if turn_row:
        timeline.append(
            {
                "step": "final_answer",
                "label": "Final answer",
                "status": turn_row.status or "unknown",
                "detail": f"{_admin_int(turn_row.latency_ms)} ms end-to-end",
                "latency_ms": _admin_int(turn_row.latency_ms),
            }
        )

    return timeline


def build_admin_traces(
    db: Session,
    *,
    limit: int = 40,
    trace_type: Optional[str] = None,
    status_filter: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    query = db.query(ModelToolTrace)
    if trace_type:
        query = query.filter(ModelToolTrace.trace_type == trace_type)
    if status_filter:
        query = query.filter(ModelToolTrace.status == status_filter)
    if user_id:
        query = query.filter(ModelToolTrace.user_id == user_id)

    rows = (
        query.order_by(ModelToolTrace.created_at.desc(), ModelToolTrace.id.desc())
        .limit(limit)
        .all()
    )
    grouped: Dict[str, List[ModelToolTrace]] = defaultdict(list)
    for row in rows:
        grouped[row.turn_id or f"{row.trace_type}-{row.id}"].append(row)

    return {
        "rows": [_format_trace_row(row) for row in rows],
        "runs": [
            {
                "turn_id": turn_id,
                "created_at": _admin_iso(max((row.created_at for row in group if row.created_at), default=None)),
                "status": next((row.status for row in group if row.trace_type == "turn"), group[0].status if group else "unknown"),
                "user_id": next((row.user_id for row in group if row.user_id), ""),
                "session_id": next((row.session_id for row in group if row.session_id), ""),
                "total_latency_ms": sum(
                    _admin_int(row.latency_ms) for row in group
                    if row.trace_type != "route"
                    and not (row.trace_type == "dispatch" and any(item.trace_type == "turn" for item in group))
                ),
                "estimated_cost_usd": round(sum(_admin_float(row.estimated_cost_usd) for row in group), 8),
                "models": sorted({row.model for row in group if row.model}),
                "timeline": _timeline_for_trace_group(group),
                "rows": [_format_trace_row(row) for row in sorted(group, key=lambda item: item.created_at or datetime.utcnow())],
            }
            for turn_id, group in grouped.items()
        ],
        "summary": get_observability_summary(db),
    }


def build_admin_model_registry(db: Session) -> Dict[str, Any]:
    quality = _trace_quality_summary(db)
    model_rows = (
        db.query(
            ModelToolTrace.provider,
            ModelToolTrace.model,
            func.count(ModelToolTrace.id),
            func.avg(ModelToolTrace.latency_ms),
            func.sum(ModelToolTrace.estimated_cost_usd),
        )
        .filter(ModelToolTrace.trace_type == "model")
        .group_by(ModelToolTrace.provider, ModelToolTrace.model)
        .all()
    )
    versions = [
        {
            "version": f"{provider or 'provider'}:{model or 'model'}",
            "provider": provider or "",
            "model": model or "",
            "samples": _admin_int(samples),
            "avg_latency_ms": round(_admin_float(avg_latency), 1),
            "estimated_cost_usd": round(_admin_float(cost), 8),
            "status": "observed",
            "quality_score": quality["quality_score"],
        }
        for provider, model, samples, avg_latency, cost in model_rows
    ]

    live_model = coach_settings.tutor_model or coach_settings.fast_model
    live_provider = coach_settings.provider
    for item in versions:
        if item["model"] == live_model or (not item["model"] and item["provider"] == live_provider):
            item["status"] = "live"

    return {
        "current": {
            "llm_provider": live_provider,
            "llm_model": live_model,
            "fast_model": coach_settings.fast_model,
            "review_model": coach_settings.review_model,
            "embedding_model": os.getenv("EMBEDDING_MODEL") or None,
            "rag_index_version": hashlib_data_version(),
            "prompt_version": os.getenv("PROMPT_VERSION") or _prompt_fingerprint(),
            "agent_workflow_version": os.getenv("AGENT_WORKFLOW_VERSION", "agentic-control-plane-v1"),
            "deployment_version": os.getenv("RENDER_GIT_COMMIT", "2.5.0-production-guardrails"),
            "quality_score": quality["quality_score"],
            "latency_ms": get_observability_summary(db).get("avg_model_latency_ms"),
            "failure_rate": model_failure_rate(db),
            "grounded_answer_rate": quality["grounded_answer_rate"],
            "samples": quality["samples"],
        },
        "versions": versions,
        "unsupported_actions": [
            "promote_model_version requires a deployment/version management backend before it can be enabled.",
        ],
    }


def hashlib_data_version() -> str:
    digest = sha256()
    for item in _iter_data_files():
        digest.update(str(item.get("path", "")).encode("utf-8"))
        digest.update(str(item.get("bytes", "")).encode("utf-8"))
        digest.update(str(item.get("modified_at", "")).encode("utf-8"))
    return f"data-{digest.hexdigest()[:12]}"


def model_failure_rate(db: Session) -> Optional[float]:
    total = _query_count(db.query(ModelToolTrace).filter(ModelToolTrace.trace_type == "model"))
    if not total:
        return None
    failures = _query_count(
        db.query(ModelToolTrace).filter(
            ModelToolTrace.trace_type == "model",
            ModelToolTrace.status.notin_(["success", "skipped"]),
        )
    )
    return round((failures / total) * 100, 1)


def build_admin_overview(db: Session) -> Dict[str, Any]:
    now = datetime.utcnow()
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    quality = _trace_quality_summary(db)
    data_registry = build_admin_data_registry(db)

    progress_users = db.query(UserProgress).all()
    user_ids = {user.user_id for user in progress_users if user.user_id}
    for (user_id_value,) in db.query(distinct(TestHistory.user_id)).all():
        if user_id_value:
            user_ids.add(user_id_value)
    for (user_id_value,) in db.query(distinct(AICoachProfile.user_id)).all():
        if user_id_value:
            user_ids.add(user_id_value)

    total_questions = sum(_admin_int(user.total_questions) for user in progress_users)
    total_correct = sum(_admin_int(user.total_correct) for user in progress_users)
    total_tests = sum(_admin_int(user.total_tests) for user in progress_users)

    active_today_ids = {
        user.user_id
        for user in progress_users
        if user.user_id and user.last_active_date == today
    }
    active_today_ids.update(
        user_id
        for (user_id,) in db.query(distinct(TestHistory.user_id)).filter(TestHistory.date == today).all()
        if user_id
    )
    active_today_ids.update(
        user_id
        for (user_id,) in db.query(distinct(AICoachInteraction.user_id))
        .filter(AICoachInteraction.created_at >= today_start)
        .all()
        if user_id
    )

    active_sessions = _query_count(
        db.query(distinct(ModelToolTrace.session_id)).filter(
            ModelToolTrace.created_at >= now - timedelta(hours=1),
            ModelToolTrace.session_id.isnot(None),
            ModelToolTrace.session_id != "",
        )
    )
    if active_sessions == 0:
        active_sessions = _query_count(
            db.query(distinct(AgentChatMemory.session_id)).filter(
                AgentChatMemory.timestamp >= now - timedelta(hours=1),
                AgentChatMemory.session_id.isnot(None),
                AgentChatMemory.session_id != "",
            )
        )

    observed = get_observability_summary(db)
    failed_requests = _query_count(
        db.query(ObservabilityEvent).filter(ObservabilityEvent.severity.in_(["error", "critical"]))
    )
    total_cost = _sum_scalar(db, ModelToolTrace, ModelToolTrace.estimated_cost_usd)
    total_input_tokens = _sum_scalar(db, ModelToolTrace, ModelToolTrace.estimated_input_tokens)
    total_output_tokens = _sum_scalar(db, ModelToolTrace, ModelToolTrace.estimated_output_tokens)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "users": {
            "total": len(user_ids),
            "active_today": len(active_today_ids),
            "active_sessions": active_sessions,
        },
        "learning": {
            "questions_asked": _query_count(db.query(AICoachInteraction).filter(AICoachInteraction.role == "user")),
            "revision_generations": _query_count(
                db.query(ObservabilityEvent).filter(
                    ObservabilityEvent.agent_id == "revision",
                    ObservabilityEvent.event_type == "task_complete",
                )
            ),
            "exam_generations": _query_count(
                db.query(ObservabilityEvent).filter(
                    ObservabilityEvent.agent_id == "exam",
                    ObservabilityEvent.event_type == "task_complete",
                )
            ),
            "mcqs_generated": _query_count(
                db.query(ObservabilityEvent).filter(
                    ObservabilityEvent.agent_id == "exam",
                    ObservabilityEvent.summary.ilike("%mcq%"),
                )
            ),
            "mcqs_attempted": total_questions,
            "exam_attempts": total_tests,
            "avg_accuracy": round((total_correct / total_questions) * 100, 1) if total_questions else None,
            "grounded_answer_rate": quality["grounded_answer_rate"],
        },
        "operations": {
            "failed_requests": failed_requests,
            "avg_latency_ms": observed.get("avg_turn_latency_ms") or observed.get("avg_model_latency_ms"),
            "model_calls": observed.get("model_calls"),
            "tool_calls": observed.get("tool_calls"),
            "turns": observed.get("turns"),
            "estimated_input_tokens": _admin_int(total_input_tokens),
            "estimated_output_tokens": _admin_int(total_output_tokens),
            "estimated_cost_usd": round(_admin_float(total_cost), 8),
        },
        "data": {
            "total_study_data_indexed_bytes": data_registry["summary"]["total_file_size_bytes"],
            "total_study_data_indexed_label": data_registry["summary"]["total_file_size_label"],
            "total_chunks": data_registry["summary"]["chunks"],
            "total_embeddings": data_registry["summary"]["embeddings"],
            "last_data_sync": data_registry["summary"]["last_indexed_time"],
            "last_index_build": None,
            "vector_index_status": data_registry["vector_index"]["status"],
        },
        "no_data": {
            "has_users": bool(user_ids),
            "has_traces": bool(quality["samples"]),
            "has_vector_index": False,
        },
    }


def build_admin_students(db: Session, limit: int = 50, class_level: str = "") -> Dict[str, Any]:
    query = db.query(UserProgress)
    if class_level:
        query = query.join(UserProfile, UserProfile.user_id == UserProgress.user_id).filter(
            UserProfile.class_level == class_level
        )
    users = query.order_by(
        func.coalesce(UserProgress.last_active_date, date(1970, 1, 1)).desc(),
        UserProgress.xp.desc(),
    ).limit(limit).all()
    profiles = {
        profile.user_id: profile
        for profile in db.query(UserProfile).filter(
            UserProfile.user_id.in_([user.user_id for user in users])
        ).all()
    } if users else {}
    rows: List[Dict[str, Any]] = []
    for user in users:
        sessions = (
            db.query(TestHistory)
            .filter(TestHistory.user_id == user.user_id)
            .order_by(TestHistory.id.desc())
            .limit(5)
            .all()
        )
        weak_topics = (
            db.query(TopicPerformance)
            .filter(TopicPerformance.user_id == user.user_id, TopicPerformance.weak == True)  # noqa: E712
            .order_by(TopicPerformance.last_practiced.desc())
            .limit(5)
            .all()
        )
        recent_question = (
            db.query(AICoachInteraction)
            .filter(AICoachInteraction.user_id == user.user_id, AICoachInteraction.role == "user")
            .order_by(AICoachInteraction.created_at.desc())
            .first()
        )
        rows.append(
            {
                "user_id": user.user_id,
                "display_name": profiles.get(user.user_id).display_name if profiles.get(user.user_id) else "",
                "class_level": profiles.get(user.user_id).class_level if profiles.get(user.user_id) else "",
                "onboarding_completed": (
                    bool(profiles.get(user.user_id).onboarding_completed)
                    if profiles.get(user.user_id)
                    else False
                ),
                "sessions": _query_count(db.query(TestHistory).filter(TestHistory.user_id == user.user_id)),
                "recent_sessions": [
                    format_test_session(
                        session,
                        profiles.get(user.user_id).class_level
                        if profiles.get(user.user_id)
                        else "",
                    )
                    for session in sessions
                ],
                "recent_question": recent_question.message[:220] if recent_question and recent_question.message else "",
                "topics_studied": sorted({session.topic for session in sessions if session.topic}),
                "exam_attempts": _admin_int(user.total_tests),
                "accuracy": round(float(user.accuracy), 1),
                "weak_topics": [
                    {
                        "topic": item.topic,
                        "accuracy": round(float(item.accuracy), 1),
                        "attempts": _admin_int(item.attempts),
                    }
                    for item in weak_topics
                ],
                "xp": _admin_int(user.xp),
                "streak": _admin_int(user.streak),
                "last_activity": user.last_active_date.isoformat() if user.last_active_date else None,
            }
        )

    return {"students": rows, "total": _query_count(db.query(UserProgress))}


def build_admin_system_health(db: Session) -> Dict[str, Any]:
    started = time.perf_counter()
    database_ok = check_db_health()
    db_latency_ms = round((time.perf_counter() - started) * 1000, 1)
    data_dir_ok = os.path.isdir(ADMIN_DATA_DIR)
    llm_configured = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "services": [
            {"name": "Backend", "status": "live", "detail": "FastAPI process responded"},
            {"name": "Database", "status": "ready" if database_ok else "error", "latency_ms": db_latency_ms},
            {"name": "Auth", "status": "ready" if security.firebase_ready() else "degraded", "detail": security.firebase_error() or "Firebase Admin ready"},
            {"name": "LLM provider", "status": "configured" if llm_configured else "missing_config", "detail": coach_settings.provider},
            {"name": "Embedding provider", "status": "not_configured", "detail": "No embedding provider endpoint is configured"},
            {"name": "Vector index/RAG", "status": "source_only", "detail": "Retriever uses platform source files; no vector index is connected"},
            {"name": "Storage", "status": "ready" if data_dir_ok else "error", "detail": ADMIN_DATA_DIR},
            {"name": "Knowledge graph", "status": "ready" if knowledge_graph.list_chapters() else "empty", "detail": f"{len(knowledge_graph.concepts)} concepts loaded"},
        ],
        "environment": {
            "version": "2.5.0-production-guardrails",
            "python_env": os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or "local",
            "rate_limits": config.RATE_LIMIT_ENABLED,
            "cors_origins_configured": len(config.ALLOWED_ORIGINS),
        },
        "metrics": {
            "avg_api_latency_ms": get_observability_summary(db).get("avg_turn_latency_ms"),
            "error_rate": model_failure_rate(db),
            "queue_jobs": None,
        },
    }


def _start_of_today() -> datetime:
    return datetime.combine(date.today(), datetime_time.min)


def _safe_count(query) -> int:
    try:
        return int(query.count() or 0)
    except Exception:
        return 0


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _metric(value: Optional[float | int | str], *, source: str = "db", unit: str = "", note: str = "") -> Dict[str, Any]:
    return {
        "value": value,
        "source": source,
        "unit": unit,
        "note": note,
    }




ADMIN_AGENT_DEFINITIONS = [
    {
        "agent_id": "orchestrator",
        "display_name": "Supervisor Orchestrator",
        "role": "Routes tasks, policies, traces, and handoffs.",
        "keywords": ["orchestrator", "router", "route", "gateway", "handoff"],
    },
    {
        "agent_id": "tutor",
        "display_name": "Subject Tutor",
        "role": "Answers doubts with grounded study context.",
        "keywords": ["tutor", "answer", "explain", "study", "doubt", "coach"],
    },
    {
        "agent_id": "revision",
        "display_name": "Revision Specialist",
        "role": "Generates notes, recall, summaries, and quick review.",
        "keywords": ["revision", "revise", "summary", "notes", "recall"],
    },
    {
        "agent_id": "exam",
        "display_name": "Exam Generator",
        "role": "Builds MCQs, tests, probable questions, and scoring.",
        "keywords": ["exam", "mcq", "test", "quiz", "question"],
    },
    {
        "agent_id": "planner",
        "display_name": "Study Planner",
        "role": "Chooses next best action from progress and weak topics.",
        "keywords": ["planner", "plan", "mission", "path", "next"],
    },
    {
        "agent_id": "coach",
        "display_name": "Personal AI Coach",
        "role": "Maintains continuity, memory, motivation, and progress.",
        "keywords": ["coach", "memory", "profile", "daily", "motivation"],
    },
]


def _agent_definition(agent_id: str) -> Dict[str, Any]:
    return next((item for item in ADMIN_AGENT_DEFINITIONS if item["agent_id"] == agent_id), {
        "agent_id": agent_id,
        "display_name": agent_id.replace("_", " ").title(),
        "role": "Observed runtime agent.",
        "keywords": [agent_id],
    })


def _agent_match_text(row: Any) -> str:
    metadata = _safe_json_dict(getattr(row, "metadata_json", {}) or getattr(row, "data_json", {}) or {})
    try:
        metadata_text = json.dumps(metadata, default=str)
    except TypeError:
        metadata_text = str(metadata)
    pieces = [
        getattr(row, "agent_id", ""),
        getattr(row, "event_type", ""),
        getattr(row, "summary", ""),
        getattr(row, "name", ""),
        getattr(row, "trace_type", ""),
        getattr(row, "provider", ""),
        getattr(row, "model", ""),
        metadata_text,
    ]
    return " ".join(str(piece or "") for piece in pieces).lower()


def _matches_agent(row: Any, agent_id: str) -> bool:
    text_value = _agent_match_text(row)
    definition = _agent_definition(agent_id)
    return any(str(keyword).lower() in text_value for keyword in definition.get("keywords", [agent_id]))


def _safe_created_at(row: Any) -> Optional[datetime]:
    value = getattr(row, "created_at", None) or getattr(row, "timestamp", None)
    return value if isinstance(value, datetime) else None


def _average(values: List[float]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def _quality_scores_from_traces(rows: List[ModelToolTrace]) -> List[float]:
    scores: List[float] = []
    for row in rows:
        metadata = _safe_json_dict(row.metadata_json)
        quality = _safe_json_dict(metadata.get("quality"))
        score = quality.get("score") or quality.get("overall_score")
        if score is None and "passed" in quality:
            score = 1.0 if quality.get("passed") else 0.0
        if score is not None:
            numeric = _admin_float(score)
            scores.append(numeric if numeric <= 1 else numeric / 100)
    return scores


def _agent_source_usage(rows: List[ModelToolTrace]) -> List[Dict[str, Any]]:
    usage: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        metadata = _safe_json_dict(row.metadata_json)
        retrieval = _safe_json_dict(metadata.get("retrieval"))
        source = (
            retrieval.get("source")
            or retrieval.get("section_id")
            or metadata.get("source")
            or metadata.get("section_id")
            or "runtime_trace"
        )
        key = str(source)
        if key not in usage:
            usage[key] = {"source": key, "chunks": 0, "rows": 0}
        usage[key]["rows"] += 1
        usage[key]["chunks"] += _admin_int(retrieval.get("paragraphs_found") or retrieval.get("chunks") or 0)
    return sorted(usage.values(), key=lambda item: (item["chunks"], item["rows"]), reverse=True)[:6]


def _agent_learning_signal(quality_delta: Optional[float], latency_delta_ms: Optional[float], errors: int, runs_24h: int) -> str:
    if runs_24h == 0:
        return "needs_data"
    if errors > 0 and quality_delta is not None and quality_delta < -0.02:
        return "regressing"
    if quality_delta is not None and quality_delta > 0.02:
        return "improving"
    if latency_delta_ms is not None and latency_delta_ms < -250:
        return "faster"
    return "stable"


def build_admin_agent_intelligence(db: Session, agent_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.utcnow()
    cutoff_24h = now - timedelta(hours=24)
    previous_cutoff = now - timedelta(hours=48)
    trace_rows = db.query(ModelToolTrace).order_by(ModelToolTrace.created_at.desc(), ModelToolTrace.id.desc()).limit(3500).all()
    event_rows = db.query(ObservabilityEvent).order_by(ObservabilityEvent.created_at.desc(), ObservabilityEvent.id.desc()).limit(3500).all()
    interaction_rows = db.query(AICoachInteraction).order_by(AICoachInteraction.created_at.desc(), AICoachInteraction.id.desc()).limit(2000).all()

    bus_agents = {str(row.get("agent_id") or row.get("id") or ""): row for row in agent_rows if isinstance(row, dict)}
    agent_ids = list(dict.fromkeys([item["agent_id"] for item in ADMIN_AGENT_DEFINITIONS] + list(bus_agents.keys())))
    enriched: List[Dict[str, Any]] = []

    for agent_id in agent_ids:
        if not agent_id:
            continue
        definition = _agent_definition(agent_id)
        bus_agent = bus_agents.get(agent_id, {})
        matched_traces = [row for row in trace_rows if _matches_agent(row, agent_id)]
        matched_events = [row for row in event_rows if _matches_agent(row, agent_id)]
        current_traces = [row for row in matched_traces if (_safe_created_at(row) or now) >= cutoff_24h]
        previous_traces = [
            row for row in matched_traces
            if previous_cutoff <= (_safe_created_at(row) or now) < cutoff_24h
        ]
        current_events = [row for row in matched_events if (_safe_created_at(row) or now) >= cutoff_24h]
        previous_events = [
            row for row in matched_events
            if previous_cutoff <= (_safe_created_at(row) or now) < cutoff_24h
        ]
        success_rows = [row for row in matched_traces if str(row.status or "").lower() in {"success", "skipped"}]
        error_rows = [row for row in matched_traces if str(row.status or "").lower() not in {"success", "skipped"}]
        latency_values = [_admin_float(row.latency_ms) for row in matched_traces if row.latency_ms is not None]
        current_latency = _average([_admin_float(row.latency_ms) for row in current_traces if row.latency_ms is not None])
        previous_latency = _average([_admin_float(row.latency_ms) for row in previous_traces if row.latency_ms is not None])
        quality_current = _average(_quality_scores_from_traces(current_traces))
        quality_previous = _average(_quality_scores_from_traces(previous_traces))

        if agent_id in {"coach", "tutor"} and quality_current is None:
            current_quality_rows = [
                float(row.quality_score or 0)
                for row in interaction_rows
                if row.created_at and row.created_at >= cutoff_24h and row.quality_score is not None and float(row.quality_score or 0) > 0
            ]
            quality_current = _average(current_quality_rows)
        if agent_id in {"coach", "tutor"} and quality_previous is None:
            previous_quality_rows = [
                float(row.quality_score or 0)
                for row in interaction_rows
                if row.created_at and previous_cutoff <= row.created_at < cutoff_24h and row.quality_score is not None and float(row.quality_score or 0) > 0
            ]
            quality_previous = _average(previous_quality_rows)

        quality_delta = round(quality_current - quality_previous, 3) if quality_current is not None and quality_previous is not None else None
        latency_delta = round(current_latency - previous_latency, 1) if current_latency is not None and previous_latency is not None else None
        total_requests = max(_admin_int(bus_agent.get("total_requests")), len(matched_traces) + len(matched_events))
        total_errors = max(_admin_int(bus_agent.get("total_errors")), len(error_rows))
        success_rate = round((len(success_rows) / len(matched_traces)) * 100, 1) if matched_traces else _admin_float(bus_agent.get("success_rate"))
        avg_latency = round(sum(latency_values) / len(latency_values), 1) if latency_values else _admin_float(bus_agent.get("avg_latency_ms"))
        last_activity = max(
            [item for item in [_safe_created_at(row) for row in matched_traces + matched_events] if item],
            default=None,
        )
        input_tokens = sum(_admin_int(row.estimated_input_tokens) for row in matched_traces)
        output_tokens = sum(_admin_int(row.estimated_output_tokens) for row in matched_traces)
        estimated_cost = round(sum(_admin_float(row.estimated_cost_usd) for row in matched_traces), 8)
        sessions = {row.session_id for row in matched_traces if row.session_id}
        model_calls = len([row for row in matched_traces if row.trace_type == "model"])
        tool_calls = len([row for row in matched_traces if row.trace_type == "tool"])
        turn_calls = len([row for row in matched_traces if row.trace_type == "turn"])
        memory_rows = 0
        if agent_id == "coach":
            memory_rows = _safe_count(db.query(AICoachMemory)) + _safe_count(db.query(AgentChatMemory))

        enriched.append({
            **bus_agent,
            "agent_id": agent_id,
            "display_name": str(bus_agent.get("display_name") or bus_agent.get("name") or definition["display_name"]),
            "role": str(bus_agent.get("role") or definition["role"]),
            "status": str(bus_agent.get("status") or ("observing" if total_requests else "not_reporting")),
            "health": str(bus_agent.get("health") or ("healthy" if total_errors == 0 and total_requests else "idle")),
            "current_task": str(bus_agent.get("current_task") or "Watching traces, data usage, model calls, and quality drift."),
            "last_activity": _admin_iso(last_activity) if last_activity else str(bus_agent.get("last_activity") or ""),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "total_success": max(_admin_int(bus_agent.get("total_success")), len(success_rows)),
            "avg_latency_ms": avg_latency,
            "last_quality_score": quality_current if quality_current is not None else _admin_float(bus_agent.get("last_quality_score")),
            "success_rate": success_rate,
            "data_intake": {
                "trace_rows": len(matched_traces),
                "event_rows": len(matched_events),
                "new_trace_rows_24h": len(current_traces),
                "previous_trace_rows_24h": len(previous_traces),
                "new_event_rows_24h": len(current_events),
                "previous_event_rows_24h": len(previous_events),
                "model_calls": model_calls,
                "tool_calls": tool_calls,
                "turns": turn_calls,
                "sessions": len(sessions),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": estimated_cost,
                "memory_rows": memory_rows,
                "sources": _agent_source_usage(matched_traces),
            },
            "evolution": {
                "version": str(bus_agent.get("version") or f"observed-v{1 + min(8, total_requests // 250)}"),
                "runs_total": len(matched_traces),
                "runs_24h": len(current_traces),
                "runs_previous_24h": len(previous_traces),
                "events_24h": len(current_events),
                "quality_score_current": quality_current,
                "quality_score_previous": quality_previous,
                "quality_delta": quality_delta,
                "latency_current_ms": current_latency,
                "latency_previous_ms": previous_latency,
                "latency_delta_ms": latency_delta,
                "success_rate": success_rate,
                "learning_signal": _agent_learning_signal(quality_delta, latency_delta, len(error_rows), len(current_traces)),
                "trained_on_samples": len(matched_traces) + len(matched_events) + memory_rows,
                "new_data_rows": len(current_traces) + len(current_events),
                "historical_data_rows": max(0, len(matched_traces) + len(matched_events) - len(current_traces) - len(current_events)),
            },
        })

    return enriched


def build_admin_data_intake(
    db: Session,
    *,
    data_registry: Dict[str, Any],
    model_registry: Dict[str, Any],
    chapters: List[ContentChapter],
    recent_content_jobs: List[ContentIngestionJob],
) -> Dict[str, Any]:
    now = datetime.utcnow()
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)
    total_traces = _safe_count(db.query(ModelToolTrace))
    traces_24h = _safe_count(db.query(ModelToolTrace).filter(ModelToolTrace.created_at >= cutoff_24h))
    traces_7d = _safe_count(db.query(ModelToolTrace).filter(ModelToolTrace.created_at >= cutoff_7d))
    total_events = _safe_count(db.query(ObservabilityEvent))
    events_24h = _safe_count(db.query(ObservabilityEvent).filter(ObservabilityEvent.created_at >= cutoff_24h))
    events_7d = _safe_count(db.query(ObservabilityEvent).filter(ObservabilityEvent.created_at >= cutoff_7d))
    total_chunks = _safe_count(db.query(ContentChunk))
    summary = data_registry.get("summary", {})
    current_model = model_registry.get("current", {})
    total_input_tokens = int(_safe_scalar(db.query(func.sum(ModelToolTrace.estimated_input_tokens)).scalar()))
    total_output_tokens = int(_safe_scalar(db.query(func.sum(ModelToolTrace.estimated_output_tokens)).scalar()))
    total_cost = round(_safe_scalar(db.query(func.sum(ModelToolTrace.estimated_cost_usd)).scalar()), 8)
    latest_trace = db.query(ModelToolTrace).order_by(ModelToolTrace.created_at.desc(), ModelToolTrace.id.desc()).first()
    latest_event = db.query(ObservabilityEvent).order_by(ObservabilityEvent.created_at.desc(), ObservabilityEvent.id.desc()).first()

    return {
        "generated_at": now.isoformat(),
        "totals": {
            "study_materials": _admin_int(summary.get("study_materials")),
            "study_material_bytes": _admin_int(summary.get("total_file_size_bytes")),
            "study_material_label": summary.get("total_file_size_label") or "0 B",
            "content_chapters": len(chapters),
            "approved_chapters": len([chapter for chapter in chapters if chapter.status in {"approved", "published"}]),
            "content_chunks": total_chunks or _admin_int(summary.get("chunks")),
            "topics": _admin_int(summary.get("topics")),
            "trace_rows": total_traces,
            "event_rows": total_events,
            "model_calls": _safe_count(db.query(ModelToolTrace).filter(ModelToolTrace.trace_type == "model")),
            "tool_calls": _safe_count(db.query(ModelToolTrace).filter(ModelToolTrace.trace_type == "tool")),
            "turns": _safe_count(db.query(ModelToolTrace).filter(ModelToolTrace.trace_type == "turn")),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost_usd": total_cost,
            "model_versions": len(model_registry.get("versions", [])),
            "training_samples": _admin_int(current_model.get("samples")) or total_traces,
        },
        "freshness": {
            "traces_24h": traces_24h,
            "traces_7d": traces_7d,
            "events_24h": events_24h,
            "events_7d": events_7d,
            "historical_traces": max(0, total_traces - traces_24h),
            "historical_events": max(0, total_events - events_24h),
            "latest_trace_at": _admin_iso(latest_trace.created_at) if latest_trace else None,
            "latest_event_at": _admin_iso(latest_event.created_at) if latest_event else None,
            "last_indexed_time": summary.get("last_indexed_time"),
        },
        "pipeline": {
            **data_registry.get("progress", {}),
            "rag_index_version": hashlib_data_version(),
            "prompt_version": current_model.get("prompt_version"),
            "agent_workflow_version": current_model.get("agent_workflow_version"),
        },
        "source_coverage": data_registry.get("source_coverage", []),
        "recent_jobs": [
            {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status,
                "source_path": job.source_path,
                "created_at": job.created_at.isoformat() if job.created_at else "",
                "summary": job.summary or {},
            }
            for job in recent_content_jobs
        ],
    }


def build_admin_console_payload(db: Session) -> Dict[str, Any]:
    today_start = _start_of_today()
    cutoff_24h = datetime.utcnow() - timedelta(hours=24)

    interactions_today = db.query(AICoachInteraction).filter(AICoachInteraction.created_at >= today_start)
    interactions_24h = db.query(AICoachInteraction).filter(AICoachInteraction.created_at >= cutoff_24h)
    tests_today = db.query(TestHistory).filter(TestHistory.date == date.today())
    traces_24h = db.query(ModelToolTrace).filter(ModelToolTrace.created_at >= cutoff_24h)
    events_24h = db.query(ObservabilityEvent).filter(ObservabilityEvent.created_at >= cutoff_24h)

    active_sessions_today = int(
        interactions_today.with_entities(
            func.count(distinct(AICoachInteraction.metadata_json["session_id"].as_string()))
        )
        .filter(AICoachInteraction.metadata_json["session_id"].as_string() != "")
        .scalar()
        or 0
    )
    test_users_today = {
        user_id for (user_id,) in tests_today.with_entities(TestHistory.user_id).distinct().all() if user_id
    }
    coach_users_today = {
        user_id for (user_id,) in interactions_today.with_entities(AICoachInteraction.user_id).distinct().all() if user_id
    }

    total_questions_today = _safe_count(interactions_today.filter(AICoachInteraction.role == "user"))
    exam_generations_today = _safe_count(tests_today.filter(TestHistory.session_type == "exam"))
    mcq_attempted_today = int(tests_today.with_entities(func.sum(TestHistory.total_questions)).scalar() or 0)
    avg_accuracy_today = tests_today.with_entities(func.avg(TestHistory.accuracy_rate)).scalar()
    quota_today = int(
        db.query(func.sum(DailyQuotaUsage.count))
        .filter(DailyQuotaUsage.quota_date == date.today())
        .scalar()
        or 0
    )
    estimated_cost_24h = round(_safe_scalar(traces_24h.with_entities(func.sum(ModelToolTrace.estimated_cost_usd)).scalar()), 8)
    total_tokens_24h = int(
        _safe_scalar(traces_24h.with_entities(func.sum(ModelToolTrace.estimated_input_tokens + ModelToolTrace.estimated_output_tokens)).scalar())
    )
    avg_latency_24h = traces_24h.with_entities(func.avg(ModelToolTrace.latency_ms)).scalar()
    failures_24h = _safe_count(events_24h.filter(ObservabilityEvent.severity.in_(["error", "critical"]))) + _safe_count(
        traces_24h.filter(ModelToolTrace.status.notin_(["success", "skipped"]))
    )

    turn_traces = traces_24h.filter(ModelToolTrace.trace_type == "turn").order_by(ModelToolTrace.id.desc()).limit(500).all()
    grounded_candidates = [
        row for row in turn_traces
        if "retrieval" in (row.metadata_json or {}) or "ground" in json.dumps(row.metadata_json or {}).lower()
    ]
    grounded_rate = round((len(grounded_candidates) / len(turn_traces)) * 100, 1) if turn_traces else None

    quality_scores = [
        float(score)
        for (score,) in interactions_24h.with_entities(AICoachInteraction.quality_score).all()
        if score is not None and float(score or 0) > 0
    ]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None
    low_quality_count = len([score for score in quality_scores if score < 0.65])
    slow_trace_count = _safe_count(traces_24h.filter(ModelToolTrace.latency_ms >= 7000))
    failed_mcq_events = _safe_count(
        events_24h
        .filter(ObservabilityEvent.severity.in_(["error", "critical"]))
        .filter(ObservabilityEvent.summary.ilike("%mcq%"))
    )
    missing_source_events = _safe_count(events_24h.filter(ObservabilityEvent.summary.ilike("%source%")))
    fallback_events = _safe_count(events_24h.filter(ObservabilityEvent.summary.ilike("%fallback%")))
    empty_retrieval_events = _safe_count(events_24h.filter(ObservabilityEvent.summary.ilike("%retrieval%")).filter(ObservabilityEvent.summary.ilike("%empty%")))

    chapters = db.query(ContentChapter).all()
    published_count = len([chapter for chapter in chapters if chapter.status in {"approved", "published"}])
    content_quality = round(
        sum(float(chapter.coverage_score or 0.0) for chapter in chapters) / len(chapters),
        3,
    ) if chapters else None
    recent_content_jobs = db.query(ContentIngestionJob).order_by(ContentIngestionJob.id.desc()).limit(8).all()

    system_stats = event_bus.get_system_stats()
    observability = get_observability_summary(db)
    data_registry = build_admin_data_registry(db)
    model_registry = build_admin_model_registry(db)
    agent_rows = build_admin_agent_intelligence(db, event_bus.get_all_agents())
    data_intake = build_admin_data_intake(
        db,
        data_registry=data_registry,
        model_registry=model_registry,
        chapters=chapters,
        recent_content_jobs=recent_content_jobs,
    )

    recent_traces = db.query(ModelToolTrace).order_by(ModelToolTrace.created_at.desc(), ModelToolTrace.id.desc()).limit(80).all()
    recent_students = db.query(UserProgress).order_by(UserProgress.last_active_date.desc(), UserProgress.xp.desc()).limit(20).all()
    recent_profiles = {
        profile.user_id: profile
        for profile in db.query(UserProfile).filter(
            UserProfile.user_id.in_([student.user_id for student in recent_students])
        ).all()
    } if recent_students else {}
    recent_audits = db.query(AdminAuditLog).order_by(AdminAuditLog.id.desc()).limit(40).all()
    recent_events = get_recent_observability_events(db, limit=80)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "environment": os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or os.getenv("ENV") or "development",
        "header": {
            "system_status": system_stats.get("uptime_status") or "online",
            "backend_ready": check_db_health(),
            "database_status": "ready" if check_db_health() else "degraded",
            "auth_status": "configured" if security.firebase_ready() else "not_configured",
            "llm_status": "configured" if os.getenv("GROQ_API_KEY") else "not_configured",
            "rag_status": "ready" if published_count else "waiting_for_approved_content",
            "last_sync_time": datetime.utcnow().isoformat(),
        },
        "overview": {
            "total_users": _metric(_safe_count(db.query(UserProgress)), source="user_progress"),
            "active_students_today": _metric(len(coach_users_today | test_users_today), source="coach_interactions+test_history"),
            "active_sessions": _metric(active_sessions_today, source="coach_interaction_metadata"),
            "questions_asked": _metric(total_questions_today, source="ai_coach_interactions", unit="today"),
            "revision_generations": _metric(None, source="not_instrumented", note="Revision generation needs a dedicated event counter."),
            "exam_generations": _metric(exam_generations_today, source="test_history", unit="today"),
            "mcqs_attempted": _metric(mcq_attempted_today, source="test_history", unit="today"),
            "average_accuracy": _metric(round(float(avg_accuracy_today), 1) if avg_accuracy_today is not None else None, source="test_history", unit="%"),
            "api_llm_usage": _metric(quota_today, source="daily_quota_usage", unit="requests_today"),
            "llm_tokens": _metric(total_tokens_24h, source="model_tool_traces", unit="24h"),
            "llm_cost": _metric(estimated_cost_24h, source="model_tool_traces", unit="usd_24h"),
            "errors_failures": _metric(failures_24h, source="observability_events+model_tool_traces", unit="24h"),
            "avg_latency": _metric(round(float(avg_latency_24h), 1) if avg_latency_24h is not None else None, source="model_tool_traces", unit="ms_24h"),
            "grounded_answer_rate": _metric(grounded_rate, source="model_tool_traces", unit="%"),
        },
        "agents": agent_rows,
        "system": {
            "event_bus": system_stats,
            "observability": observability,
            "services": [
                {"name": "Backend live/ready", "status": "ready" if check_db_health() else "degraded"},
                {"name": "Database", "status": "ready" if check_db_health() else "degraded"},
                {"name": "Firebase Auth", "status": "configured" if security.firebase_ready() else "not_configured"},
                {"name": "LLM provider", "status": "configured" if os.getenv("GROQ_API_KEY") else "not_configured"},
                {"name": "RAG/content index", "status": "ready" if published_count else "waiting_for_approved_content"},
                {"name": "Storage", "status": "ready" if os.path.isdir(ADMIN_DATA_DIR) else "missing"},
            ],
        },
        "quality": {
            "avg_quality_score": avg_quality,
            "low_quality_answers": low_quality_count,
            "hallucination_risk": missing_source_events + fallback_events,
            "missing_sources": missing_source_events,
            "failed_mcq_generation": failed_mcq_events,
            "empty_retrieval": empty_retrieval_events,
            "slow_responses": slow_trace_count,
            "fallback_used": fallback_events,
            "badges": {
                "grounded": len(grounded_candidates),
                "verified": len([score for score in quality_scores if score >= 0.82]),
                "needs_review": low_quality_count + missing_source_events,
                "failed": failures_24h,
                "fallback_used": fallback_events,
            },
        },
        "content": {
            "chapters_total": len(chapters),
            "approved_or_published": published_count,
            "coverage_score_avg": content_quality,
            "status_counts": {
                status_name: len([chapter for chapter in chapters if chapter.status == status_name])
                for status_name in sorted({chapter.status for chapter in chapters if chapter.status})
            },
            "recent_jobs": [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "status": job.status,
                    "source_path": job.source_path,
                    "created_at": job.created_at.isoformat() if job.created_at else "",
                    "summary": job.summary or {},
                }
                for job in recent_content_jobs
            ],
        },
        "students": [
            _student_payload(row, recent_profiles.get(row.user_id))
            for row in recent_students
        ],
        "traces": [_trace_payload(row) for row in recent_traces],
        "events": recent_events,
        "audit": [_serialize_audit_log(row) for row in recent_audits],
        "data_intake": data_intake,
        "data_registry": data_registry,
        "model_registry": model_registry,
    }
