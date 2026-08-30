"""Application startup/shutdown wiring via a FastAPI lifespan context manager.

All side effects that previously ran at module import time (table creation, the
telemetry-column shim, event-bus sink wiring, Firebase initialization, and the
knowledge-graph load) now run here, once, when the ASGI server starts the app.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import inspect, text

import models  # noqa: F401  (ensures all tables are registered on Base.metadata)
from app import security
from app.telemetry import init_telemetry, shutdown_telemetry
from database import Base, engine
from Logic.agent_event_bus import event_bus
from Logic.knowledge_graph import knowledge_graph
from Logic.observability_store import persist_event_from_bus

logger = logging.getLogger("ai_educator.lifespan")

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KNOWLEDGE_GRAPH_PATH = os.path.join(_BACKEND_DIR, "data", "Chapters", "basic_concepts_of_chemistry.json")
_CONTENT_RELEASE_BUNDLE = (
    Path(_BACKEND_DIR) / "data" / "releases" / "ncert_class_11_chemistry_v1.json.gz"
)


def _ensure_session_telemetry_columns() -> None:
    """Backfill telemetry columns until the production Alembic pass lands."""
    ddl_by_column = {
        "started_at": "TIMESTAMP",
        "completed_at": "TIMESTAMP",
        "response_latency_ms": "INTEGER DEFAULT 0",
        "hint_count": "INTEGER DEFAULT 0",
        "retry_count": "INTEGER DEFAULT 0",
        "confidence_before": "FLOAT",
        "confidence_after": "FLOAT",
    }
    try:
        inspector = inspect(engine)
        if "test_history" not in inspector.get_table_names():
            return
        existing = {column["name"] for column in inspector.get_columns("test_history")}
        missing = [(name, ddl) for name, ddl in ddl_by_column.items() if name not in existing]
        if not missing:
            return
        with engine.begin() as conn:
            for name, ddl in missing:
                conn.execute(text(f"ALTER TABLE test_history ADD COLUMN {name} {ddl}"))
        logger.info("DATABASE: Added session telemetry columns: %s", ", ".join(name for name, _ in missing))
    except Exception as exc:
        logger.warning("DATABASE: Session telemetry column check skipped: %s", exc)


def _log_retrieval_mode() -> None:
    """Make the active retrieval mode obvious in the boot logs. Semantic search
    needs EMBEDDINGS_API_KEY; without it retrieval is lexical-only."""
    try:
        from database import SessionLocal
        from services.retrieval_readiness import retrieval_embedding_status

        db = SessionLocal()
        try:
            retrieval = retrieval_embedding_status(db)
        finally:
            db.close()
        if retrieval["status"] == "ready":
            logger.info(
                "RETRIEVAL: semantic search ENABLED (model=%s@%s, dimensions=%d).",
                retrieval["configured_model"],
                retrieval["configured_endpoint_host"],
                retrieval["stored_dimensions"],
            )
        else:
            logger.warning(
                "RETRIEVAL: semantic search %s; lexical retrieval remains available "
                "(configured_model=%s@%s, stored_model=%s@%s, stored_dimensions=%d).",
                retrieval["status"],
                retrieval["configured_model"] or "none",
                retrieval["configured_endpoint_host"] or "none",
                retrieval["stored_model"] or "none",
                retrieval["stored_endpoint_host"] or "none",
                retrieval["stored_dimensions"],
            )
    except Exception as exc:
        logger.warning("RETRIEVAL: embeddings status check skipped: %s", exc)


def _load_knowledge_graph() -> None:
    logger.info("Loading knowledge graph from: %s", _KNOWLEDGE_GRAPH_PATH)
    try:
        knowledge_graph.load_chapter(_KNOWLEDGE_GRAPH_PATH, "basic-concepts-of-chemistry")
        logger.info(
            "Knowledge graph loaded: basic-concepts-of-chemistry (%d concepts)",
            len(knowledge_graph.concepts),
        )
    except Exception as exc:
        logger.warning("Could not load basic-concepts-of-chemistry chapter: %s", exc)


def _restore_bundled_content_release() -> None:
    """Promote the verified curriculum snapshot on remote deployments.

    This lives in the application lifespan, instead of relying only on a
    Procfile/Docker command, because hosted dashboards can override those
    commands. SQLite development/test databases opt out by default; operators
    can explicitly set BUNDLED_CONTENT_BOOTSTRAP=true to exercise the same path.
    """
    from database import SessionLocal, USE_SQLITE
    from services.content_release_bundle import restore_content_release

    configured = os.getenv("BUNDLED_CONTENT_BOOTSTRAP")
    enabled = (
        configured.strip().lower() in {"1", "true", "yes", "on"}
        if configured is not None
        else not USE_SQLITE
    )
    if not enabled:
        logger.info("CONTENT RELEASE: bundled restore skipped for local SQLite.")
        return
    db = SessionLocal()
    try:
        result = restore_content_release(db, _CONTENT_RELEASE_BUNDLE)
    finally:
        db.close()
    logger.info(
        "CONTENT RELEASE: verified %s; restored=%d skipped=%d.",
        result["digest"],
        len(result["restored"]),
        len(result["skipped"]),
    )
    semantic_status = result["semantic_retrieval"]
    if semantic_status["status"] != "ready":
        logger.warning(
            "CONTENT RELEASE: semantic retrieval is %s (configured=%s@%s, stored=%s@%s); "
            "lexical retrieval remains available.",
            semantic_status["status"],
            semantic_status["configured_model"] or "none",
            semantic_status["configured_endpoint_host"] or "none",
            semantic_status["stored_model"],
            semantic_status["stored_endpoint_host"],
        )


@asynccontextmanager
async def lifespan(app):
    # ── startup ──────────────────────────────────────────────────────────
    # In production Alembic is the single schema authority (run `alembic
    # upgrade head` on deploy); create_all is a dev/test convenience only.
    app_env = (os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or os.getenv("ENV") or "").lower()
    default_auto = "false" if app_env in {"prod", "production", "staging"} else "true"
    if os.getenv("AUTO_CREATE_TABLES", default_auto).strip().lower() in {"1", "true", "yes", "on"}:
        Base.metadata.create_all(bind=engine)
        _ensure_session_telemetry_columns()
    else:
        logger.info("AUTO_CREATE_TABLES disabled; relying on Alembic migrations.")
    _restore_bundled_content_release()
    event_bus.set_sink(persist_event_from_bus)

    from database import SessionLocal
    from services.retention_service import prune_telemetry_safely

    prune_telemetry_safely(SessionLocal)

    security.initialize_firebase_admin()
    _load_knowledge_graph()
    _log_retrieval_mode()
    init_telemetry()

    from services.job_queue import job_queue

    job_queue.start()

    yield

    # ── shutdown ─────────────────────────────────────────────────────────
    job_queue.stop()
    shutdown_telemetry()
    engine.dispose()
