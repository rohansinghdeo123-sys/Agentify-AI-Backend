"""Liveness, readiness, and public health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app import config, security
from database import SessionLocal, check_db_health
from Logic.knowledge_graph import knowledge_graph
from Logic.tools.artifact_generator import available_artifact_sections
from services.content_release_readiness import (
    content_release_readiness,
    unavailable_release_readiness,
)
from services.retrieval_readiness import retrieval_embedding_status
from services.ttl_cache import TTLCache

router = APIRouter(tags=["health"])

_database_readiness_cache = TTLCache(max_entries=1)
_DATABASE_READINESS_TTL_SECONDS = 30.0


@router.get("/health/live")
def liveness_probe():
    return {"status": "ok", "service": "agentifyai-backend"}


@router.get("/health/ready")
def readiness_probe():
    import os

    db_ready = check_db_health()
    firebase_ready = bool(security.firebase_ready())
    llm_ready = bool(
        os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    knowledge_ready = bool(knowledge_graph.list_chapters())
    artifact_ready = bool(available_artifact_sections())
    semantic_retrieval = {
        "status": "database_unavailable",
        "configured": False,
        "configured_model": "",
        "configured_endpoint_host": "",
        "stored_model": "",
        "stored_endpoint_host": "",
        "stored_dimensions": 0,
    }
    chemistry_release = unavailable_release_readiness()
    if db_ready:
        try:
            def build_database_readiness():
                db = SessionLocal()
                try:
                    return {
                        "semantic_retrieval": retrieval_embedding_status(db),
                        "chemistry_release": content_release_readiness(db),
                    }
                finally:
                    db.close()

            database_readiness = _database_readiness_cache.get_or_build(
                "production_content",
                _DATABASE_READINESS_TTL_SECONDS,
                build_database_readiness,
            )
            semantic_retrieval = database_readiness["semantic_retrieval"]
            chemistry_release = database_readiness["chemistry_release"]
        except Exception:
            semantic_retrieval["status"] = "readiness_check_failed"
            chemistry_release = unavailable_release_readiness("readiness_check_failed")
    ready = db_ready and firebase_ready and llm_ready
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if ready else "degraded",
            "database": db_ready,
            "firebase": firebase_ready,
            "llm": llm_ready,
            "knowledge_graph": knowledge_ready,
            "artifacts": artifact_ready,
            "semantic_retrieval": semantic_retrieval,
            "chemistry_release": chemistry_release,
            "version": "2.5.0-production-guardrails",
        },
    )


@router.get("/health")
def health_check():
    return {
        "status": "online" if check_db_health() else "degraded",
        "version": "2.5.0-production-guardrails",
        "service": "agentifyai-backend",
        "request_ids": True,
        "rate_limits": config.RATE_LIMIT_ENABLED,
        "cors_origins_configured": len(config.ALLOWED_ORIGINS),
    }
