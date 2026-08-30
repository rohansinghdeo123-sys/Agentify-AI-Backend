"""Read-only semantic retrieval contract checks for health and boot logs."""

from __future__ import annotations

import re
from typing import Any, Dict

from sqlalchemy.orm import Session

from Logic import embeddings as embeddings_service
from models import ContentChunk


def _key(value: Any) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def retrieval_embedding_status(db: Session) -> Dict[str, Any]:
    sample = (
        db.query(ContentChunk)
        .filter(ContentChunk.embedding.isnot(None))
        .order_by(ContentChunk.id.desc())
        .first()
    )
    configured = embeddings_service.embeddings_enabled()
    configured_model = embeddings_service.embedding_model() if configured else ""
    configured_host = embeddings_service.embedding_endpoint_host() if configured else ""
    if sample is None or not sample.embedding:
        return {
            "status": "no_embedded_content",
            "configured": configured,
            "configured_model": configured_model,
            "configured_endpoint_host": configured_host,
            "stored_model": "",
            "stored_endpoint_host": "",
            "stored_dimensions": 0,
        }
    metadata = dict(sample.metadata_json or {})
    stored_model = str(metadata.get("embedding_model") or "")
    stored_host = str(metadata.get("embedding_endpoint_host") or "")
    stored_dimensions = len(sample.embedding)
    declared_dimensions = int(metadata.get("embedding_dimensions") or stored_dimensions)
    if declared_dimensions != stored_dimensions:
        status = "invalid_content_contract"
    elif not configured:
        status = "lexical_only"
    elif not stored_model:
        status = "unknown_stored_model"
    elif _key(configured_model) != _key(stored_model):
        status = "model_mismatch"
    elif stored_host and configured_host.lower() != stored_host.lower():
        status = "endpoint_mismatch"
    else:
        status = "ready"
    return {
        "status": status,
        "configured": configured,
        "configured_model": configured_model,
        "configured_endpoint_host": configured_host,
        "stored_model": stored_model,
        "stored_endpoint_host": stored_host,
        "stored_dimensions": stored_dimensions,
    }
