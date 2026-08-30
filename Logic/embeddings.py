"""Embedding client and vector helpers for hybrid (semantic + lexical) retrieval.

Vectors are stored in ``ContentChunk.embedding`` (JSON list of floats), which
keeps the SQLite dev path and the Postgres prod path identical and requires no
pgvector extension. The approved-content search already loads every candidate
chunk into Python for lexical scoring, so dot-product scoring the same rows
adds negligible cost at the current corpus size (thousands of chunks). If the
corpus outgrows in-process scoring, migrate the column to pgvector and push
the similarity ordering into SQL — the call sites here are the only places
that need to change.

Configuration (any OpenAI-compatible /embeddings endpoint):
    EMBEDDINGS_API_KEY   (falls back to OPENAI_API_KEY)
    EMBEDDINGS_BASE_URL  (default https://api.openai.com/v1)
    EMBEDDINGS_MODEL     (default text-embedding-3-small)
    EMBEDDINGS_TIMEOUT_SECONDS (default 20)

When no API key is configured every entry point degrades gracefully and
retrieval stays lexical-only.
"""

from __future__ import annotations

import logging
import math
import operator
import os
import re
import threading
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger("ai_educator.embeddings")

_BATCH_SIZE = 96
_MAX_INPUT_CHARS = 8000  # stay well inside embedding-model token limits
_QUERY_CACHE_MAX = 256

_query_cache: "OrderedDict[Tuple[str, str], List[float]]" = OrderedDict()
_query_cache_lock = threading.Lock()


def _model_key(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _api_key() -> str:
    return (os.getenv("EMBEDDINGS_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()


def _base_url() -> str:
    return os.getenv("EMBEDDINGS_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def embedding_model() -> str:
    configured = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small").strip()
    # AgentifyAI's verified Chemistry release uses Google's current
    # ``gemini-embedding-001`` vector space.  Older Render environments may
    # still carry the predecessor ``text-embedding-004`` setting; using it for
    # queries would make every released vector incompatible.  Migrate only
    # that known legacy Google setting and preserve every other explicit
    # provider/model choice.
    if (
        _model_key(configured) == "text_embedding_004"
        and embedding_endpoint_host() == "generativelanguage.googleapis.com"
    ):
        return "gemini-embedding-001"
    return configured


def embedding_endpoint_host() -> str:
    return str(urlparse(_base_url()).hostname or "").lower()


def _timeout_seconds() -> float:
    try:
        return float(os.getenv("EMBEDDINGS_TIMEOUT_SECONDS", "20"))
    except ValueError:
        return 20.0


def embeddings_enabled() -> bool:
    return bool(_api_key())


def _is_transient_error(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in {429, 500, 502, 503, 504} or isinstance(
        exc, (requests.Timeout, requests.ConnectionError)
    )


@retry(
    retry=retry_if_exception(_is_transient_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=4.0),
    reraise=True,
)
def _embed_batch(batch: Sequence[str]) -> List[List[float]]:
    response = requests.post(
        f"{_base_url()}/embeddings",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={"model": embedding_model(), "input": list(batch)},
        timeout=_timeout_seconds(),
    )
    if not response.ok:
        # Surface the provider's actual message (e.g. "API key not valid") so
        # config mistakes are obvious in the logs instead of a bare status code.
        logger.warning("Embeddings API %s error: %s", response.status_code, response.text[:300])
    response.raise_for_status()
    data = response.json().get("data") or []
    ordered = sorted(data, key=lambda item: int(item.get("index") or 0))
    vectors = [normalize([float(value) for value in item.get("embedding") or []]) for item in ordered]
    if len(vectors) != len(batch) or any(not vector for vector in vectors):
        raise RuntimeError(
            f"Embedding response shape mismatch: sent {len(batch)} inputs, got {len(vectors)} vectors."
        )
    return vectors


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Embed texts in order. Raises when embeddings are unconfigured or the API fails."""
    if not embeddings_enabled():
        raise RuntimeError(
            "Embeddings are not configured. Set EMBEDDINGS_API_KEY (or OPENAI_API_KEY)."
        )
    prepared = [str(text or " ")[:_MAX_INPUT_CHARS] or " " for text in texts]
    vectors: List[List[float]] = []
    for start in range(0, len(prepared), _BATCH_SIZE):
        vectors.extend(_embed_batch(prepared[start:start + _BATCH_SIZE]))
    return vectors


def embed_query(text: str) -> Optional[List[float]]:
    """Embed one query, with a small LRU cache. Returns None when embeddings
    are unavailable so callers can fall back to lexical-only retrieval."""
    if not embeddings_enabled():
        return None
    normalized_text = " ".join(str(text or "").split())[:_MAX_INPUT_CHARS]
    if not normalized_text:
        return None
    key = (embedding_model(), normalized_text.lower())

    with _query_cache_lock:
        cached = _query_cache.get(key)
        if cached is not None:
            _query_cache.move_to_end(key)
            return cached

    try:
        vector = embed_texts([normalized_text])[0]
    except Exception as exc:
        logger.warning("Query embedding failed; retrieval continues lexical-only | error=%s", exc)
        return None

    with _query_cache_lock:
        _query_cache[key] = vector
        _query_cache.move_to_end(key)
        while len(_query_cache) > _QUERY_CACHE_MAX:
            _query_cache.popitem(last=False)
    return vector


def clear_query_cache() -> None:
    with _query_cache_lock:
        _query_cache.clear()


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return list(vector)
    return [value / norm for value in vector]


def similarity(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Cosine similarity for unit vectors (stored vectors are normalized at
    write time, so this is a plain dot product)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(map(operator.mul, a, b)))


def compatible_with_stored(
    query_vector: Optional[Sequence[float]],
    stored_vector: Optional[Sequence[float]],
    *,
    stored_model: str = "",
    stored_endpoint_host: str = "",
) -> bool:
    """Reject vectors from a different model or embedding endpoint."""
    if not query_vector or not stored_vector or len(query_vector) != len(stored_vector):
        return False
    if stored_model and _model_key(stored_model) != _model_key(embedding_model()):
        return False
    if (
        stored_endpoint_host
        and stored_endpoint_host.strip().lower() != embedding_endpoint_host()
    ):
        return False
    return True
