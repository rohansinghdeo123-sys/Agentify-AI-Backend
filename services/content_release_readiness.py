"""Cheap, public-safe readiness summary for the bundled Chemistry release.

The health endpoint needs to prove that the production database contains the
released curriculum without returning chapter text, internal paths, or admin
details.  All queries are bounded to the nine known Class XI Chemistry chapter
rows and use indexed foreign keys for the derived-layer counts.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Mapping

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import ContentChapter, ContentChunk, ContentConcept, ContentIngestionJob, ContentPage


RELEASE_SCOPE = "ncert_class_11_chemistry"
RELEASE_CHAPTER_NUMBERS = tuple(range(1, 10))
RELEASE_EXPECTED_INVENTORY: Dict[str, int] = {
    "chapters": 9,
    "pages": 313,
    "concepts": 83,
    "chunks": 751,
    "embedded_chunks": 751,
}

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CLASS_11_VALUES = ("11", "class 11", "class_11", "xi")


def unavailable_release_readiness(status: str = "database_unavailable") -> Dict[str, Any]:
    """Return the stable public contract when DB-backed checks cannot run."""

    return {
        "status": status,
        "scope": RELEASE_SCOPE,
        "expected": dict(RELEASE_EXPECTED_INVENTORY),
        "published": {
            "chapters": 0,
            "pages": 0,
            "concepts": 0,
            "chunks": 0,
            "embedded_chunks": 0,
        },
        "release": {
            "provenance": "unavailable",
            "digest": "",
            "restored_at": None,
        },
    }


def _count(db: Session, model: Any, chapter_ids: list[int]) -> int:
    if not chapter_ids:
        return 0
    return int(
        db.query(func.count(model.id))
        .filter(model.chapter_id.in_(chapter_ids))
        .scalar()
        or 0
    )


def _iso(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _chapter_numbers(scope: Mapping[str, Any]) -> list[int]:
    try:
        return sorted(int(value) for value in scope.get("chapter_numbers") or [])
    except (TypeError, ValueError):
        return []


def _is_chemistry_scope(scope: Mapping[str, Any]) -> bool:
    return (
        str(scope.get("board") or "NCERT").strip().lower() == "ncert"
        and str(scope.get("class_level") or "").strip().lower() in _CLASS_11_VALUES
        and str(scope.get("subject") or "").strip().lower() == "chemistry"
    )


def _release_provenance(job: ContentIngestionJob | None) -> Dict[str, Any]:
    if job is None:
        return {"provenance": "missing", "digest": "", "restored_at": None}

    summary = dict(job.summary or {})
    digest = str(summary.get("digest") or "").strip().lower()
    scope = dict(summary.get("scope") or {})
    inventory = dict(summary.get("inventory") or {})
    scope_matches = (
        _is_chemistry_scope(scope)
        and _chapter_numbers(scope) == list(RELEASE_CHAPTER_NUMBERS)
    )
    inventory_matches = all(
        int(inventory.get(key) or 0) == expected
        for key, expected in RELEASE_EXPECTED_INVENTORY.items()
        if key != "embedded_chunks"
    )
    provenance = (
        "verified"
        if _DIGEST_PATTERN.fullmatch(digest) and scope_matches and inventory_matches
        else "mismatch"
    )
    last_result = dict(summary.get("last_result") or {})
    return {
        "provenance": provenance,
        "digest": digest if _DIGEST_PATTERN.fullmatch(digest) else "",
        "restored_at": _iso(last_result.get("verified_at")) or _iso(job.updated_at),
    }


def content_release_readiness(db: Session) -> Dict[str, Any]:
    """Return exact, content-free production inventory and release provenance."""

    chapter_rows = (
        db.query(ContentChapter.id, ContentChapter.chapter_number)
        .filter(
            func.lower(ContentChapter.board) == "ncert",
            func.lower(ContentChapter.class_level).in_(_CLASS_11_VALUES),
            func.lower(ContentChapter.subject) == "chemistry",
            ContentChapter.chapter_number.in_(RELEASE_CHAPTER_NUMBERS),
            ContentChapter.status == "published",
        )
        .all()
    )
    chapter_ids = [int(row.id) for row in chapter_rows]
    published = {
        "chapters": len(chapter_rows),
        "pages": _count(db, ContentPage, chapter_ids),
        "concepts": _count(db, ContentConcept, chapter_ids),
        "chunks": _count(db, ContentChunk, chapter_ids),
        "embedded_chunks": (
            int(
                db.query(func.count(ContentChunk.id))
                .filter(
                    ContentChunk.chapter_id.in_(chapter_ids),
                    ContentChunk.embedding.isnot(None),
                )
                .scalar()
                or 0
            )
            if chapter_ids
            else 0
        ),
    }
    release_jobs = (
        db.query(ContentIngestionJob)
        .filter(
            ContentIngestionJob.job_type == "content_release_restore",
            ContentIngestionJob.status == "completed",
        )
        .order_by(ContentIngestionJob.updated_at.desc(), ContentIngestionJob.id.desc())
        .limit(20)
        .all()
    )
    release_job = next(
        (
            job
            for job in release_jobs
            if _is_chemistry_scope(dict((job.summary or {}).get("scope") or {}))
        ),
        None,
    )
    release = _release_provenance(release_job)
    inventory_ready = published == RELEASE_EXPECTED_INVENTORY
    if not inventory_ready:
        readiness_status = "incomplete"
    elif release["provenance"] == "verified":
        readiness_status = "ready"
    else:
        readiness_status = "content_ready_unverified_provenance"
    return {
        "status": readiness_status,
        "scope": RELEASE_SCOPE,
        "expected": dict(RELEASE_EXPECTED_INVENTORY),
        "published": published,
        "release": release,
    }
