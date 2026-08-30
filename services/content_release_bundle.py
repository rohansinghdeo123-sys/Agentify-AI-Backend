"""Versioned export and restore for production-ready curriculum content.

The ingestion pipeline intentionally stores its derived corpus in the database.
This module turns an approved, fully embedded curriculum slice into a portable,
integrity-checked release artifact so a deployment can restore that exact slice
without repeating PDF extraction, LLM generation, or embedding calls.

Restores are idempotent. Existing chapter registry rows are updated in place so
student records that retain a chapter id remain valid; only that chapter's
derived pages, concepts, and chunks are replaced when the released source does
not already match the database.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from models import ContentChapter, ContentChunk, ContentConcept, ContentIngestionJob, ContentPage


BUNDLE_SCHEMA_VERSION = 1
LIVE_STATUSES = {"approved", "published"}

_CHAPTER_FIELDS = (
    "board",
    "class_level",
    "subject",
    "book_name",
    "chapter_number",
    "chapter_name",
    "slug",
    "source_hash",
    "status",
    "version",
    "published_source_hash",
    "page_count",
    "extracted_page_count",
    "chunk_count",
    "concept_count",
    "coverage_score",
    "extraction_quality",
    "validation_report",
)
_PAGE_FIELDS = (
    "page_number",
    "text",
    "char_count",
    "extraction_quality",
    "metadata_json",
)
_CONCEPT_FIELDS = (
    "concept_id",
    "title",
    "definition",
    "core_explanation",
    "key_points",
    "examples",
    "formulas",
    "properties",
    "applications",
    "common_mistakes",
    "prerequisites",
    "related_concepts",
    "learning_objectives",
    "source_pages",
    "difficulty_level",
    "blooms_taxonomy",
    "typical_exam_weightage",
    "importance_level",
    "raw_json",
    "validation_issues",
)
_CHUNK_FIELDS = (
    "chunk_id",
    "text",
    "page_start",
    "page_end",
    "section_title",
    "token_estimate",
    "lexical_terms",
    "embedding",
    "metadata_json",
)


class ContentReleaseError(RuntimeError):
    """Raised when a curriculum release cannot be trusted or restored."""


def _serialize(row: Any, fields: Iterable[str]) -> Dict[str, Any]:
    return {field: getattr(row, field) for field in fields}


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _with_digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return {**payload, "digest": f"sha256:{digest}"}


def _normalized_key(value: Any) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _class_key(value: Any) -> str:
    return _normalized_key(value).removeprefix("class_")


def _chapter_identity(value: Any) -> tuple[str, str, str, int]:
    getter = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
    return (
        _normalized_key(getter("board") or "NCERT"),
        _class_key(getter("class_level")),
        _normalized_key(getter("subject")),
        int(getter("chapter_number") or 0),
    )


def _version_number(value: Any) -> int:
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else 0


def _replacement_version(chapter: ContentChapter | None, chapter_data: Dict[str, Any]) -> str:
    bundled = str(chapter_data.get("version") or "v1")
    if chapter is None:
        return bundled
    if chapter.source_hash == chapter_data.get("source_hash"):
        return chapter.version or bundled
    existing_number = _version_number(chapter.version)
    bundled_number = _version_number(bundled)
    if existing_number or bundled_number:
        return f"v{max(existing_number, bundled_number) + 1}"
    return f"{bundled}-release"


def _validate_chapter_item(item: Dict[str, Any], embedding_contract: Dict[str, Any]) -> None:
    item_digest = str(item.get("digest") or "")
    digest_payload = {key: value for key, value in item.items() if key != "digest"}
    expected_digest = f"sha256:{hashlib.sha256(_canonical_bytes(digest_payload)).hexdigest()}"
    if item_digest != expected_digest:
        raise ContentReleaseError("A chapter in the content release failed its integrity check.")

    chapter = item.get("chapter") or {}
    slug = str(chapter.get("slug") or "").strip()
    pages = item.get("pages") or []
    concepts = item.get("concepts") or []
    chunks = item.get("chunks") or []
    if not slug or chapter.get("status") not in LIVE_STATUSES:
        raise ContentReleaseError("Every released chapter must have a live status and stable slug.")
    if not chapter.get("source_hash") or chapter.get("published_source_hash") != chapter.get("source_hash"):
        raise ContentReleaseError(f"{slug} is not bound to its approved source revision.")
    if not pages or not concepts or not chunks:
        raise ContentReleaseError(f"{slug} is missing a required derived content layer.")
    declared_counts = {
        "page_count": len(pages),
        "extracted_page_count": sum(1 for row in pages if int(row.get("char_count") or 0) > 0),
        "concept_count": len(concepts),
        "chunk_count": len(chunks),
    }
    for field, actual in declared_counts.items():
        if int(chapter.get(field) or 0) != actual:
            raise ContentReleaseError(f"{slug} declares an incorrect {field}.")

    page_numbers = [int(row.get("page_number") or 0) for row in pages]
    page_set = set(page_numbers)
    if any(number <= 0 for number in page_numbers) or len(page_set) != len(page_numbers):
        raise ContentReleaseError(f"{slug} has invalid or duplicate source pages.")

    concept_ids = [str(row.get("concept_id") or "").strip() for row in concepts]
    if any(not value for value in concept_ids) or len(set(concept_ids)) != len(concept_ids):
        raise ContentReleaseError(f"{slug} has missing or duplicate concept IDs.")
    for concept in concepts:
        refs = {int(value) for value in concept.get("source_pages") or []}
        if not refs or not refs.issubset(page_set):
            raise ContentReleaseError(f"{slug} has a concept with invalid source-page coverage.")

    chunk_ids = [str(row.get("chunk_id") or "").strip() for row in chunks]
    if any(not value for value in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise ContentReleaseError(f"{slug} has missing or duplicate chunk IDs.")
    expected_dimensions = int(embedding_contract.get("dimensions") or 0)
    for chunk in chunks:
        if not str(chunk.get("text") or "").strip():
            raise ContentReleaseError(f"{slug} has an empty retrieval chunk.")
        start = int(chunk.get("page_start") or 0)
        end = int(chunk.get("page_end") or start)
        if start not in page_set or end not in page_set or start > end:
            raise ContentReleaseError(f"{slug} has a chunk outside its source-page range.")
        vector = chunk.get("embedding") or []
        if len(vector) != expected_dimensions or any(not math.isfinite(float(value)) for value in vector):
            raise ContentReleaseError(f"{slug} has an invalid embedding vector.")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if not 0.99 <= norm <= 1.01:
            raise ContentReleaseError(f"{slug} has a non-normalized embedding vector.")

    from Logic.content_pipeline import build_coverage_report

    stored_report = chapter.get("validation_report") or {}
    report = build_coverage_report(pages, concepts, chunks, stored_report.get("issues") or [])
    if not report.get("ready_for_approval") or report.get("blocking_issue_count"):
        raise ContentReleaseError(f"{slug} no longer passes the publication quality gate.")
    if abs(float(chapter.get("coverage_score") or 0) - float(report["coverage_score"])) > 0.001:
        raise ContentReleaseError(f"{slug} declares an incorrect coverage score.")


def _validate_release_payload(payload: Dict[str, Any]) -> None:
    scope = payload.get("scope") or {}
    chapters = payload.get("chapters") or []
    embedding = payload.get("embedding") or {}
    if (
        not embedding.get("model")
        or not embedding.get("endpoint_host")
        or int(embedding.get("dimensions") or 0) <= 0
    ):
        raise ContentReleaseError("Content release has no valid embedding contract.")
    expected_numbers = sorted(int(value) for value in scope.get("chapter_numbers") or [])
    bundled_numbers = sorted(
        int(item.get("chapter", {}).get("chapter_number") or 0) for item in chapters
    )
    if bundled_numbers != expected_numbers or len(set(bundled_numbers)) != len(bundled_numbers):
        raise ContentReleaseError("Content release chapter inventory does not match its declared scope.")
    all_chunk_ids: list[str] = []
    for item in chapters:
        _validate_chapter_item(item, embedding)
        all_chunk_ids.extend(str(row["chunk_id"]) for row in item["chunks"])
    if len(set(all_chunk_ids)) != len(all_chunk_ids):
        raise ContentReleaseError("Content release contains duplicate chunk IDs across chapters.")


def _verified_payload(bundle_path: Path) -> Dict[str, Any]:
    if not bundle_path.is_file():
        raise ContentReleaseError(f"Content release bundle is missing: {bundle_path}")
    try:
        payload = json.loads(gzip.decompress(bundle_path.read_bytes()).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentReleaseError(f"Content release bundle is unreadable: {bundle_path}") from exc

    digest = str(payload.pop("digest", ""))
    expected = f"sha256:{hashlib.sha256(_canonical_bytes(payload)).hexdigest()}"
    if not digest or digest != expected:
        raise ContentReleaseError("Content release bundle failed its SHA-256 integrity check.")
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ContentReleaseError(
            f"Unsupported content release schema: {payload.get('schema_version')!r}"
        )
    chapters = payload.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ContentReleaseError("Content release bundle contains no chapters.")
    payload["digest"] = digest
    _validate_release_payload(payload)
    return payload


def export_content_release(
    db: Session,
    bundle_path: Path,
    *,
    class_level: str,
    subject: str,
    chapter_numbers: Sequence[int],
    embedding_model: str,
    embedding_dimensions: int,
    embedding_provider: str = "openai-compatible",
    embedding_endpoint_host: str = "",
) -> Dict[str, Any]:
    """Export one complete, live curriculum slice into a deterministic bundle."""
    required_numbers = sorted({int(number) for number in chapter_numbers})
    chapters = (
        db.query(ContentChapter)
        .filter(
            func.lower(ContentChapter.class_level) == class_level.strip().lower(),
            func.lower(ContentChapter.subject) == subject.strip().lower(),
            ContentChapter.chapter_number.in_(required_numbers),
            ContentChapter.status.in_(sorted(LIVE_STATUSES)),
        )
        .order_by(ContentChapter.chapter_number, ContentChapter.id)
        .all()
    )
    actual_numbers = [int(chapter.chapter_number or 0) for chapter in chapters]
    if actual_numbers != required_numbers:
        raise ContentReleaseError(
            f"Expected live chapters {required_numbers}, found {actual_numbers or 'none'}."
        )

    serialized_chapters = []
    for chapter in chapters:
        pages = (
            db.query(ContentPage)
            .filter(ContentPage.chapter_id == chapter.id)
            .order_by(ContentPage.page_number, ContentPage.id)
            .all()
        )
        concepts = (
            db.query(ContentConcept)
            .filter(ContentConcept.chapter_id == chapter.id)
            .order_by(ContentConcept.id)
            .all()
        )
        chunks = (
            db.query(ContentChunk)
            .filter(ContentChunk.chapter_id == chapter.id)
            .order_by(ContentChunk.page_start, ContentChunk.id)
            .all()
        )
        embedded = sum(1 for chunk in chunks if chunk.embedding)
        if not pages or not concepts or not chunks:
            raise ContentReleaseError(f"{chapter.slug} is missing a required derived content layer.")
        if embedded != len(chunks):
            raise ContentReleaseError(
                f"{chapter.slug} has {len(chunks) - embedded} chunks without embeddings."
            )
        if not chapter.source_hash or chapter.published_source_hash != chapter.source_hash:
            raise ContentReleaseError(f"{chapter.slug} is not bound to its approved source revision.")

        chapter_data = _serialize(chapter, _CHAPTER_FIELDS)
        chapter_data["pdf_path"] = (
            f"bundled://ncert/class_{class_level}/{subject.lower()}/chapter_{int(chapter.chapter_number):02d}"
        )
        serialized_chapters.append(
            _with_digest({
                "chapter": chapter_data,
                "pages": [_serialize(row, _PAGE_FIELDS) for row in pages],
                "concepts": [_serialize(row, _CONCEPT_FIELDS) for row in concepts],
                "chunks": [_serialize(row, _CHUNK_FIELDS) for row in chunks],
            })
        )

    payload = _with_digest(
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "scope": {
                "board": "NCERT",
                "class_level": class_level,
                "subject": subject,
                "chapter_numbers": required_numbers,
            },
            "embedding": {
                "provider": embedding_provider,
                "model": embedding_model,
                "endpoint_host": embedding_endpoint_host,
                "dimensions": int(embedding_dimensions),
                "normalized": True,
            },
            "chapters": serialized_chapters,
        }
    )
    _validate_release_payload(payload)
    raw = _canonical_bytes(payload)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {
        "bundle": str(bundle_path),
        "digest": payload["digest"],
        "embedding": payload["embedding"],
        "chapters": len(serialized_chapters),
        "pages": sum(len(item["pages"]) for item in serialized_chapters),
        "concepts": sum(len(item["concepts"]) for item in serialized_chapters),
        "chunks": sum(len(item["chunks"]) for item in serialized_chapters),
        "embedded_chunks": sum(len(item["chunks"]) for item in serialized_chapters),
    }


def verify_content_release(bundle_path: Path) -> Dict[str, Any]:
    """Validate integrity and report bundle inventory without touching a DB."""
    payload = _verified_payload(bundle_path)
    chapters = payload["chapters"]
    return {
        "bundle": str(bundle_path),
        "digest": payload["digest"],
        "scope": payload["scope"],
        "embedding": payload["embedding"],
        "chapters": len(chapters),
        "pages": sum(len(item.get("pages") or []) for item in chapters),
        "concepts": sum(len(item.get("concepts") or []) for item in chapters),
        "chunks": sum(len(item.get("chunks") or []) for item in chapters),
        "embedded_chunks": sum(
            1
            for item in chapters
            for chunk in (item.get("chunks") or [])
            if chunk.get("embedding")
        ),
    }


def _matches_release(
    db: Session,
    chapter: ContentChapter,
    item: Dict[str, Any],
    *,
    bundle_digest: str,
) -> bool:
    chapter_data = item["chapter"]
    if (
        chapter.status not in LIVE_STATUSES
        or chapter.source_hash != chapter_data.get("source_hash")
        or chapter.published_source_hash != chapter_data.get("published_source_hash")
    ):
        return False
    release_meta = dict((chapter.validation_report or {}).get("content_release") or {})
    if (
        release_meta.get("bundle_digest") != bundle_digest
        or release_meta.get("chapter_digest") != item.get("digest")
    ):
        return False
    page_count = db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).count()
    concept_count = db.query(ContentConcept).filter(ContentConcept.chapter_id == chapter.id).count()
    chunk_count = db.query(ContentChunk).filter(ContentChunk.chapter_id == chapter.id).count()
    embedded_count = (
        db.query(ContentChunk)
        .filter(ContentChunk.chapter_id == chapter.id, ContentChunk.embedding.isnot(None))
        .count()
    )
    return (
        page_count == len(item["pages"])
        and concept_count == len(item["concepts"])
        and chunk_count == len(item["chunks"])
        and embedded_count == chunk_count
    )


def _existing_chapter(db: Session, chapter_data: Dict[str, Any]) -> ContentChapter | None:
    slug = str(chapter_data["slug"])
    number = int(chapter_data["chapter_number"])
    candidates = (
        db.query(ContentChapter)
        .filter(or_(ContentChapter.slug == slug, ContentChapter.chapter_number == number))
        .all()
    )
    target_identity = _chapter_identity(chapter_data)
    slug_matches = [row for row in candidates if row.slug == slug]
    if any(_chapter_identity(row) != target_identity for row in slug_matches):
        raise ContentReleaseError(f"Slug {slug} already belongs to a different curriculum chapter.")
    matches = {
        row.id: row
        for row in candidates
        if row.slug == slug or _chapter_identity(row) == target_identity
    }
    if len(matches) > 1:
        raise ContentReleaseError(
            f"Ambiguous existing rows for {slug}; resolve duplicate chapter identity before release."
        )
    return next(iter(matches.values()), None)


def _semantic_readiness(embedding_contract: Dict[str, Any]) -> Dict[str, Any]:
    from Logic import embeddings as embeddings_service

    configured = embeddings_service.embeddings_enabled()
    configured_model = embeddings_service.embedding_model() if configured else ""
    configured_host = embeddings_service.embedding_endpoint_host() if configured else ""
    stored_model = str(embedding_contract["model"])
    stored_host = str(embedding_contract["endpoint_host"])
    if not configured:
        status = "lexical_only"
    elif _normalized_key(configured_model) != _normalized_key(stored_model):
        status = "model_mismatch"
    elif configured_host.lower() != stored_host.lower():
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
        "stored_dimensions": int(embedding_contract["dimensions"]),
    }


def restore_content_release(db: Session, bundle_path: Path) -> Dict[str, Any]:
    """Idempotently restore a verified content bundle into the configured DB."""
    payload = _verified_payload(bundle_path)
    embedding_contract = dict(payload["embedding"])
    restored: list[str] = []
    skipped: list[str] = []
    try:
        for item in payload["chapters"]:
            chapter_data = dict(item.get("chapter") or {})
            slug = str(chapter_data.get("slug") or "").strip()
            chapter = _existing_chapter(db, chapter_data)
            if chapter is not None and _matches_release(
                db,
                chapter,
                item,
                bundle_digest=payload["digest"],
            ):
                skipped.append(slug)
                continue
            next_version = _replacement_version(chapter, chapter_data)
            if chapter is None:
                chapter = ContentChapter(slug=slug)
                db.add(chapter)
                db.flush()
            else:
                db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).delete(
                    synchronize_session=False
                )
                db.query(ContentConcept).filter(ContentConcept.chapter_id == chapter.id).delete(
                    synchronize_session=False
                )
                db.query(ContentChunk).filter(ContentChunk.chapter_id == chapter.id).delete(
                    synchronize_session=False
                )

            for field in _CHAPTER_FIELDS:
                setattr(chapter, field, chapter_data.get(field))
            chapter.status = "published"
            chapter.version = next_version
            chapter.pdf_path = str(chapter_data.get("pdf_path") or "")
            chapter.page_count = len(item["pages"])
            chapter.extracted_page_count = len(item["pages"])
            chapter.concept_count = len(item["concepts"])
            chapter.chunk_count = len(item["chunks"])
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            chapter.approved_by = f"content-release:{payload['digest'].split(':', 1)[-1][:12]}"
            chapter.approved_at = now
            chapter.published_at = now
            validation_report = dict(chapter_data.get("validation_report") or {})
            validation_report["content_release"] = {
                "bundle_digest": payload["digest"],
                "chapter_digest": item["digest"],
                "embedding_model": embedding_contract["model"],
                "embedding_dimensions": embedding_contract["dimensions"],
                "embedding_endpoint_host": embedding_contract["endpoint_host"],
            }
            chapter.validation_report = validation_report
            db.flush()

            db.add_all(
                [ContentPage(chapter_id=chapter.id, **{field: row.get(field) for field in _PAGE_FIELDS})
                 for row in item["pages"]]
            )
            db.add_all(
                [ContentConcept(
                    chapter_id=chapter.id,
                    **{field: row.get(field) for field in _CONCEPT_FIELDS},
                ) for row in item["concepts"]]
            )
            chunk_rows = []
            for row in item["chunks"]:
                values = {field: row.get(field) for field in _CHUNK_FIELDS}
                metadata = dict(values.get("metadata_json") or {})
                metadata["embedding_model"] = embedding_contract["model"]
                metadata["embedding_dimensions"] = embedding_contract["dimensions"]
                metadata["embedding_endpoint_host"] = embedding_contract["endpoint_host"]
                values["metadata_json"] = metadata
                chunk_rows.append(ContentChunk(chapter_id=chapter.id, **values))
            db.add_all(chunk_rows)
            restored.append(slug)

        release_job_id = f"content_release_{payload['digest'].split(':', 1)[-1][:24]}"
        release_job = (
            db.query(ContentIngestionJob)
            .filter(ContentIngestionJob.job_id == release_job_id)
            .one_or_none()
        )
        if release_job is None:
            release_job = ContentIngestionJob(
                job_id=release_job_id,
                job_type="content_release_restore",
                source_path=f"bundled://{bundle_path.name}",
            )
            db.add(release_job)
        release_job.status = "completed"
        release_job.error = ""
        prior_summary = dict(release_job.summary or {})
        restore_result = {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "restored": list(restored),
            "skipped": list(skipped),
        }
        release_job.summary = {
            "digest": payload["digest"],
            "scope": payload["scope"],
            "embedding": embedding_contract,
            "inventory": {
                "chapters": len(payload["chapters"]),
                "pages": sum(len(item["pages"]) for item in payload["chapters"]),
                "concepts": sum(len(item["concepts"]) for item in payload["chapters"]),
                "chunks": sum(len(item["chunks"]) for item in payload["chapters"]),
            },
            "initial_result": prior_summary.get("initial_result") or restore_result,
            "last_result": restore_result,
        }
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "bundle": str(bundle_path),
        "digest": payload["digest"],
        "scope": payload["scope"],
        "embedding": embedding_contract,
        "semantic_retrieval": _semantic_readiness(embedding_contract),
        "restored": restored,
        "skipped": skipped,
        "chapter_count": len(payload["chapters"]),
    }
