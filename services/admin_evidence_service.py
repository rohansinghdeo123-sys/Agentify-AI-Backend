"""Read-only, content-safe evidence contracts for the admin console.

These builders intentionally expose provenance and quality signals, not source
text, user prompts, session identifiers, raw trace metadata, or embeddings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session

from Logic.agent_router import get_agent_registry
from models import ContentChapter, ContentChunk, ContentConcept, ContentPage, ModelToolTrace
from services.content_release_readiness import content_release_readiness
from services.retrieval_readiness import retrieval_embedding_status


def _iso(value: Any) -> Optional[str]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_cutoff(hours: int) -> datetime:
    """Return a naive UTC cutoff for the project's naive-UTC DateTime columns."""

    return (_utc_now() - timedelta(hours=hours)).replace(tzinfo=None)


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pages(value: Any) -> List[int]:
    pages: set[int] = set()
    for item in _list(value):
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.add(page)
    return sorted(pages)


def _source_reference_evidence(value: Any, available_pages: set[int]) -> Dict[str, Any]:
    referenced = _pages(value)
    verified = [page for page in referenced if page in available_pages]
    missing = [page for page in referenced if page not in available_pages]
    return {
        "referenced_pages": referenced,
        "verified_pages": verified,
        "missing_pages": missing,
        "reference_count": len(referenced),
        "verified_reference_count": len(verified),
        # A proof is verified only when at least one page is referenced and
        # every reference resolves to a stored source page for this chapter.
        "verified": bool(referenced) and not missing,
    }


def _pagination(*, limit: int, offset: int, total: int) -> Dict[str, Any]:
    return {
        "limit": limit,
        "offset": offset,
        "total": total,
        "has_more": offset + limit < total,
    }


def _chapter_query(
    db: Session,
    *,
    class_level: Optional[str] = None,
    subject: Optional[str] = None,
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
):
    query = db.query(ContentChapter)
    if class_level:
        query = query.filter(func.lower(ContentChapter.class_level) == class_level.strip().lower())
    if subject:
        query = query.filter(func.lower(ContentChapter.subject) == subject.strip().lower())
    if status_filter:
        query = query.filter(func.lower(ContentChapter.status) == status_filter.strip().lower())
    if search:
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                ContentChapter.chapter_name.ilike(term),
                ContentChapter.subject.ilike(term),
                ContentChapter.book_name.ilike(term),
            )
        )
    return query


def build_content_evidence(
    db: Session,
    *,
    class_level: Optional[str] = None,
    subject: Optional[str] = None,
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    query = _chapter_query(
        db,
        class_level=class_level,
        subject=subject,
        status_filter=status_filter,
        search=search,
    )
    total = int(query.count())
    chapters = (
        query.order_by(
            ContentChapter.class_level,
            ContentChapter.subject,
            ContentChapter.chapter_number,
            ContentChapter.id,
        )
        .offset(offset)
        .limit(limit)
        .all()
    )
    chapter_ids = [row.id for row in chapters]
    concepts: Dict[int, List[Any]] = defaultdict(list)
    chunk_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"chunks": 0, "embedded": 0})
    available_pages: Dict[int, set[int]] = defaultdict(set)
    if chapter_ids:
        for row in db.query(
            ContentConcept.chapter_id,
            ContentConcept.source_pages,
            ContentConcept.validation_issues,
        ).filter(ContentConcept.chapter_id.in_(chapter_ids)).all():
            concepts[row.chapter_id].append(row)
        for chapter_id, total_chunks, embedded_chunks in (
            db.query(
                ContentChunk.chapter_id,
                func.count(ContentChunk.id),
                func.sum(case((func.json_array_length(ContentChunk.embedding) > 0, 1), else_=0)),
            )
            .filter(ContentChunk.chapter_id.in_(chapter_ids))
            .group_by(ContentChunk.chapter_id)
            .all()
        ):
            chunk_counts[chapter_id] = {
                "chunks": int(total_chunks or 0),
                "embedded": int(embedded_chunks or 0),
            }
        for chapter_id, page_number in (
            db.query(ContentPage.chapter_id, ContentPage.page_number)
            .filter(ContentPage.chapter_id.in_(chapter_ids))
            .all()
        ):
            try:
                normalized_page = int(page_number)
            except (TypeError, ValueError):
                continue
            if normalized_page > 0:
                available_pages[int(chapter_id)].add(normalized_page)

    items: List[Dict[str, Any]] = []
    for chapter in chapters:
        chapter_concepts = concepts[chapter.id]
        chapter_pages = available_pages[chapter.id]
        source_references = [
            _source_reference_evidence(row.source_pages, chapter_pages)
            for row in chapter_concepts
        ]
        cited = sum(bool(item["referenced_pages"]) for item in source_references)
        verified_cited = sum(bool(item["verified"]) for item in source_references)
        referenced_pages = sorted(
            {page for item in source_references for page in item["referenced_pages"]}
        )
        verified_pages = sorted(
            {page for item in source_references for page in item["verified_pages"]}
        )
        missing_pages = sorted(
            {page for item in source_references for page in item["missing_pages"]}
        )
        embedded = chunk_counts[chapter.id]["embedded"]
        issue_count = sum(len(_list(row.validation_issues)) for row in chapter_concepts)
        report = _dict(chapter.validation_report)
        blocking, blocking_count = _blocking_issues_from_report(report)
        concept_count = len(chapter_concepts)
        chunk_count = chunk_counts[chapter.id]["chunks"]
        items.append(
            {
                "chapter_id": chapter.id,
                "board": chapter.board,
                "class_level": chapter.class_level,
                "subject": chapter.subject,
                "chapter_number": chapter.chapter_number,
                "chapter_name": chapter.chapter_name,
                "slug": chapter.slug,
                "status": chapter.status,
                "version": chapter.version,
                "source_integrity": {
                    "source_hash": chapter.source_hash or "",
                    "published_source_hash": chapter.published_source_hash or "",
                    "published_hash_matches": bool(
                        chapter.source_hash
                        and chapter.published_source_hash
                        and chapter.source_hash == chapter.published_source_hash
                    ),
                },
                "published_at": _iso(chapter.published_at),
                "updated_at": _iso(chapter.updated_at),
                "counts": {
                    "pages": len(chapter_pages),
                    "subtopics": concept_count,
                    "chunks": chunk_count,
                    "embedded_chunks": embedded,
                },
                "quality": {
                    "coverage_score": round(float(chapter.coverage_score or 0), 4),
                    "extraction_quality": round(float(chapter.extraction_quality or 0), 4),
                    "validation_issue_count": issue_count,
                    "blocking_issues": blocking,
                    "blocking_issue_count": blocking_count,
                    "ready": bool(
                        chapter.status in {"approved", "published"}
                        and blocking_count == 0
                        and concept_count > 0
                        and verified_cited == concept_count
                    ),
                },
                "evidence": {
                    "subtopics_with_source_pages": cited,
                    "subtopics_with_verified_source_pages": verified_cited,
                    "referenced_source_pages": referenced_pages,
                    "verified_source_pages": verified_pages,
                    "missing_source_pages": missing_pages,
                    "source_page_coverage_percent": round((verified_cited / concept_count) * 100, 1)
                    if concept_count
                    else 0.0,
                    "embedding_coverage_percent": round((embedded / chunk_count) * 100, 1) if chunk_count else 0.0,
                },
            }
        )
    return {
        "items": items,
        "pagination": _pagination(limit=limit, offset=offset, total=total),
        "filters": {
            "class_level": class_level or "",
            "subject": subject or "",
            "status": status_filter or "",
            "search": search or "",
        },
    }


def _safe_issues(value: Any) -> List[Dict[str, str]]:
    safe: List[Dict[str, str]] = []
    for item in _list(value)[:20]:
        if isinstance(item, str):
            safe.append({"code": "validation", "severity": "warning", "message": item[:240]})
        elif isinstance(item, dict):
            safe.append(
                {
                    "code": str(item.get("code") or item.get("type") or "validation")[:80],
                    "severity": str(item.get("severity") or "warning")[:20],
                    "message": str(item.get("message") or item.get("detail") or "")[:240],
                }
            )
    return safe


def _safe_blocking_issues(value: Any) -> List[str]:
    issues: List[str] = []
    for item in _list(value)[:20]:
        if isinstance(item, dict):
            label = item.get("code") or item.get("type") or item.get("message")
        else:
            label = item
        if label:
            issues.append(str(label)[:160])
    return issues


def _blocking_issues_from_report(report: Dict[str, Any]) -> tuple[List[str], int]:
    """Return only release-blocking validation issues.

    Production coverage reports store every warning and error under ``issues``
    and expose the authoritative error count separately. Treating all issues as
    blockers would incorrectly mark healthy chapters as unavailable.
    """

    explicit = report.get("blocking_issues")
    if isinstance(explicit, list):
        candidates = explicit
    else:
        candidates = [
            item
            for item in _list(report.get("issues"))
            if isinstance(item, dict)
            and str(item.get("severity") or "").strip().casefold() == "error"
        ]
    issues = _safe_blocking_issues(candidates)
    try:
        recorded_count = max(0, int(report.get("blocking_issue_count") or 0))
    except (TypeError, ValueError):
        recorded_count = 0
    return issues, max(recorded_count, len(issues))


def build_chapter_evidence(
    db: Session,
    chapter_id: int,
    *,
    limit: int = 25,
    offset: int = 0,
) -> Optional[Dict[str, Any]]:
    chapter = db.query(ContentChapter).filter(ContentChapter.id == chapter_id).first()
    if chapter is None:
        return None
    available_pages = {
        int(page_number)
        for (page_number,) in db.query(ContentPage.page_number)
        .filter(ContentPage.chapter_id == chapter_id)
        .all()
        if page_number is not None and int(page_number) > 0
    }
    concept_filter = ContentConcept.chapter_id == chapter_id
    total = int(db.query(func.count(ContentConcept.id)).filter(concept_filter).scalar() or 0)
    rows = (
        db.query(
            ContentConcept.id,
            ContentConcept.concept_id,
            ContentConcept.title,
            ContentConcept.difficulty_level,
            ContentConcept.importance_level,
            ContentConcept.typical_exam_weightage,
            ContentConcept.blooms_taxonomy,
            ContentConcept.source_pages,
            ContentConcept.key_points,
            ContentConcept.examples,
            ContentConcept.formulas,
            ContentConcept.learning_objectives,
            ContentConcept.validation_issues,
            (func.length(ContentConcept.definition) > 0).label("has_definition"),
            (func.length(ContentConcept.core_explanation) > 0).label("has_explanation"),
        )
        .filter(concept_filter)
        .order_by(ContentConcept.id)
        .offset(offset)
        .limit(limit)
        .all()
    )
    chunk_count, embedded_count = db.query(
        func.count(ContentChunk.id),
        func.sum(case((func.json_array_length(ContentChunk.embedding) > 0, 1), else_=0)),
    ).filter(ContentChunk.chapter_id == chapter_id).one()
    chunk_count, embedded_count = int(chunk_count or 0), int(embedded_count or 0)
    sample_embedding = (
        db.query(ContentChunk.embedding)
        .filter(
            ContentChunk.chapter_id == chapter_id,
            func.json_array_length(ContentChunk.embedding) > 0,
        )
        .first()
    )
    embedding_dimensions = len(sample_embedding[0]) if sample_embedding and sample_embedding[0] else 0
    page_range_cap = 500
    page_rows = (
        db.query(ContentChunk.page_start, ContentChunk.page_end)
        .filter(
            ContentChunk.chapter_id == chapter_id,
            ContentChunk.page_start.isnot(None),
        )
        .distinct()
        .order_by(ContentChunk.page_start, ContentChunk.page_end)
        .limit(page_range_cap + 1)
        .all()
    )
    page_ranges_truncated = len(page_rows) > page_range_cap
    page_rows = page_rows[:page_range_cap]
    page_ranges = sorted(
        {
            (int(row.page_start), int(row.page_end or row.page_start))
            for row in page_rows
            if row.page_start is not None and int(row.page_start) > 0
        }
    )
    subtopics = []
    for row in rows:
        source_proof = _source_reference_evidence(row.source_pages, available_pages)
        issues = _safe_issues(row.validation_issues)
        recorded_issue_codes = {item["code"] for item in issues}
        if not source_proof["referenced_pages"] and "missing_source_pages" not in recorded_issue_codes:
            issues.append(
                {
                    "code": "missing_source_pages",
                    "severity": "error",
                    "message": "No source page is referenced for this subtopic.",
                }
            )
        elif source_proof["missing_pages"] and "source_page_not_ingested" not in recorded_issue_codes:
            issues.append(
                {
                    "code": "source_page_not_ingested",
                    "severity": "error",
                    "message": (
                        "Referenced source page(s) are not present in the ingested chapter: "
                        + ", ".join(str(page) for page in source_proof["missing_pages"])
                    )[:240],
                }
            )
        subtopics.append(
            {
                "id": row.id,
                "concept_id": row.concept_id,
                "title": row.title,
                "difficulty_level": row.difficulty_level,
                "importance_level": row.importance_level,
                "exam_weightage": row.typical_exam_weightage,
                "blooms_taxonomy": row.blooms_taxonomy,
                "source_proof": {
                    # ``page_numbers`` remains for older clients and means
                    # recorded references, not automatically verified proof.
                    "page_numbers": source_proof["referenced_pages"],
                    "referenced_page_numbers": source_proof["referenced_pages"],
                    "verified_page_numbers": source_proof["verified_pages"],
                    "missing_page_numbers": source_proof["missing_pages"],
                    "reference_count": source_proof["reference_count"],
                    "verified_reference_count": source_proof["verified_reference_count"],
                    "citation_count": source_proof["verified_reference_count"],
                    "verified": source_proof["verified"],
                },
                "content_checks": {
                    "has_definition": bool(row.has_definition),
                    "has_explanation": bool(row.has_explanation),
                    "key_point_count": len(_list(row.key_points)),
                    "example_count": len(_list(row.examples)),
                    "formula_count": len(_list(row.formulas)),
                    "learning_objective_count": len(_list(row.learning_objectives)),
                },
                "validation": {"passed": not issues, "issues": issues},
            }
        )
    report = _dict(chapter.validation_report)
    blocking_issues, blocking_issue_count = _blocking_issues_from_report(report)
    return {
        "chapter": {
            "chapter_id": chapter.id,
            "board": chapter.board,
            "class_level": chapter.class_level,
            "subject": chapter.subject,
            "chapter_number": chapter.chapter_number,
            "chapter_name": chapter.chapter_name,
            "status": chapter.status,
            "version": chapter.version,
            "source_integrity": {
                "source_hash": chapter.source_hash or "",
                "published_source_hash": chapter.published_source_hash or "",
                "published_hash_matches": bool(
                    chapter.source_hash
                    and chapter.published_source_hash
                    and chapter.source_hash == chapter.published_source_hash
                ),
            },
            "coverage_score": round(float(chapter.coverage_score or 0), 4),
            "extraction_quality": round(float(chapter.extraction_quality or 0), 4),
            "blocking_issues": blocking_issues,
            "blocking_issue_count": blocking_issue_count,
        },
        "retrieval_evidence": {
            "chunk_count": chunk_count,
            "embedded_chunk_count": embedded_count,
            "embedding_coverage_percent": round((embedded_count / chunk_count) * 100, 1) if chunk_count else 0.0,
            "stored_embedding_dimensions": [embedding_dimensions] if embedding_dimensions else [],
            "source_page_ranges": [
                {"page_start": start, "page_end": end} for start, end in page_ranges
            ],
            "source_page_ranges_truncated": page_ranges_truncated,
        },
        "subtopics": subtopics,
        "pagination": _pagination(limit=limit, offset=offset, total=total),
    }


def _grounding(metadata: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = _dict(metadata.get("retrieval"))
    gate = _dict(retrieval.get("gate"))
    status = str(gate.get("grounding_status") or "not_recorded")
    policy = str(retrieval.get("policy") or gate.get("policy") or "none")
    paragraphs = int(retrieval.get("paragraphs_found") or gate.get("paragraphs_found") or 0)
    source_pages = _pages(retrieval.get("source_pages"))
    return {
        "policy": policy,
        "status": status,
        "source": str(retrieval.get("source") or gate.get("source") or ""),
        "section_id": str(retrieval.get("section_id") or gate.get("section_id") or ""),
        "paragraphs_found": paragraphs,
        "source_pages": source_pages,
        "citation_count": len(source_pages),
        "supported": bool(retrieval.get("supported", gate.get("material_supported", False))),
    }


def _agent_name(value: Any) -> str:
    name = str(value or "unknown")
    return name[:-5] if name.endswith("_turn") else name


def build_activity_evidence(
    db: Session,
    *,
    agent: Optional[str] = None,
    status_filter: Optional[str] = None,
    grounding_status: Optional[str] = None,
    hours: int = 24,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    cutoff = _utc_cutoff(hours)
    query = db.query(ModelToolTrace).filter(
        ModelToolTrace.trace_type == "turn",
        ModelToolTrace.created_at >= cutoff,
    )
    if agent:
        query = query.filter(ModelToolTrace.name == f"{agent.strip()}_turn")
    if status_filter:
        query = query.filter(func.lower(ModelToolTrace.status) == status_filter.strip().lower())
    ordered = query.order_by(ModelToolTrace.created_at.desc(), ModelToolTrace.id.desc())
    # Grounding lives in JSON with provider-dependent SQL semantics, so only
    # that optional filter needs a bounded in-memory sample. Normal activity
    # pagination is executed by the database and loads just the requested page.
    if grounding_status:
        rows = ordered.limit(2000).all()
        wanted = grounding_status.strip().lower()
        rows = [row for row in rows if _grounding(_dict(row.metadata_json))["status"].lower() == wanted]
        total = len(rows)
        selected = rows[offset : offset + limit]
        sample_cap = 2000
    else:
        total = int(query.count())
        selected = ordered.offset(offset).limit(limit).all()
        sample_cap = 0
    items = []
    for row in selected:
        metadata = _dict(row.metadata_json)
        quality = _dict(metadata.get("quality"))
        score = _float(quality.get("score"))
        issues = [str(item)[:160] for item in _list(quality.get("issues"))[:10]]
        items.append(
            {
                "trace_id": row.id,
                "created_at": _iso(row.created_at),
                "agent": _agent_name(row.name),
                "status": row.status or "unknown",
                "latency_ms": int(row.latency_ms or 0),
                "estimated_tokens": int(row.estimated_input_tokens or 0) + int(row.estimated_output_tokens or 0),
                "quality": {
                    "score": round(score, 4) if score is not None else None,
                    "passed": bool(quality.get("passed")) if "passed" in quality else None,
                    "grounding_score": _float(quality.get("grounding")),
                    "hallucination_risk": _float(quality.get("hallucination_risk")),
                    "issues": issues,
                },
                "grounding": _grounding(metadata),
                "fallback_count": len(_list(metadata.get("fallbacks"))),
            }
        )
    return {
        "items": items,
        "pagination": _pagination(limit=limit, offset=offset, total=total),
        "window_hours": hours,
        "sample_cap": sample_cap,
        "filters": {
            "agent": agent or "",
            "status": status_filter or "",
            "grounding_status": grounding_status or "",
        },
    }


def build_evidence_overview(db: Session, *, hours: int = 24) -> Dict[str, Any]:
    chapter_total = int(db.query(func.count(ContentChapter.id)).scalar() or 0)
    # The overview is an inventory total, not a list page. Load the complete
    # chapter set so future subjects cannot silently disappear after row 100.
    content = build_content_evidence(db, limit=max(chapter_total, 1), offset=0)
    chapters = content["items"]
    counts = Counter()
    subjects = set()
    for row in chapters:
        subjects.add((row["class_level"], row["subject"]))
        counts.update(row["counts"])
    activity = build_activity_evidence(db, hours=hours, limit=2000, offset=0)
    turns = activity["items"]
    scores = [item["quality"]["score"] for item in turns if item["quality"]["score"] is not None]
    quality_recorded = [item for item in turns if item["quality"]["passed"] is not None]
    retrieval_turns = [item for item in turns if item["grounding"]["policy"] != "none"]
    grounded = [item for item in retrieval_turns if item["grounding"]["status"] == "grounded"]
    source_proof = [item for item in retrieval_turns if item["grounding"]["citation_count"] > 0]
    latest_by_agent = {
        _agent_name(name): _iso(last_activity)
        for name, last_activity in db.query(
            ModelToolTrace.name,
            func.max(ModelToolTrace.created_at),
        )
        .filter(ModelToolTrace.trace_type == "turn")
        .group_by(ModelToolTrace.name)
        .all()
    }
    agents: Dict[str, Dict[str, Any]] = {}
    for registry_row in get_agent_registry():
        agent_id = str(registry_row.get("agent_id") or "unknown")
        agents[agent_id] = {
            "agent": agent_id,
            "display_name": str(registry_row.get("display_name") or agent_id),
            "role": str(registry_row.get("role") or ""),
            "registered": True,
            "runs": 0,
            "errors": 0,
            "last_activity": latest_by_agent.get(agent_id),
            "_latencies": [],
            "_quality_scores": [],
        }
    for item in turns:
        bucket = agents.setdefault(
            item["agent"],
            {
                "agent": item["agent"],
                "display_name": item["agent"],
                "role": "",
                "registered": False,
                "runs": 0,
                "errors": 0,
                "last_activity": latest_by_agent.get(item["agent"]) or item["created_at"],
                "_latencies": [],
                "_quality_scores": [],
            },
        )
        bucket["runs"] += 1
        bucket["_latencies"].append(item["latency_ms"])
        if item["quality"]["score"] is not None:
            bucket["_quality_scores"].append(item["quality"]["score"])
        if item["status"] not in {"success", "skipped"}:
            bucket["errors"] += 1
    agent_rows = []
    for bucket in agents.values():
        latencies = bucket.pop("_latencies")
        agent_scores = bucket.pop("_quality_scores")
        successes = bucket["runs"] - bucket["errors"]
        if bucket["runs"]:
            activity_state = "active"
            health = "healthy" if bucket["errors"] == 0 else "attention"
            success_rate = round(successes / bucket["runs"] * 100, 1)
        elif bucket["last_activity"]:
            activity_state = "stale"
            health = "attention"
            success_rate = None
        else:
            activity_state = "not_observed"
            health = "attention"
            success_rate = None
        bucket.update(
            {
                "health": health,
                "activity_state": activity_state,
                "success_rate_percent": success_rate,
                "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "average_quality_score": round(sum(agent_scores) / len(agent_scores), 4)
                if agent_scores
                else None,
            }
        )
        agent_rows.append(bucket)
    return {
        "generated_at": _iso(_utc_now()),
        "window_hours": hours,
        "agents": sorted(agent_rows, key=lambda item: item["agent"]),
        "content": {
            "subject_catalogs": len(subjects),
            "chapters": len(chapters),
            "published_chapters": sum(row["status"] == "published" for row in chapters),
            "pages": counts["pages"],
            "subtopics": counts["subtopics"],
            "chunks": counts["chunks"],
            "embedded_chunks": counts["embedded_chunks"],
            "chapters_ready": sum(row["quality"]["ready"] for row in chapters),
            "source_page_coverage_percent": round(
                sum(row["evidence"]["subtopics_with_verified_source_pages"] for row in chapters)
                / counts["subtopics"]
                * 100,
                1,
            ) if counts["subtopics"] else 0.0,
            "embedding_coverage_percent": round(counts["embedded_chunks"] / counts["chunks"] * 100, 1)
            if counts["chunks"] else 0.0,
        },
        "quality": {
            "turns": len(turns),
            "successful_turns": sum(item["status"] in {"success", "skipped"} for item in turns),
            "average_score": round(sum(scores) / len(scores), 4) if scores else None,
            "quality_pass_rate_percent": round(
                sum(item["quality"]["passed"] is True for item in quality_recorded) / len(quality_recorded) * 100,
                1,
            ) if quality_recorded else None,
            "retrieval_turns": len(retrieval_turns),
            "grounded_turns": len(grounded),
            "grounded_rate_percent": round(len(grounded) / len(retrieval_turns) * 100, 1)
            if retrieval_turns else None,
            "source_page_proof_rate_percent": round(len(source_proof) / len(retrieval_turns) * 100, 1)
            if retrieval_turns else None,
        },
        "readiness": {
            "release": content_release_readiness(db),
            "semantic_retrieval": retrieval_embedding_status(db),
        },
    }
