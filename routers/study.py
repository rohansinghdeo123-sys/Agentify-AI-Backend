"""Study endpoints: section AI, MCQ/probable generation, and artifacts."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.request_models import (
    ArtifactGenerateRequest,
    GenerateMCQRequest,
    GenerateProbableRequest,
    PlanningLearningEventRequest,
    SectionAIRequest,
)
from app.security import (
    enforce_user_quota,
    require_authenticated_user_id,
    require_owned_study_session,
    session_id_belongs_to_user,
    verify_firebase_user,
)
from app.serializers import normalize_topic
from database import get_db
from Logic.section_doubt import (
    generate_structured_mcqs,
    generate_structured_probable_questions,
    section_doubt,
)
from Logic.tools.artifact_generator import (
    ARTIFACT_DATA_NOT_AVAILABLE,
    available_artifact_sections,
    generate_study_artifacts,
)
from services.catalog_service import build_catalog, resolve_catalog_topic
from services.profile_service import profile_learning_context
from services.planning_progress_service import (
    confirm_study_answer_event,
    record_study_answer_event,
)
from services.ttl_cache import TTLCache

from models import ContentChapter, ContentConcept
from Logic.content_pipeline import APPROVED_STATUSES

router = APIRouter(tags=["study"])

_catalog_cache = TTLCache(max_entries=2)


def _latest_planning_answer_receipt(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    answer: str,
) -> Dict[str, Any] | None:
    from models import AgentChatMemory

    if not session_id_belongs_to_user(session_id, user_id):
        return None
    row = (
        db.query(AgentChatMemory)
        .filter(
            AgentChatMemory.session_id == session_id,
            AgentChatMemory.role == "assistant",
        )
        .order_by(AgentChatMemory.id.desc())
        .first()
    )
    metadata = row.metadata_json if row and isinstance(row.metadata_json, dict) else {}
    if (
        row is None
        or str(row.content or "").strip() != answer.strip()
        or metadata.get("event_type") != "study_answer"
    ):
        return None
    return record_study_answer_event(
        db,
        user_id=user_id,
        interaction_id=f"study_answer:{row.id}",
        scope={
            "catalog_source": metadata.get("catalog_source"),
            "chapter_slug": metadata.get("chapter_slug"),
            "planning_unit_id": metadata.get("unit_id"),
            "section_id": metadata.get("primary_topic_id"),
            "subject": metadata.get("subject"),
            "class_level": metadata.get("class_level"),
        },
        source_session_id=session_id,
    )


@router.get("/catalog")
def learning_catalog(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    return _catalog_cache.get_or_build("catalog", 300.0, lambda: build_catalog(db))


@router.post("/section-ai")
def section_ai(
    request: SectionAIRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_owned_study_session(request.session_id, current_user)
    enforce_user_quota(user_id, "coach")
    learner_profile = profile_learning_context(db, user_id)
    learner_class = learner_profile.get("class_level", "")
    requested_class = request.class_level or learner_class
    resolved_topic = resolve_catalog_topic(
        db,
        request.section_id,
        subject=request.subject,
        chapter=request.chapter,
        topic=request.topic,
        class_level=requested_class,
        catalog_source=request.catalog_source,
    )
    section_id = (
        resolved_topic["section_id"]
        if resolved_topic
        else normalize_topic(request.section_id)
    )
    content_scope = {
        "section_id": section_id,
        "subject": request.subject or "",
        "chapter": request.chapter or "",
        "topic": request.topic or request.section_id,
        "class_level": requested_class,
        "catalog_source": request.catalog_source or "",
    }
    if resolved_topic:
        content_scope.update(resolved_topic)
    effective_class_level = str(content_scope.get("class_level") or requested_class or "")
    answer = section_doubt(
        question=request.question,
        section_id=section_id,
        session_id=request.session_id,
        mode=request.mode,
        difficulty=request.difficulty,
        strict_grounding=request.strict_grounding or request.retrieval_required,
        required_not_found_response=request.required_not_found_response,
        class_level=effective_class_level,
        content_scope=content_scope,
    )
    planning_learning = _latest_planning_answer_receipt(
        db,
        user_id=user_id,
        session_id=request.session_id,
        answer=answer,
    )
    response: Dict[str, Any] = {"answer": answer}
    if planning_learning:
        response.update(
            {
                "interaction_id": planning_learning["interaction_id"],
                "planning_learning": planning_learning,
            }
        )
    return response


@router.post("/planning/learning-events")
def confirm_planning_learning_event(
    request: PlanningLearningEventRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    result = confirm_study_answer_event(
        db,
        user_id=user_id,
        interaction_id=request.interaction_id,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="This interaction is not a verified Planning-scoped Study answer.",
        )
    return result


@router.post("/generate-mcqs")
def generate_mcqs(
    request: GenerateMCQRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_owned_study_session(request.session_id, current_user)
    enforce_user_quota(user_id, "exam")
    learner_profile = profile_learning_context(db, user_id)
    learner_class = learner_profile.get("class_level", "")
    requested_class = request.class_level or learner_class
    requested_section = normalize_topic(request.section_id or request.topic)
    resolved_topic = (
        resolve_catalog_topic(
            db,
            requested_section,
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            class_level=requested_class,
            catalog_source=request.catalog_source,
        )
        if request.subject or request.chapter or requested_section.startswith("unit_")
        else None
    )
    section_id = (
        resolved_topic["section_id"]
        if resolved_topic
        else requested_section
    )
    content_scope = {
        "section_id": section_id,
        "subject": request.subject or "",
        "chapter": request.chapter or "",
        "topic": request.topic,
        "class_level": requested_class,
        "catalog_source": request.catalog_source or "",
    }
    if resolved_topic:
        content_scope.update(resolved_topic)

    return generate_structured_mcqs(
        topic=str(content_scope.get("topic") or request.topic),
        section_id=section_id,
        session_id=request.session_id,
        difficulty=request.difficulty,
        count=request.count,
        strict_grounding=request.strict_grounding or request.retrieval_required,
        required_not_found_response=request.required_not_found_response,
        include_source=request.include_source,
        class_level=str(content_scope.get("class_level") or requested_class),
        content_scope=content_scope,
    )


@router.post("/generate-probable-questions")
def generate_probable_questions(
    request: GenerateProbableRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_owned_study_session(request.session_id, current_user)
    enforce_user_quota(user_id, "exam")
    learner_profile = profile_learning_context(db, user_id)
    learner_class = learner_profile.get("class_level", "")
    requested_class = request.class_level or learner_class
    requested_section = normalize_topic(request.section_id or request.topic)
    resolved_topic = (
        resolve_catalog_topic(
            db,
            requested_section,
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            class_level=requested_class,
            catalog_source=request.catalog_source,
        )
        if request.subject or request.chapter or requested_section.startswith("unit_")
        else None
    )
    section_id = (
        resolved_topic["section_id"]
        if resolved_topic
        else requested_section
    )
    content_scope = {
        "section_id": section_id,
        "subject": request.subject or "",
        "chapter": request.chapter or "",
        "topic": request.topic,
        "class_level": requested_class,
        "catalog_source": request.catalog_source or "",
    }
    if resolved_topic:
        content_scope.update(resolved_topic)

    return generate_structured_probable_questions(
        topic=str(content_scope.get("topic") or request.topic),
        section_id=section_id,
        session_id=request.session_id,
        difficulty=request.difficulty,
        strict_grounding=request.strict_grounding or request.retrieval_required,
        required_not_found_response=request.required_not_found_response,
        include_source=request.include_source,
        class_level=str(content_scope.get("class_level") or requested_class),
        content_scope=content_scope,
    )


@router.post("/artifacts/generate")
def generate_artifacts(
    request: ArtifactGenerateRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    enforce_user_quota(user_id, "artifact")
    learner_profile = profile_learning_context(db, user_id)
    requested_section_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        (request.section_id or request.topic or "").strip().lower(),
    ).strip("_")
    learner_class = learner_profile.get("class_level", "")
    requested_class = request.class_level or learner_class
    resolved_topic = resolve_catalog_topic(
        db,
        requested_section_id,
        subject=request.subject,
        chapter=request.chapter,
        topic=request.topic,
        class_level=requested_class,
        catalog_source=request.catalog_source,
    )
    section_id = str((resolved_topic or {}).get("section_id") or requested_section_id)
    content_scope = {
        "section_id": section_id,
        "subject": request.subject or "",
        "chapter": request.chapter or "",
        "topic": request.topic or request.section_id,
        "class_level": requested_class,
        "catalog_source": request.catalog_source or "",
    }
    if resolved_topic:
        content_scope.update(resolved_topic)
    try:
        result = generate_study_artifacts(
            section_id=section_id,
            topic=str(content_scope.get("topic") or request.topic or ""),
            subject=str(content_scope.get("subject") or request.subject or ""),
            chapter=str(content_scope.get("chapter") or request.chapter or ""),
            content_scope=content_scope,
        )
        if isinstance(result, dict):
            result["class_level"] = str(content_scope.get("class_level") or requested_class)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ARTIFACT_DATA_NOT_AVAILABLE,
        ) from exc


@router.get("/artifacts/catalog")
def artifact_catalog():
    return {
        "subject": "Chemistry",
        "available_sections": available_artifact_sections(),
        "message": "Artifacts are generated only from ingested platform study data.",
    }


# =========================================================
# NEW: CHAPTER NOTES VIEWER API
# =========================================================
@router.get("/notes/{chapter_slug}")
def get_chapter_notes(
    chapter_slug: str,
    topic_id: Optional[str] = Query(None, description="Filter notes by specific topic"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    """Fetch the published master notes for a chapter. If topic_id is provided, filter to that topic."""
    
    from services.topic_notes_service import mapped_notes
    mapped = mapped_notes(chapter_slug, topic_id, db, str(current_user.get("uid") or ""))
    if mapped is not None:
        return mapped

    chapter = (
        db.query(ContentChapter)
        .filter(
            ContentChapter.slug == chapter_slug,
            ContentChapter.status.in_(APPROVED_STATUSES)
        )
        .first()
    )
    
    if not chapter:
        chapter = (
            db.query(ContentChapter)
            .filter(
                ContentChapter.slug.endswith(chapter_slug),
                ContentChapter.status.in_(APPROVED_STATUSES)
            )
            .first()
        )

    if not chapter:
        chapter_name_search = chapter_slug.replace("_", " ").title()
        chapter = (
            db.query(ContentChapter)
            .filter(
                ContentChapter.chapter_name.ilike(f"%{chapter_name_search}%"),
                ContentChapter.status.in_(APPROVED_STATUSES)
            )
            .first()
        )

    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter notes not found or not yet published."
        )

    query = db.query(ContentConcept).filter(ContentConcept.chapter_id == chapter.id)
    
    # STRICT FILTER: Just match the concept_id directly. No complex resolution.
    if topic_id:
        query = query.filter(ContentConcept.concept_id == topic_id)

    concepts = query.order_by(ContentConcept.id).all()

    notes = []
    for concept in concepts:
        notes.append({
            "id": concept.concept_id,
            "title": concept.title,
            "definition": concept.definition,
            "explanation": concept.core_explanation,
            "key_points": concept.key_points or [],
            "formulas": concept.formulas or [],
            "examples": concept.examples or [],
            "common_mistakes": concept.common_mistakes or [],
        })

    return {
        "chapter": {
            "slug": chapter.slug,
            "name": chapter.chapter_name,
            "class_level": chapter.class_level,
            "subject": chapter.subject,
            "chapter_number": chapter.chapter_number,
        },
        "concepts": notes
    }