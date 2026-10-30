"""Exam paper upload, pattern intelligence, and probable-question endpoints.

All routes require authentication and are strictly scoped to the caller: a paper,
analysis, or question set that belongs to another user is reported as 404 so its
existence is never leaked.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app import config
from app.exam_schemas import (
    PaperAnalysisResponse,
    PaperListResponse,
    PaperQuestionsResponse,
    PaperReanalyzeRequest,
    PaperUploadResponse,
    PatternAnalysisOut,
    PatternAnalyzeRequest,
    PatternGroupedResponse,
    PatternSummaryResponse,
    ProbableGenerateRequest,
    ProbableListResponse,
    ProbableQuestionSetOut,
)
from app.security import (
    enforce_user_quota,
    require_authenticated_user_id,
    verify_firebase_user,
)
from database import get_db
from services import exam_paper_service as paper_service
from services import exam_pattern_service as pattern_service

logger = logging.getLogger("ai_educator.routers.exam_papers")

router = APIRouter(tags=["exam-intelligence"])

PAPER_NOT_FOUND = "Paper not found."


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload in chunks, rejecting it once it exceeds the configured cap
    so an oversize file is never fully buffered."""
    cap = config.EXAM_UPLOAD_MAX_BYTES
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            max_mb = cap / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_mb:.0f} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# =========================================================
# PAPER UPLOAD & ANALYSIS
# =========================================================
@router.post("/exam/papers/upload", response_model=PaperUploadResponse)
async def upload_paper(
    file: UploadFile = File(...),
    class_level: str = Form(""),
    subject: str = Form(""),
    chapter_name: str = Form(""),
    chapter_id: Optional[int] = Form(None),
    exam_type: str = Form(""),
    paper_title: str = Form(""),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    await run_in_threadpool(enforce_user_quota, user_id, "exam_paper")

    data = await _read_capped(file)
    # PDF extraction, analysis and DB persistence are synchronous. Keep them
    # together on a worker so a long upload cannot stall every learner's API.
    def process_upload():
        try:
            paper = paper_service.create_paper_from_upload(
                db,
                user_id,
                filename=file.filename or "",
                content_type=file.content_type or "",
                data=data,
                class_level=class_level,
                subject=subject,
                chapter_name=chapter_name,
                chapter_id=chapter_id,
                exam_type=exam_type,
                paper_title=paper_title,
            )
        except paper_service.PaperValidationError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        message = {
            "analyzed": "Paper uploaded and analyzed.",
            "analyzed_empty": "Paper uploaded, but no structured questions could be extracted.",
            "needs_ocr": "Paper uploaded. It looks like a scanned image; OCR is required to read it.",
            "failed": "Paper uploaded, but the text could not be read.",
        }.get(paper.parse_status, "Paper uploaded.")

        return {
            "paper": paper_service.serialize_paper(paper),
            "analysis": paper.analysis_json or {},
            "questions_extracted": paper.extracted_question_count or 0,
            "warnings": list(paper.warnings_json or []),
            "message": message,
        }

    return await run_in_threadpool(process_upload)


@router.get("/exam/papers", response_model=PaperListResponse)
def list_papers(
    subject: Optional[str] = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    total, rows = paper_service.list_papers(db, user_id, subject=subject, limit=limit, offset=offset)
    return {"total": total, "papers": [paper_service.serialize_paper(row) for row in rows]}


def _require_owned_paper(db: Session, user_id: str, paper_id: int):
    paper = paper_service.get_owned_paper(db, user_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PAPER_NOT_FOUND)
    return paper


@router.get("/exam/papers/{paper_id}")
def get_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    paper = _require_owned_paper(db, user_id, paper_id)
    return {
        "paper": paper_service.serialize_paper(paper),
        "analysis": paper.analysis_json or {},
    }


@router.get("/exam/papers/{paper_id}/questions", response_model=PaperQuestionsResponse)
def get_paper_questions(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    paper = _require_owned_paper(db, user_id, paper_id)
    questions = paper_service.list_questions(db, paper)
    return {
        "paper_id": paper.id,
        "count": len(questions),
        "questions": [paper_service.serialize_question(q) for q in questions],
    }


@router.get("/exam/papers/{paper_id}/analysis", response_model=PaperAnalysisResponse)
def get_paper_analysis(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    paper = _require_owned_paper(db, user_id, paper_id)
    return {
        "paper_id": paper.id,
        "parse_status": paper.parse_status or "",
        "extraction_confidence": paper.extraction_confidence or 0.0,
        "analysis": paper.analysis_json or {},
        "warnings": list(paper.warnings_json or []),
    }


@router.post("/exam/papers/{paper_id}/reanalyze")
def reanalyze_paper(
    paper_id: int,
    payload: PaperReanalyzeRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    enforce_user_quota(user_id, "exam_paper")
    paper = _require_owned_paper(db, user_id, paper_id)
    paper_service.reanalyze_paper(
        db,
        paper,
        class_level=payload.class_level,
        subject=payload.subject,
        chapter_name=payload.chapter_name,
        exam_type=payload.exam_type,
    )
    return {
        "paper": paper_service.serialize_paper(paper),
        "analysis": paper.analysis_json or {},
        "questions_extracted": paper.extracted_question_count or 0,
        "warnings": list(paper.warnings_json or []),
        "message": "Paper re-analyzed.",
    }


@router.delete("/exam/papers/{paper_id}")
def delete_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    paper = _require_owned_paper(db, user_id, paper_id)
    paper_service.delete_paper(db, paper)
    return {"status": "deleted", "id": paper_id}


# =========================================================
# PATTERN INTELLIGENCE
# =========================================================
@router.post("/exam/pattern/analyze", response_model=PatternAnalysisOut)
def analyze_pattern(
    payload: PatternAnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    enforce_user_quota(user_id, "exam_paper")
    try:
        row = pattern_service.run_pattern_analysis(
            db,
            user_id,
            paper_ids=payload.paper_ids,
            class_level=payload.class_level,
            subject=payload.subject,
            chapter_name=payload.chapter_name,
        )
    except pattern_service.PatternError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return pattern_service.serialize_pattern(row)


@router.get("/exam/pattern/summary", response_model=PatternSummaryResponse)
def pattern_summary(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    return pattern_service.pattern_summary(db, user_id)


@router.get("/exam/pattern/by-subject", response_model=PatternGroupedResponse)
def pattern_by_subject(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    return pattern_service.pattern_grouped(db, user_id, group_by="subject")


@router.get("/exam/pattern/by-chapter", response_model=PatternGroupedResponse)
def pattern_by_chapter(
    subject: Optional[str] = Query(default=None, max_length=120),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    return pattern_service.pattern_grouped(db, user_id, group_by="chapter", subject=subject)


# =========================================================
# PROBABLE QUESTIONS
# =========================================================
@router.post("/exam/probable-questions/generate", response_model=ProbableQuestionSetOut)
def generate_probable_questions(
    payload: ProbableGenerateRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    enforce_user_quota(user_id, "exam_paper")
    try:
        row = pattern_service.generate_probable(
            db,
            user_id,
            analysis_id=payload.analysis_id,
            paper_ids=payload.paper_ids,
            class_level=payload.class_level,
            subject=payload.subject,
            chapter_name=payload.chapter_name,
            generation_mode=payload.generation_mode,
            count=payload.count,
            use_syllabus_grounding=payload.use_syllabus_grounding,
        )
    except pattern_service.PatternError as exc:
        # A missing analysis_id is a not-found; an empty corpus is a bad request.
        message = str(exc)
        code = status.HTTP_404_NOT_FOUND if "not found" in message.lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=message) from exc
    return pattern_service.serialize_probable(row)


@router.get("/exam/probable-questions", response_model=ProbableListResponse)
def list_probable_questions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    total, rows = pattern_service.list_probable(db, user_id, limit=limit, offset=offset)
    return {"total": total, "sets": [pattern_service.serialize_probable(row) for row in rows]}


@router.get("/exam/probable-questions/{set_id}", response_model=ProbableQuestionSetOut)
def get_probable_question_set(
    set_id: int,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_firebase_user),
):
    user_id = require_authenticated_user_id(current_user)
    row = pattern_service.get_owned_probable(db, user_id, set_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Probable question set not found.")
    return pattern_service.serialize_probable(row)
