"""Protected indexing of uploaded master notes using existing curriculum IDs."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError
from sqlalchemy.orm import Session
from sqlalchemy.orm import defer
from starlette.concurrency import run_in_threadpool

from app.security import require_admin
from database import get_db
from models import ContentChapter, ContentChunk
from services.admin_service import record_admin_audit
from services.notes_documents_service import (
    MASTER_NOTES, MAX_PDF_BYTES, register_master_document, serialize_master_document,
)

router = APIRouter(prefix='/admin/notes',tags=['admin-notes'])


class MasterSection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1,max_length=160)
    title: str = Field(min_length=1,max_length=240)
    unit_id: str = Field(min_length=1,max_length=240)
    subtopic_ids: list[str] = Field(min_length=1,max_length=100)
    text: Optional[str] = Field(default=None,max_length=50000)
    source_pages: list[StrictInt] = Field(default_factory=list,max_length=1000)
    start_anchor: Optional[str] = Field(default=None,max_length=400)
    end_anchor: Optional[str] = Field(default=None,max_length=400)
    teaching_format: str = 'concept'
    reviewed: StrictBool = False
    complete: StrictBool = False
    review_required: StrictBool = False
    review_notes: str = Field(default='',max_length=2000)


class MasterDocumentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1,max_length=240)
    document_key: str = Field(min_length=1,max_length=160)
    class_level: str = Field(min_length=1,max_length=64)
    subject: str = Field(min_length=1,max_length=120)
    chapter_ref: str = Field(min_length=1,max_length=240)
    sections: list[MasterSection] = Field(min_length=1,max_length=2000)
    embed: bool = False


def _curriculum(db,payload):
    from services.notes_scope_service import resolve_notes_curriculum
    curriculum = resolve_notes_curriculum(db,payload.chapter_ref,subject=payload.subject,class_level=payload.class_level)
    if not curriculum:
        raise ValueError('Select an existing, unambiguous curriculum chapter in the exact class and subject')
    return curriculum


def _save(db,payload,actor,request,pdf_bytes=None):
    try:
        curriculum = _curriculum(db,payload)
        document = register_master_document(db,curriculum=curriculum,title=payload.title,
            document_key=payload.document_key,sections=[s.model_dump() for s in payload.sections],
            actor_uid=actor['uid'],pdf_bytes=pdf_bytes,embed=payload.embed)
        record_admin_audit(db,current_admin=actor,action='master_notes_index',target_type='master_notes_document',
                           target_id=str(document.id),request=request,
                           metadata={'curriculum_key':curriculum['curriculum_key'],'source_sha256':document.source_hash,'sections':document.chunk_count})
        return {'document':serialize_master_document(document),'status':'indexed'}
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400,detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=503,detail='Master notes could not be indexed; verify extraction/embedding configuration.') from exc


@router.post('/master-documents')
def register_structured_master(payload:MasterDocumentRequest,request:Request,db:Session=Depends(get_db),actor:dict=Depends(require_admin)):
    return _save(db,payload,actor,request)


@router.post('/master-documents/upload')
async def upload_master_pdf(request:Request,file:UploadFile=File(...),title:str=Form(...,max_length=240),
    document_key:str=Form(...,max_length=160),class_level:str=Form(...,max_length=64),
    subject:str=Form(...,max_length=120),chapter_ref:str=Form(...,max_length=240),
    sections_json:str=Form(...,max_length=2_000_000),embed:bool=Form(False),
    db:Session=Depends(get_db),actor:dict=Depends(require_admin)):
    if file.content_type not in {'application/pdf','application/octet-stream',None}:
        raise HTTPException(status_code=400,detail='Upload a PDF document')
    data, total = [], 0
    while block := await file.read(1024*1024):
        total += len(block)
        if total > MAX_PDF_BYTES:
            raise HTTPException(status_code=413,detail='Master PDFs must not exceed 25 MB')
        data.append(block)
    try:
        payload = MasterDocumentRequest(title=title,document_key=document_key,class_level=class_level,
            subject=subject,chapter_ref=chapter_ref,sections=json.loads(sections_json),embed=embed)
    except (ValueError,ValidationError) as exc:
        raise HTTPException(status_code=400,detail='Provide valid explicit section mappings for this document') from exc
    return await run_in_threadpool(_save,db,payload,actor,request,b''.join(data))


@router.get('/master-documents')
def list_master_documents(chapter_ref:str=Query(...,min_length=1,max_length=240),subject:str=Query(...,min_length=1,max_length=120),
    class_level:str=Query(...,min_length=1,max_length=64),db:Session=Depends(get_db),actor:dict=Depends(require_admin)):
    from services.notes_scope_service import resolve_notes_curriculum
    curriculum = resolve_notes_curriculum(db,chapter_ref,subject=subject,class_level=class_level)
    if not curriculum:
        raise HTTPException(status_code=404,detail='Curriculum chapter not found')
    rows = db.query(ContentChapter).filter(ContentChapter.status=='indexed_master',
        ContentChapter.validation_report['source_type'].as_string()==MASTER_NOTES,
        ContentChapter.validation_report['target_curriculum_key'].as_string()==curriculum['curriculum_key']).order_by(ContentChapter.id).all()
    return {'documents':[serialize_master_document(row) for row in rows]}


@router.get('/master-documents/{document_id}')
def read_master_document(document_id:int,db:Session=Depends(get_db),actor:dict=Depends(require_admin)):
    document = db.query(ContentChapter).filter_by(id=document_id,status='indexed_master').one_or_none()
    if not document or (document.validation_report or {}).get('source_type') != MASTER_NOTES:
        raise HTTPException(status_code=404,detail='Master document not found')
    data = serialize_master_document(document)
    chunks = {row.chunk_id:row for row in db.query(ContentChunk).options(defer(ContentChunk.embedding)).filter_by(chapter_id=document.id).all()}
    data['sections'] = [{**section,'text':chunks[section['chunk_id']].text if section['chunk_id'] in chunks else ''} for section in data['sections']]
    return {'document':data}


class TopicDurationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chapter_ref: str = Field(min_length=1,max_length=240)
    class_level: str = Field(min_length=1,max_length=64)
    subject: str = Field(min_length=1,max_length=120)
    unit_id: str = Field(min_length=1,max_length=240)
    minimum_minutes: StrictInt = Field(ge=5,le=600)
    maximum_minutes: StrictInt = Field(ge=5,le=600)


@router.get('/topic-durations')
def read_topic_durations(chapter_ref:str,subject:str,class_level:str,db:Session=Depends(get_db),actor:dict=Depends(require_admin)):
    from services.notes_scope_service import resolve_notes_curriculum
    curriculum = resolve_notes_curriculum(db,chapter_ref,subject=subject,class_level=class_level)
    if not curriculum:
        raise HTTPException(404,detail='Select a supported curriculum chapter')
    return {'curriculum_key':curriculum['curriculum_key'],'topics':[{'id':u['id'],'title':u['title'],
            'estimated_minutes':u['estimated_minutes'],'duration_source':u.get('duration_source','configured')} for u in curriculum['units']]}


@router.put('/topic-duration')
def update_topic_duration(payload:TopicDurationRequest,request:Request,db:Session=Depends(get_db),actor:dict=Depends(require_admin)):
    from services.notes_config_service import set_topic_duration
    try:
        curriculum = _curriculum(db,payload)
        result = set_topic_duration(db,curriculum=curriculum,unit_id=payload.unit_id,minimum=payload.minimum_minutes,
                                   maximum=payload.maximum_minutes,actor_uid=actor['uid'])
        record_admin_audit(db,current_admin=actor,action='notes_topic_duration',target_type='planning_unit',
                           target_id=payload.unit_id,metadata=result,request=request)
        return result
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400,detail=str(exc)) from exc
