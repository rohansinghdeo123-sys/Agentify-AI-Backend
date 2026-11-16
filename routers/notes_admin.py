"""Administrator overrides for Planning topic durations."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy.orm import Session

from app.security import require_admin
from database import get_db
from services.admin_service import record_admin_audit

router = APIRouter(prefix='/admin/notes', tags=['admin-notes'])


def _curriculum(db,payload):
    from services.notes_scope_service import resolve_notes_curriculum
    curriculum = resolve_notes_curriculum(db,payload.chapter_ref,subject=payload.subject,class_level=payload.class_level)
    if not curriculum:
        raise ValueError('Select an existing, unambiguous curriculum chapter in the exact class and subject')
    return curriculum


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
