"""NCERT teaching-notes adapter retaining canonical and archived Planning links."""
import json
from functools import lru_cache
from pathlib import Path
from Logic.planning.curriculum_registry import resolve_planning_curriculum

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def chapter_one_mapping():
    return json.loads((ROOT / 'data/notes/chapter_1_notes_mapping.json').read_text(encoding='utf-8'))

def mapped_notes(chapter_slug, topic_id, db, user_id):
    curriculum = resolve_planning_curriculum(chapter_ref=chapter_slug, subject='Chemistry', class_level='Class 11')
    if not curriculum or curriculum['chapter_number'] != 1:
        return None
    from services.study_route_service import _get_state
    state = _get_state(db,user_id)
    from services.study_route_service import _today, refresh_notes_session_anchor
    if state and (state.runtime_json or {}).get('notes_session_date') != _today().isoformat():
        state = _get_state(db,user_id,lock=True)
        refresh_notes_session_anchor(db, state)
    from services.master_notes_service import resolve_unit
    resolved = resolve_unit(chapter_slug, topic_id, state.current_unit_id if state else None)
    if not resolved:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="This legacy topic is not mapped to verified NCERT teaching notes. Open this chapter from Planning.")
    _,unit = resolved
    from services.teacher_notes_service import (
        VERSION, SOURCE, compile_lesson, learner_settings, session_portion,
    )
    proficiency, style, allocation = learner_settings(db, user_id, state, curriculum, unit)
    pages = compile_lesson(curriculum, unit, proficiency=proficiency, style=style)
    positions = ((state.runtime_json or {}).get('notes_positions_by_version', {}).get(VERSION, {})) if state else {}
    saved = max(0, min(9, positions.get(unit['id'], 0)))
    starts = ((state.runtime_json or {}).get('notes_session_start_positions', {}).get(VERSION, {})) if state else {}
    session_start = max(0, min(9, starts.get(unit['id'], 0)))
    return {'chapter':{'slug':curriculum['chapter_slug'],'name':curriculum['chapter_title'],'class_level':curriculum['class_level'],'subject':curriculum['subject'],'chapter_number':1},
        'active_unit_id':state.current_unit_id if state else None,
        'topic_id':unit['id'],'topic_title':unit['title'],'estimated_minutes':dict(unit['estimated_minutes']),
        'subtopics':unit['ncert_subtopics'],'pages':pages,'saved_page':saved,
        'review_required':[], 'source':SOURCE, 'content_version':VERSION,
        'source_document':dict(curriculum['source']), 'proficiency':proficiency,
        'explanation_style':style, 'allocated_session_minutes':allocation,
        'session_portion':session_portion(pages, unit, start_page=session_start, session_minutes=allocation),
        'concepts':[]}
