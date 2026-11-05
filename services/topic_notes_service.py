"""NCERT teaching-notes adapter retaining canonical and archived Planning links."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def chapter_one_mapping():
    return json.loads((ROOT / 'data/notes/chapter_1_notes_mapping.json').read_text(encoding='utf-8'))

def mapped_notes(chapter_slug, topic_id, db, user_id):
    from services.teacher_notes_service import resolve_teacher_curriculum, resolve_teacher_unit, _load_lesson
    curriculum = resolve_teacher_curriculum(chapter_slug)
    if not curriculum:
        return None
    from services.study_route_service import _get_state
    state = _get_state(db,user_id)
    from services.study_route_service import _today, refresh_notes_session_anchor
    if state and (state.runtime_json or {}).get('notes_session_date') != _today().isoformat():
        state = _get_state(db,user_id,lock=True)
        refresh_notes_session_anchor(db, state)
    resolved = resolve_teacher_unit(chapter_slug, topic_id, state.current_unit_id if state else None)
    if not resolved:
        # Archived mixed end-of-chapter reviews span several canonical units.
        # Preserve their original scoped reader instead of pretending that
        # one teaching unit covers the entire mixed review.
        if topic_id in curriculum.get('legacy_review_topic_ids', []):
            return None
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="This legacy topic is not mapped to verified NCERT teaching notes. Open this chapter from Planning.")
    _,unit = resolved
    from services.teacher_notes_service import (
        VERSION, SOURCE, compile_lesson, learner_settings, session_portion, selected_teacher_chapter,
    )
    proficiency, style, allocation = learner_settings(db, user_id, state, curriculum, unit)
    pages = compile_lesson(curriculum, unit, proficiency=proficiency, style=style)
    positions = ((state.runtime_json or {}).get('notes_positions_by_version', {}).get(VERSION, {})) if state else {}
    saved = max(0, min(9, positions.get(unit['id'], 0)))
    starts = ((state.runtime_json or {}).get('notes_session_start_positions', {}).get(VERSION, {})) if state else {}
    session_start = max(0, min(9, starts.get(unit['id'], 0)))
    can_save = selected_teacher_chapter(state, curriculum) is not None
    return {'chapter':{'slug':curriculum['chapter_slug'],'name':curriculum['chapter_title'],'class_level':curriculum['class_level'],'subject':curriculum['subject'],'chapter_number':curriculum['chapter_number']},
        'active_unit_id':state.current_unit_id if state else None,
        'can_save_position':can_save,
        'catalog_source':'published' if curriculum['source']['authority'] == 'approved_content_pipeline' or curriculum['curriculum_key'].startswith('published_') else 'planning_manifest',
        'topic_id':unit['id'],'topic_title':_load_lesson(unit['id']).get('title',unit['title']),'estimated_minutes':dict(unit['estimated_minutes']),
        'subtopics':unit['ncert_subtopics'],'pages':pages,'saved_page':saved,
        'review_required':[], 'source':SOURCE, 'content_version':VERSION,
        'source_document':dict(curriculum['source']), 'proficiency':proficiency,
        'explanation_style':style, 'allocated_session_minutes':allocation,
        'session_portion':session_portion(pages, unit, start_page=session_start, session_minutes=allocation),
        'concepts':[]}
