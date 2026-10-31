"""Explicit source excerpts; never return neighbouring merged master-note content."""
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
    from services.master_notes_service import resolve_unit
    resolved = resolve_unit(chapter_slug, topic_id, state.current_unit_id if state else None)
    if not resolved:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="This legacy topic is not mapped to verified Master Notes. Open this chapter from Planning.")
    _,unit = resolved
    topic_id = unit['id']
    mapping = chapter_one_mapping()
    selected = next(u for u in mapping['units'] if u['unit_id'] == unit['id'])
    saved = ((state.runtime_json or {}).get('notes_positions', {}).get(unit['id'], 0)) if state else 0
    from services.master_notes_service import retrieve_notes, lesson_pages
    retrieval = retrieve_notes(db, chapter_slug, topic_id)
    pages = lesson_pages(retrieval)
    for page in pages:
        if '\ufffd' in page['text']:
            raise ValueError('Unverified chemical notation in mapped notes')
    return {'chapter':{'slug':curriculum['chapter_slug'],'name':curriculum['chapter_title'],'class_level':curriculum['class_level'],'subject':curriculum['subject'],'chapter_number':1},
        'active_unit_id':state.current_unit_id if state else None,
        'topic_id':unit['id'],'topic_title':unit['title'],'estimated_minutes':unit['estimated_minutes'],
        'subtopics':selected['subtopics'],'pages':pages,'saved_page':saved,
        'review_required':retrieval['conflicts'], 'source':'Amit Master Notes first; NCERT supplements only for gaps',
        'proficiency': next((c.get('chapter_proficiency') for c in (state.selections_json or []) if c.get('chapter_ref') in {chapter_slug,curriculum['chapter_slug'],curriculum['chapter_title']}),'know_a_little') if state else 'know_a_little',
        'concepts':[]}
