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
    if not curriculum or curriculum['chapter_number'] != 1 or not topic_id:
        return None
    unit = next((u for u in curriculum['units'] if topic_id in {u['id'],u['primary_topic_id'],*u.get('legacy_topic_ids', [])}),None)
    if not unit:
        return None
    mapping = chapter_one_mapping()
    selected = next(u for u in mapping['units'] if u['unit_id'] == unit['id'])
    from services.study_route_service import _get_state
    state = _get_state(db,user_id)
    saved = ((state.runtime_json or {}).get('notes_positions', {}).get(unit['id'], 0)) if state else 0
    for page in selected['pages']:
        if '\ufffd' in page['text']:
            raise ValueError('Unverified chemical notation in mapped notes')
    return {'chapter':{'slug':curriculum['chapter_slug'],'name':curriculum['chapter_title'],'class_level':curriculum['class_level'],'subject':curriculum['subject'],'chapter_number':1},
        'active_unit_id':state.current_unit_id if state else None,
        'topic_id':unit['id'],'topic_title':unit['title'],'estimated_minutes':unit['estimated_minutes'],
        'subtopics':selected['subtopics'],'pages':selected['pages'],'saved_page':saved,
        'review_required':mapping['review_required'], 'source':mapping['source'],
        'proficiency': next((c.get('chapter_proficiency') for c in (state.selections_json or []) if c.get('chapter_ref') in {chapter_slug,curriculum['chapter_slug'],curriculum['chapter_title']}),'know_a_little') if state else 'know_a_little',
        'concepts':[]}
