"""NCERT teaching-notes adapter retaining canonical and archived Planning links."""
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def chapter_one_mapping():
    return json.loads((ROOT / 'data/notes/chapter_1_notes_mapping.json').read_text(encoding='utf-8'))


def _reading_page(value):
    try:
        page = float(value) if not isinstance(value, bool) else 0
        return max(0, min(9, math.floor(page))) if math.isfinite(page) else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def _published_topic_units(db, curriculum, topic_id):
    """Resolve a newly regrouped catalog ID through exact approved member IDs.

    A different grouping can produce a new hash without changing its source
    concepts. Never guess from a hash/title or silently drop an unknown member.
    """
    from services.catalog_service import resolve_catalog_topic
    from services.teacher_notes_service import resolve_teacher_topics, resolve_teacher_curriculum
    scoped = resolve_catalog_topic(db, topic_id, chapter=curriculum['chapter_slug'],
                                   subject='Chemistry', class_level='Class 11', catalog_source='published')
    if not scoped or str(scoped.get('class_level')).lower() not in {'11','class 11','xi','class xi'}:
        return []
    source = resolve_teacher_curriculum(scoped.get('chapter_slug'))
    if not source or source['curriculum_key'] != curriculum['curriculum_key']:
        return []
    identities = set()
    for member in scoped.get('concept_ids') or []:
        resolved = resolve_teacher_topics(curriculum['chapter_slug'], member)
        if not resolved or not resolved[1]:
            return []
        identities.update(u['id'] for u in resolved[1])
    return [u for u in curriculum['units'] if u['id'] in identities]


def mapped_notes(chapter_slug, topic_id, db, user_id):
    from services.teacher_notes_service import resolve_teacher_curriculum, resolve_teacher_topics, _load_lesson
    curriculum = resolve_teacher_curriculum(chapter_slug)
    if not curriculum:
        return None
    from services.study_route_service import _get_state
    state = _get_state(db,user_id)
    from services.study_route_service import _today, refresh_notes_session_anchor
    if state and (state.runtime_json or {}).get('notes_session_date') != _today().isoformat():
        state = _get_state(db,user_id,lock=True)
        refresh_notes_session_anchor(db, state)
    resolved = resolve_teacher_topics(chapter_slug, topic_id, state.current_unit_id if state else None)
    if resolved and not resolved[1] and topic_id:
        resolved = curriculum, _published_topic_units(db, curriculum, topic_id)
    if not resolved or not resolved[1]:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="This legacy topic is not mapped to verified NCERT teaching notes. Open this chapter from Planning.")
    chapter = {'slug':curriculum['chapter_slug'],'name':curriculum['chapter_title'],'class_level':curriculum['class_level'],'subject':curriculum['subject'],'chapter_number':curriculum['chapter_number']}
    if len(resolved[1]) > 1:
        return {'chapter':chapter,'resolution':'topic_choice','requested_topic_id':topic_id,
                'topic_title':'Choose a learning topic','pages':[],'concepts':[],
                'can_save_position':False,
                'topic_options':[{'id':u['id'],'title':_load_lesson(u['id']).get('title',u['title']),
                                  'estimated_minutes':dict(u['estimated_minutes'])} for u in resolved[1]]}
    unit = resolved[1][0]
    from services.teacher_notes_service import (
        VERSION, SOURCE, compile_lesson, learner_settings, session_portion, selected_teacher_chapter,
    )
    proficiency, style, allocation = learner_settings(db, user_id, state, curriculum, unit)
    pages = compile_lesson(curriculum, unit, proficiency=proficiency, style=style)
    positions = ((state.runtime_json or {}).get('notes_positions_by_version', {}).get(VERSION, {})) if state else {}
    saved = _reading_page(positions.get(unit['id'], 0))
    # A direct NCERT subtopic link opens its first teaching step instead of
    # dropping the learner at an unrelated saved page in the parent lesson.
    from services.teacher_notes_service import _topic_identity
    selected_subtopics = [s['id'] for s in unit['ncert_subtopics'] if _topic_identity(topic_id) == _topic_identity(s['id'])]
    if selected_subtopics and topic_id != unit['id']:
        saved = next((i for i,p in enumerate(pages) if set(selected_subtopics).intersection(p['source_section_ids'])), saved)
    starts = ((state.runtime_json or {}).get('notes_session_start_positions', {}).get(VERSION, {})) if state else {}
    session_start = _reading_page(starts.get(unit['id'], 0))
    can_save = selected_teacher_chapter(state, curriculum) is not None
    return {'chapter':chapter,'resolution':'lesson','requested_topic_id':topic_id,
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
