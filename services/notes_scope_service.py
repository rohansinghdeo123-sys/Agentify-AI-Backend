"""Resolve existing curricula and source aliases without creating a second hierarchy."""
from __future__ import annotations
import re
from sqlalchemy import func, or_
from sqlalchemy.orm import defer
from Logic.planning.curriculum_registry import load_planning_curricula
from Logic.planning.published_curriculum import build_published_planning_curriculum
from Logic.content_pipeline import APPROVED_STATUSES
from models import ContentChapter, ContentConcept


def identity(value):
    return re.sub(r'[^a-z0-9]+','_',str(value or '').strip().lower()).strip('_')


def class_identity(value):
    value = identity(value).removeprefix('class_').removeprefix('grade_')
    return dict(zip(['i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii'],map(str,range(1,13)))).get(value,value)


def same_scope(curriculum, subject=None, class_level=None):
    return (not subject or identity(subject) == identity(curriculum['subject'])) and (not class_level or class_identity(class_level) == class_identity(curriculum['class_level']))


def _approved_query(db, subject=None, class_level=None):
    query = db.query(ContentChapter).options(defer(ContentChapter.validation_report)).filter(ContentChapter.status.in_(APPROVED_STATUSES))
    if subject:
        query = query.filter(func.lower(ContentChapter.subject) == str(subject).strip().lower())
    if class_level:
        number = class_identity(class_level)
        roman = dict(zip(map(str,range(1,13)),['i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii'])).get(number,number)
        variants = {prefix+value for prefix in ['', 'class','grade'] for value in [number,roman]}
        query = query.filter(func.lower(func.replace(ContentChapter.class_level,' ','' )).in_(variants))
    return query


def _chapter_matches(curriculum, requested):
    names = {identity(curriculum['chapter_slug']),identity(curriculum['chapter_title']),*(identity(a) for a in curriculum.get('aliases',[]))}
    if requested in names:
        return True
    scoped = re.fullmatch(r'ncert_class_([0-9ivx]+)_(.+)_chapter_\d+_(.+)',requested)
    return bool(scoped and class_identity(scoped[1]) == class_identity(curriculum['class_level']) and
                identity(scoped[2]) == identity(curriculum['subject']) and scoped[3] == identity(curriculum['chapter_slug']))


def published_chapter(db, curriculum):
    if db is None:
        return None
    rows = _approved_query(db,curriculum['subject'],curriculum['class_level']).filter(ContentChapter.chapter_number == curriculum['chapter_number']).all()
    matches = [r for r in rows if same_scope({'subject':r.subject,'class_level':r.class_level},curriculum['subject'],curriculum['class_level'])
               and (identity(r.slug) == identity(curriculum['chapter_slug']) or identity(r.chapter_name) == identity(curriculum['chapter_title']) or
                    _chapter_matches(curriculum,identity(r.slug)))]
    return matches[0] if len(matches) == 1 else None


def resolve_notes_curriculum(db, chapter_ref, *, subject=None, class_level=None):
    from services.notes_config_service import apply_topic_durations
    requested = identity(chapter_ref)
    if not requested:
        return None
    registered = [c for c in load_planning_curricula() if same_scope(c,subject,class_level) and _chapter_matches(c,requested)]
    if len(registered) == 1:
        return apply_topic_durations(db,registered[0])
    if registered:
        return None
    from services.teacher_notes_service import resolve_teacher_curriculum
    legacy = resolve_teacher_curriculum(chapter_ref)
    if legacy and same_scope(legacy,subject,class_level):
        row = published_chapter(db,legacy)
        if row and (row.published_source_hash or row.source_hash or '').lower() == legacy['source']['sha256'].lower():
            # The reviewed release already has stable learning-unit IDs.
            # Catalog regrouping is an alias, never a replacement curriculum.
            return apply_topic_durations(db,legacy)
    if db is not None:
        names = {str(chapter_ref).strip().lower(),requested,requested.replace('_',' ')}
        rows = _approved_query(db,subject,class_level).filter(or_(func.lower(ContentChapter.slug).in_(names),func.lower(ContentChapter.chapter_name).in_(names))).all()
        matches = [r for r in rows if same_scope({'subject':r.subject,'class_level':r.class_level},subject,class_level)
                   and requested in {identity(r.slug),identity(r.chapter_name)}]
        if len(matches) == 1:
            from services.catalog_service import resolve_catalog_chapter_units
            row = matches[0]
            scope = resolve_catalog_chapter_units(db,chapter_ref=row.slug,subject=row.subject,class_level=row.class_level)
            if scope and scope.get('source') == 'published':
                return apply_topic_durations(db,build_published_planning_curriculum(scope))
        if matches:
            return None
    # Compatibility for the already approved offline Chemistry release. New
    # subjects/chapters use their registered/published mapping above.
    return apply_topic_durations(db,legacy) if legacy and same_scope(legacy,subject,class_level) else None


def _unit_keys(unit):
    return {identity(v) for v in [unit['id'],unit['primary_topic_id'],*unit.get('legacy_topic_ids',[]),
        *(c['id'] for c in unit.get('concepts',[])),*(s['id'] for s in unit['ncert_subtopics'])]}


def resolve_notes_topics(db, curriculum, topic_id, preferred_unit=None):
    units = curriculum['units']
    if not topic_id:
        return [next((u for u in units if u['id'] == preferred_unit),units[0])]
    requested = identity(topic_id)
    canonical = [u for u in units if requested == identity(u['id'])]
    if canonical:
        return canonical
    direct = [u for u in units if requested in {identity(v) for v in [u['primary_topic_id'],*(c['id'] for c in u.get('concepts',[])),*(s['id'] for s in u['ncert_subtopics'])]}]
    if direct:
        return direct
    from services.teacher_notes_service import _topic_aliases
    candidates = _topic_aliases().get(curriculum['curriculum_key'],{}).get('aliases',{}).get(requested,[])
    valid = {u['id'] for u in units}
    if candidates and set(candidates) <= valid:
        return [u for u in units if u['id'] in candidates]
    direct = [u for u in units if requested in _unit_keys(u)]
    if direct:
        return direct
    row = published_chapter(db,curriculum)
    if not row:
        return []
    concepts = db.query(ContentConcept).filter_by(chapter_id=row.id).order_by(ContentConcept.id).all()
    from services.catalog_service import _learning_units_for_chapter
    groups = _learning_units_for_chapter(row,concepts)
    members = [c for c in concepts if requested in {identity(c.concept_id),identity(c.title),*(identity(a) for a in (c.raw_json or {}).get('source_concept_ids',[]))}]
    exact_groups = [g for g in groups if requested in {identity(g['id']),identity(g['label'])}]
    if exact_groups:
        members = [c for g in exact_groups for c in g['concepts']]
    selected = set()
    for member in members:
        matching = [u for u in units if identity(member.concept_id) in _unit_keys(u)]
        if not matching:
            ids = _topic_aliases().get(curriculum['curriculum_key'],{}).get('aliases',{}).get(identity(member.concept_id),[])
            matching = [u for u in units if u['id'] in ids]
        if not matching:
            return []
        selected.update(u['id'] for u in matching)
    return [u for u in units if u['id'] in selected]
