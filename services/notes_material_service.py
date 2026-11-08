"""Actual section retrieval: verified Master Notes first, only genuine NCERT gaps second."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from models import ContentConcept
from services.notes_scope_service import identity, published_chapter

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def _sentences(text):
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|\n\s*\n',text) if p.strip()]


def _normalized(text):
    return re.sub(r'\s+',' ',text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉','0123456789')).replace('−','-')).strip().lower()


def _reviewed_sections(db, curriculum, unit, missing):
    """Reuse reviewed teaching content only if its source AND mapping still match."""
    from services.teacher_notes_service import teacher_curricula, _load_lesson, _verify_source, _source_pages
    original = next((c for c in teacher_curricula() if c['curriculum_key'] == curriculum['curriculum_key']),None)
    if not original:
        return []
    old_unit = next((u for u in original['units'] if u['id'] == unit['id']),None)
    if not old_unit or {s['id'] for s in old_unit['ncert_subtopics']} != {s['id'] for s in unit['ncert_subtopics']}:
        return []
    row = published_chapter(db,curriculum)
    expected = original['source']['sha256'].lower()
    if row and (row.published_source_hash or row.source_hash or '').lower() != expected:
        return []
    _verify_source(original['source']['path'],expected)
    lesson = _load_lesson(unit['id'])
    sections = []
    for page in lesson['pages']:
        ids = page['source_section_ids']
        if not set(ids) <= missing:
            continue
        sections.append({'id':'reviewed-ncert:'+page['id'],'title':page['title'],
            'text':'\n\n'.join(b['text'] for b in page['blocks']), 'blocks':page['blocks'],
            'questions':lesson['questions'] if page['page_number'] == len(lesson['pages']) else [],
            'subtopic_ids':ids,'source_pages':_source_pages(old_unit,ids),'source_type':'NCERT_FALLBACK',
            'teaching_format':'guided_lesson','source_document':{'title':curriculum['chapter_title'],'sha256':expected,'role':'NCERT_FALLBACK'},
            'derivation':'reviewed_teacher_explanation'})
    return sections


def _pdf_section(curriculum, unit, subtopic):
    source = curriculum.get('source') or {}
    if source.get('type') != 'pdf' or not re.fullmatch(r'\d+(?:\.\d+)+',subtopic):
        return []
    path = (ROOT / source['path']).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        return []
    from services.teacher_notes_service import _verify_source
    from Logic.tools.knowledge_search import _cached_planning_pdf_pages
    _verify_source(source['path'],source['sha256'])
    mapped = next(s for s in unit['ncert_subtopics'] if s['id'] == subtopic)
    refs = [s for s in unit.get('ncert_sections',[]) if s['id'] == mapped['section_id'] and s.get('pages')]
    allowed = {p for s in refs for p in range(s['pages']['start'],s['pages']['end']+1)}
    if unit.get('source_segments'):
        allowed &= {s['page'] for s in unit['source_segments']}
    pages = [(n,t) for n,t in _cached_planning_pdf_pages(str(path),source['sha256']) if n in allowed]
    joined = '\n'.join(t for _,t in pages)
    start = re.search(r'(?im)^\s*'+re.escape(subtopic)+r'\s+[A-Za-z]',joined)
    if not start:
        return []
    next_heading = re.search(r'(?m)^\s*\d+(?:\.\d+)+\s+[A-Za-z]',joined[start.end():])
    text = joined[start.start():start.end()+next_heading.start() if next_heading else len(joined)].strip()
    if not text:
        return []
    return [{'id':'ncert-section:'+subtopic,'title':mapped['title'],'text':text,'subtopic_ids':[subtopic],
             'source_pages':sorted(allowed),'source_type':'NCERT_FALLBACK','teaching_format':'concept',
             'source_document':{'title':curriculum['chapter_title'],'sha256':source['sha256'],'role':'NCERT_FALLBACK'}}]


def ncert_sections(db, curriculum, unit, missing):
    """Called only for uncovered IDs; no chapter-wide retrieval/embedding request."""
    sections = _reviewed_sections(db,curriculum,unit,missing)
    covered = {sid for section in sections for sid in section['subtopic_ids']}
    row, concepts = None, None
    for gap in [s['id'] for s in unit['ncert_subtopics'] if s['id'] in missing-covered]:
        exact = _pdf_section(curriculum,unit,gap)
        if exact:
            sections.extend(exact)
            continue
        row = row or published_chapter(db,curriculum)
        if not row:
            continue
        # Published subtopic IDs reference approved source concepts, not
        # neighbouring physical pages. Keep that exact binding for new subjects.
        if concepts is None:
            concepts = db.query(ContentConcept).filter_by(chapter_id=row.id).order_by(ContentConcept.id).all()
        matched = [c for c in concepts if identity(gap) in {identity(c.concept_id),*(identity(a) for a in (c.raw_json or {}).get('source_concept_ids',[]))}]
        if len(matched) != 1:
            continue
        concept = matched[0]
        blocks = []
        seen = set()
        def add(kind,text,level='essential'):
            if not isinstance(text,str) or not text.strip() or _normalized(text) in seen:
                return
            seen.add(_normalized(text)); blocks.append({'kind':kind,'text':text.strip(),'level':level})
        add('definition',concept.definition)
        for text in (concept.core_explanation or '').split('\n\n'):
            add('explanation',text)
        for text in concept.key_points or []: add('explanation',text)
        for text in concept.properties or []: add('classification',text)
        for text in concept.applications or []: add('example',text)
        for text in concept.formulas or []: add('formula',text)
        for text in concept.examples or []: add('example',text)
        for text in concept.common_mistakes or []:
            add('misconception',text if isinstance(text,str) else ' '.join(str(v) for v in text.values() if isinstance(v,str)))
        if blocks:
            sections.append({'id':'ncert-concept:'+concept.concept_id,'title':concept.title,'text':'\n\n'.join(b['text'] for b in blocks),
                'blocks':blocks,'source_type':'NCERT_FALLBACK','subtopic_ids':[gap],
                'source_pages':concept.source_pages or [],'teaching_format':'concept',
                'source_document':{'id':row.id,'title':row.chapter_name,'sha256':row.published_source_hash or row.source_hash,'role':'NCERT_FALLBACK'},
                'derivation':'approved_section_index'})
    return sections


def retrieve_material(db, curriculum, unit, requested_subtopic_ids=None):
    from services.notes_documents_service import find_master_sections, _formula_conflict
    allowed = {s['id'] for s in unit['ncert_subtopics']}
    requested = allowed if requested_subtopic_ids is None else set(requested_subtopic_ids)
    if not requested or not requested <= allowed:
        raise ValueError('Requested material is outside the selected mapped subtopics')
    master = find_master_sections(db,curriculum,unit,requested)
    sections = list(master['sections'])
    conflicts = list(master.get('conflicts') or [])
    covered = set(master['covered_subtopic_ids']) & requested
    blocked = {sid for c in conflicts if 'Conflicting master-note' in c.get('reason','') for sid in c.get('subtopic_ids',[])}
    missing = requested-covered-blocked
    # This branch is the source-priority guarantee: a sufficient Master Notes
    # response never invokes the NCERT adapter, even to warm a cache.
    supplementary = ncert_sections(db,curriculum,unit,missing) if missing else []
    known = {_normalized(sentence) for s in sections for sentence in _sentences(s['text'])}
    for section in supplementary:
        ids = set(section['subtopic_ids'])
        if not ids <= missing:
            raise ValueError('NCERT supplement crossed its requested gap')
        if any(_formula_conflict(s['text'],section['text']) for s in sections if s['source_type'] == 'MASTER_NOTES'):
            conflicts.append({'subtopic_ids':section['subtopic_ids'],'section_ids':[section['id']],
                              'reason':'Conflicting formula statements require source review; the supplement was withheld.'})
            continue
        if section.get('blocks'):
            blocks = []
            for block in section['blocks']:
                parts = _sentences(block['text'])
                if parts and all(_normalized(p) in known for p in parts):
                    continue
                if block['kind'] not in {'worked_example','derivation','formula'}:
                    block = {**block,'text':' '.join(p for p in parts if _normalized(p) not in known)}
                if block['text'].strip():
                    blocks.append(block)
            section = {**section,'blocks':blocks,'text':'\n\n'.join(b['text'] for b in blocks)}
        else:
            section = {**section,'text':' '.join(s for s in _sentences(section['text']) if _normalized(s) not in known)}
        if not section['text'].strip():
            continue
        sections.append(section)
        known.update(_normalized(s) for s in _sentences(section['text']))
    # A partial Master section is useful evidence, but its mere presence must
    # not hide an unresolved learning gap when no valid supplement was found.
    actual = covered | {sid for s in sections if s['source_type'] == 'NCERT_FALLBACK' for sid in s['subtopic_ids']}
    ordered = {s['id']:i for i,s in enumerate(unit['ncert_subtopics'])}
    # Preserve each document's organization and keep supplementary content after
    # its primary source. Never interleave the same definition across sources.
    sections.sort(key=lambda s:(s['source_type'] != 'MASTER_NOTES',min(ordered[i] for i in s['subtopic_ids'])))
    fingerprint = digest({'master':master['source_fingerprint'],'requested':sorted(requested),'sections':[
        (s['id'],s['source_type'],s.get('source_document'),s['text'],s.get('blocks'),s.get('questions')) for s in sections],
        'conflicts':conflicts,'unit':unit['id']})
    return {'curriculum':curriculum,'unit':unit,'sections':sections,'conflicts':conflicts,
            'missing_subtopics':sorted(requested-actual),'master_covered_subtopics':sorted(covered),
            'source_fingerprint':fingerprint}
