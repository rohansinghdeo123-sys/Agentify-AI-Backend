"""Audit real released notes in an isolated database; no provider calls.

Run from the backend root: python scripts/validate_adaptive_notes.py
The ignored JSON report includes honest science-body comparisons, independent
of pagination, proficiency labels, questions and active-reading interrupts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database import Base
from models import AdaptiveNoteLesson,ContentChapter,ContentConcept,StudyRouteState
from services.adaptive_content_compiler import PROFICIENCIES,PROTECTED_KINDS,_source_blocks
from services.adaptive_notes_service import compiled_lesson,reading_progress,record_reading_position
from services.notes_material_service import ncert_sections,retrieve_material
from services.notes_scope_service import resolve_notes_curriculum,resolve_notes_topics
from services.teacher_notes_service import teacher_curricula


def _normal(text):
    return re.sub(r'\s+',' ',text).strip()


def _science_body(payload):
    """Count actual teaching only; UI labels/interrupts cannot fake adaptation."""
    return _normal(' '.join(block['text'] for page in payload['pages'] for block in page['blocks']))


def _words(text):
    return len(re.findall(r'\S+',text))


def _restore_sources(db):
    with gzip.open(ROOT/'data/releases/ncert_class_11_chemistry_v1.json.gz','rt',encoding='utf-8') as handle:
        release=json.load(handle)
    concept_count=0
    for source in release['chapters']:
        row=ContentChapter(**{k:v for k,v in source['chapter'].items() if k in ContentChapter.__table__.columns})
        db.add(row);db.flush()
        for concept in source['concepts']:
            db.add(ContentConcept(chapter_id=row.id,**{k:v for k,v in concept.items() if k in ContentConcept.__table__.columns}))
            concept_count+=1
    db.commit()
    return len(release['chapters']),concept_count


def _quality(material,payloads):
    bodies=[_science_body(p) for p in payloads]
    hashes=[hashlib.sha256(body.encode()).hexdigest() for body in bodies]
    # A reordered paragraph or different grouping cannot earn credit for a
    # genuinely different teaching depth. Keep the exact body hash as evidence,
    # while comparing the taught scientific vocabulary and its multiplicity.
    # Protected-statement checks below separately preserve equation order.
    content_hashes=[hashlib.sha256(json.dumps(sorted(Counter(re.findall(r'\S+',body)).items()),ensure_ascii=False).encode()).hexdigest()
                    for body in bodies]
    groups={}
    for proficiency,fingerprint in zip(PROFICIENCIES,content_hashes):
        groups.setdefault(fingerprint,[]).append(proficiency)
    duplicate_groups=[levels for levels in groups.values() if len(levels)>1]
    word_counts=[_words(body) for body in bodies]
    inversions=[{'from':PROFICIENCIES[i],'to':PROFICIENCIES[i+1],
                 'words_before':word_counts[i],'words_after':word_counts[i+1]}
                for i in range(len(PROFICIENCIES)-1) if word_counts[i+1]>word_counts[i]]
    flexible=[b for section in material['sections'] for b in _source_blocks(section)
              if b['kind'] not in PROTECTED_KINDS]
    explanation=' '.join(b['text'] for b in flexible)
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z][a-z])',explanation)
    # A tiny definition or indivisible worked solution cannot honestly become
    # four different explanations. Flag identical views as an error only when
    # the real retrieved source has enough explanatory material to adapt.
    sufficient=len(sentences)>=4 and _words(explanation)>=40
    return {'science_words':dict(zip(PROFICIENCIES,word_counts)),
        'science_body_sha256':dict(zip(PROFICIENCIES,hashes)),
        'science_content_sha256':dict(zip(PROFICIENCIES,content_hashes)),
        'distinct_science_bodies':len(groups),'identical_science_levels':duplicate_groups,
        'sufficient_explanatory_content':sufficient,'flexible_source_words':_words(explanation),
        'depth_inversions':inversions}


def _assert_view(curriculum,unit,material,payload):
    pages=payload['pages'];allowed={s['id'] for s in unit['ncert_subtopics']}
    assert pages,'No generated pages'
    assert payload['estimated_minutes']==unit['estimated_minutes'],'Configured duration changed'
    assert [p['page_number'] for p in pages]==list(range(1,len(pages)+1)),'Nonsequential pages'
    assert len({p['id'] for p in pages})==len(pages),'Duplicate page identities'
    assert {s for p in pages for s in p['source_section_ids']}==allowed,'Page subtopic coverage differs'
    source_ids={s['id'] for s in material['sections']}
    source_pages={s['id']:set(s['source_pages']) for s in material['sections']}
    for page in pages:
        assert page['blocks'] and all(b['text'].strip() for b in page['blocks']),'Blank teaching block'
        assert set(page['source_section_ids'])<=allowed,'Page crossed topic mapping'
        assert '\ufffd' not in page['text'],'Damaged scientific encoding'
        for ref in page['source_refs']:
            assert ref['id'] in source_ids and set(ref['subtopic_ids'])<=allowed,'Unknown or unscoped provenance'
            assert set(ref['source_pages'])<=source_pages[ref['id']],'Unverified source page attribution'
    rendered=[b for page in pages for b in page['blocks']]
    assert set(payload['essential_block_ids'])<={b['id'] for b in rendered},'Essential block omitted'
    for section in material['sections']:
        evidence=' '.join(_normal(b.get('expanded_text') or b['text']) for b in rendered
                          if section['id'] in b['source_section_ids'])
        for block in _source_blocks(section):
            if block['kind'] in PROTECTED_KINDS:
                assert _normal(block['text']) in evidence,'Protected scientific statement changed or omitted'
        source_science=' '.join(b['text'] for b in _source_blocks(section)
                               if b['kind'] in PROTECTED_KINDS)
        for notation in re.findall(r'\b(?:[A-Z][a-z]?\d*[₀₁₂₃₄₅₆₇₈₉]*){2,}(?:[⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?',source_science):
            assert notation in evidence,'Chemical notation altered: '+notation


def validate():
    started=time.perf_counter()
    engine=create_engine('sqlite://')
    Base.metadata.create_all(engine)
    report={'chapters':[],'units':[],'errors':[],'quality_findings':[]}
    with Session(engine) as db:
        chapter_count,concept_count=_restore_sources(db)
        ncert_calls=[]
        def ncert(db_arg,curriculum,unit,missing):
            ncert_calls.append({'unit_id':unit['id'],'subtopic_ids':sorted(missing)})
            return ncert_sections(db_arg,curriculum,unit,missing)
        with patch('services.notes_material_service.ncert_sections',side_effect=ncert), \
             patch('Logic.embeddings.embed_query',side_effect=AssertionError('Unexpected embedding call')), \
             patch('Logic.embeddings.embed_texts',side_effect=AssertionError('Unexpected embedding call')):
            for expected in teacher_curricula():
                curriculum=resolve_notes_curriculum(db,expected['chapter_slug'],subject=expected['subject'],class_level=expected['class_level'])
                assert curriculum is not None,'Released chapter did not resolve'
                assert [u['id'] for u in curriculum['units']]==[u['id'] for u in expected['units']],'Existing curriculum IDs/order changed'
                chapter={'chapter_number':curriculum['chapter_number'],'curriculum_key':curriculum['curriculum_key'],'units':len(curriculum['units'])}
                report['chapters'].append(chapter)
                for unit in curriculum['units']:
                    item={'chapter_number':curriculum['chapter_number'],'unit_id':unit['id'],'title':unit['title']}
                    report['units'].append(item)
                    try:
                        assert resolve_notes_topics(db,curriculum,unit['id'])==[unit]
                        for subtopic in unit['ncert_subtopics']:
                            assert unit in resolve_notes_topics(db,curriculum,subtopic['id']),'Existing subtopic lost binding'
                        calls_before=len(ncert_calls)
                        material=retrieve_material(db,curriculum,unit)
                        assert not material['missing_subtopics'],'Uncovered mapped source sections: '+str(material['missing_subtopics'])
                        assert not material['conflicts'],'Source conflict/review required: '+str(material['conflicts'])
                        allowed={s['id'] for s in unit['ncert_subtopics']}
                        assert len(ncert_calls)==calls_before+1 and ncert_calls[-1]['subtopic_ids']==sorted(allowed)
                        assert all(set(s['subtopic_ids'])<=allowed for s in material['sections']),'Retrieved neighboring section'
                        payloads=[]
                        for proficiency in PROFICIENCIES:
                            payload,_=compiled_lesson(db,curriculum,unit,proficiency=proficiency,material=material)
                            _assert_view(curriculum,unit,material,payload)
                            payloads.append(payload)
                        db.commit()
                        rows_before=db.query(AdaptiveNoteLesson).count()
                        with patch('services.adaptive_content_compiler.compile_adaptive_lesson',side_effect=AssertionError('Shared variant cache missed')):
                            cached,_=compiled_lesson(db,curriculum,unit,material=material)
                        assert cached==payloads[0] and db.query(AdaptiveNoteLesson).count()==rows_before
                        # Check real, private versioned progress independently of
                        # the generated page count (including lessons >10 pages).
                        state=StudyRouteState(user_id='audit-'+unit['id'],route_date=date.today(),runtime_json={'mastery_unchanged':True})
                        db.add(state);db.flush()
                        payload=payloads[0];last=len(payload['pages'])-1
                        record_reading_position(db,state,{'notes_version':payload['content_version'],'notes_page':last,
                            'notes_page_id':payload['pages'][last]['id'],'notes_total_pages':len(payload['pages']),
                            'notes_completed_page_ids':[payload['pages'][last]['id']]},curriculum,unit['id'])
                        db.commit();db.refresh(state)
                        progress=reading_progress(state,unit,payload)
                        assert progress['saved_page']==last and progress['completed_page_ids']==[payload['pages'][last]['id']]
                        assert state.runtime_json['mastery_unchanged'] is True
                        item.update(source_types=Counter(s['source_type'] for s in material['sections']),
                            subtopics=len(allowed),
                            pages=dict(zip(PROFICIENCIES,[len(p['pages']) for p in payloads])),cache_reused=True,bookmark_verified=True,
                            **_quality(material,payloads))
                        if item['depth_inversions'] or (item['sufficient_explanatory_content'] and item['identical_science_levels']):
                            report['quality_findings'].append({k:item[k] for k in ['chapter_number','unit_id','title','distinct_science_bodies',
                                'identical_science_levels','sufficient_explanatory_content','science_words','depth_inversions']})
                    except Exception as exc:
                        db.rollback()
                        item['error']=str(exc)
                        report['errors'].append({'unit_id':unit['id'],'reason':str(exc)})
            report['summary']={'restored_chapters':chapter_count,'restored_concepts':concept_count,
                'units':len(report['units']),'persisted_variants':db.query(AdaptiveNoteLesson).count(),
                'ncert_queries':len(ncert_calls),
                'identical_science_body_units':sum(bool(u.get('identical_science_levels')) for u in report['units']),
                'depth_inversion_units':sum(bool(u.get('depth_inversions')) for u in report['units']),
                'quality_findings':len(report['quality_findings']),'errors':len(report['errors']),
                'elapsed_seconds':round(time.perf_counter()-started,2)}
    engine.dispose()
    assert report['summary']['units']==72,'The audit did not visit all72 existing units'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'tmp/adaptive_notes_validation.json')
    parser.add_argument('--allow-quality-findings',action='store_true',help='Keep quality findings visible but return failure only for source/integration errors')
    args=parser.parse_args()
    report=validate()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({**report['summary'],'report':str(args.report)},ensure_ascii=False,indent=2))
    return 1 if report['errors'] or (report['quality_findings'] and not args.allow_quality_findings) else 0


if __name__=='__main__':
    raise SystemExit(main())
