"""Exercise the actual authenticated notes/heartbeat transport for every released link.

Uses an isolated database and the real NCERT Chemistry source release.
No provider, production database or student account is used. Run:
    python scripts/audit_notes_endpoints.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.security import verify_firebase_user
from database import Base, get_db
from models import AICoachProfile, AdaptiveNoteLesson, StudyRouteState, TestHistory
from routers.planning import router as planning_router
from routers.study import router as study_router
from scripts.validate_adaptive_notes import _restore_sources
from services.adaptive_notes_service import retrieve_context
from services.catalog_service import build_catalog
from services.notes_scope_service import identity, resolve_notes_curriculum, same_scope
from services.study_route_service import _get_state, start_route
from services.teacher_notes_service import _topic_aliases, teacher_curricula

from services.adaptive_content_compiler import PROFICIENCIES
STYLES=('balanced','concise','detailed')


def _links(curriculum, catalog_chapter, source_concepts):
    topics={topic['id'] for topic in catalog_chapter['topics']}
    subtopics={}
    for unit in curriculum['units']:
        topics.update([unit['id'],unit['primary_topic_id'],*unit.get('legacy_topic_ids',[])])
        topics.update(c['id'] for c in unit.get('concepts',[]))
        for subtopic in unit['ncert_subtopics']:
            for name in [subtopic['id'],identity(subtopic['id'])]:
                topics.add(name)
                subtopics.setdefault(name,set()).add(subtopic['id'])
    for concept in source_concepts:
        topics.add(concept['concept_id'])
        topics.update((concept.get('raw_json') or {}).get('source_concept_ids') or [])
    topics.update(_topic_aliases()[curriculum['curriculum_key']]['aliases'])
    return sorted(topics),subtopics


def _read(client, curriculum, topic, *, version=None):
    params={'topic_id':topic,'class_level':curriculum['class_level'],'subject':curriculum['subject']}
    if version:
        params['content_version']=version
    response=client.get('/notes/'+curriculum['chapter_slug'],params=params)
    assert response.status_code==200,f'{response.status_code}: {response.text[:350]}'
    notes=response.json()
    assert notes['chapter']['subject']==curriculum['subject'],'Cross-subject response'
    assert notes['chapter']['chapter_number']==curriculum['chapter_number'],'Cross-chapter response'
    return notes


def _assert_lesson(notes, curriculum, expected_unit=None):
    assert notes.get('resolution')=='lesson','Unexpected notes state: '+str(notes.get('resolution'))
    assert not notes.get('review_required'),'Unexpected source coverage/review gap'
    unit=next(u for u in curriculum['units'] if u['id']==notes['topic_id'])
    assert not expected_unit or unit['id']==expected_unit,'Wrong canonical topic'
    allowed={s['id'] for s in unit['ncert_subtopics']}
    pages=notes['pages']
    assert pages and len({p['id'] for p in pages})==len(pages),'Missing/duplicate pages'
    assert [p['page_number'] for p in pages]==list(range(1,len(pages)+1)),'Nonsequential pages'
    assert {sid for page in pages for sid in page['source_section_ids']}==allowed,'Missing/unrelated mapped subtopic'
    assert notes['estimated_minutes']==unit['estimated_minutes'],'Topic duration changed'
    assert 0<=notes['saved_page']<len(pages),'Invalid saved position'
    assert notes['saved_page_id']==pages[notes['saved_page']]['id'],'Page ID/index disagree'
    for page in pages:
        assert page['blocks'] and page['text'].strip(),'Blank page'
        assert all(block['text'].strip() for block in page['blocks']),'Blank teaching block'
        assert set(page['source_section_ids'])<=allowed,'Cross-topic page'
        assert all(ref['source_type'] in {'NCERT'} for ref in page['source_refs']),'Untracked source'
    return unit


def audit(*, chapter_numbers=None, proficiencies=PROFICIENCIES, styles=STYLES, aliases=True, progress=True):
    started=time.perf_counter()
    engine=create_engine('sqlite://',poolclass=StaticPool,connect_args={'check_same_thread':False})
    Base.metadata.create_all(engine)
    report={'scope':'bundled Class XI Chemistry release','chapters':[],'errors':[]}
    token={'uid':'alias-audit-reader'}
    with Session(engine) as db:
        _restore_sources(db)
        with gzip.open(ROOT/'data/releases/ncert_class_11_chemistry_v1.json.gz','rt',encoding='utf-8') as handle:
            release=json.load(handle)
        concepts={c['chapter']['chapter_number']:c['concepts'] for c in release['chapters']}
        curricula = teacher_curricula()
        assert {c['chapter_number'] for c in curricula} == set(concepts), 'The curriculum omits a released chapter'
        app=FastAPI();app.include_router(study_router);app.include_router(planning_router)
        app.dependency_overrides[get_db]=lambda:db
        app.dependency_overrides[verify_firebase_user]=lambda:dict(token)
        with TestClient(app) as client, \
             patch('app.config.STUDY_LAB_PLANNING_INTEGRATION',True), \
             patch('Logic.embeddings.embed_query',side_effect=AssertionError('Unexpected model call')), \
             patch('Logic.embeddings.embed_texts',side_effect=AssertionError('Unexpected model call')):
            catalog=build_catalog(db)
            catalog_chapters = {ch['chapter_number']:ch for group in catalog['subjects']
                                if same_scope(group,'Chemistry','Class XI') for ch in group['chapters']}
            assert set(catalog_chapters) == set(concepts), 'The Chemistry catalog differs from the release'
            for original in curricula:
                if chapter_numbers and original['chapter_number'] not in chapter_numbers:
                    continue
                curriculum=resolve_notes_curriculum(db,original['chapter_slug'],subject=original['subject'],class_level=original['class_level'])
                assert curriculum is not None,'Released curriculum missing'
                catalog_chapter=catalog_chapters[curriculum['chapter_number']]
                item={'chapter_number':curriculum['chapter_number'],'title':curriculum['chapter_title'],
                      'units':len(curriculum['units']),'subtopics':sum(len(u['ncert_subtopics']) for u in curriculum['units']),
                      'link_requests':0,'choices':0,'choice_options_opened':0,'variants':0,'progress_roundtrips':0,
                      'subtopic_bookmark_checks':0,'subtopic_reanchors':0,'subtopic_tutor_checks':0,'page_counts':Counter()}
                report['chapters'].append(item)
                if aliases:
                    token['uid']='alias-audit-reader'
                    links,subtopic_links=_links(curriculum,catalog_chapter,concepts[curriculum['chapter_number']])
                    opened=set()
                    for topic in links:
                        try:
                            notes=_read(client,curriculum,topic)
                            item['link_requests']+=1
                            if notes.get('resolution')=='topic_choice':
                                options=[o['id'] for o in notes['topic_options']]
                                assert len(options)>1 and len(set(options))==len(options),'Empty/duplicate choices'
                                assert options==[u['id'] for u in curriculum['units'] if u['id'] in options],'Unscoped/reordered choices'
                                item['choices']+=1
                                for unit_id in set(options)-opened:
                                    _assert_lesson(_read(client,curriculum,unit_id),curriculum,unit_id)
                                    opened.add(unit_id);item['choice_options_opened']+=1
                            else:
                                _assert_lesson(notes,curriculum)
                                requested=subtopic_links.get(topic,set())
                                if requested:
                                    assert requested.intersection(notes['pages'][notes['saved_page']]['source_section_ids']),'Subtopic opens on unrelated page'
                        except Exception as exc:
                            db.rollback();report['errors'].append({'chapter_number':curriculum['chapter_number'],'topic':topic,'phase':'link','reason':str(exc)})
                for proficiency in proficiencies:
                    for style in styles:
                        user_id=f'http-c{curriculum["chapter_number"]}-{proficiency}-{style}'
                        token['uid']=user_id
                        db.add(AICoachProfile(user_id=user_id,coach_id=user_id,study_preferences={'preferred_style':style}))
                        db.commit()
                        start_route(db,user_id=user_id,class_level=curriculum['class_level'],subject=curriculum['subject'],study_time_today='15',
                                    selections=[{'chapter_ref':curriculum['chapter_slug'],'chapter_proficiency':proficiency}])
                        for unit in curriculum['units']:
                            try:
                                notes=_read(client,curriculum,unit['id'])
                                _assert_lesson(notes,curriculum,unit['id'])
                                assert notes['proficiency']==proficiency and notes['explanation_style']==style,'Learner variant ignored'
                                item['variants']+=1;item['page_counts'][len(notes['pages'])]+=1
                                if progress:
                                    state=_get_state(db,user_id);active=state.current_unit_id;last=len(notes['pages'])-1
                                    response=client.post('/planning/route/heartbeat/'+user_id+'?compact=true',json={'last_position':{
                                        'unit_id':active,'notes_topic':unit['id'],'notes_version':notes['content_version'],
                                        'notes_page':last,'notes_page_id':notes['pages'][last]['id'],'notes_total_pages':len(notes['pages']),
                                        'notes_completed_page_ids':[notes['pages'][last]['id']]}})
                                    assert response.status_code==200,f'Progress failed: {response.status_code}: {response.text[:200]}'
                                    reopened=_read(client,curriculum,unit['id'],version=notes['content_version'])
                                    assert reopened['saved_page']==last and reopened['saved_page_id']==notes['pages'][last]['id'],'Bookmark lost'
                                    assert reopened['completed_page_ids']==[notes['pages'][last]['id']],'Completion lost'
                                    assert _get_state(db,user_id).current_unit_id==active,'Reading changed active route'
                                    item['progress_roundtrips']+=1
                                    if proficiency==proficiencies[0] and style==styles[0]:
                                        for subtopic in unit['ncert_subtopics']:
                                            sid=subtopic['id']
                                            entry=_read(client,curriculum,sid)
                                            _assert_lesson(entry,curriculum,unit['id'])
                                            page=entry['pages'][entry['saved_page']]
                                            assert sid in page['source_section_ids'],f'{sid} opened at an unrelated saved page'
                                            assert entry['completed_page_ids']==reopened['completed_page_ids'],'Subtopic link changed completion'
                                            item['subtopic_bookmark_checks']+=1
                                            if sid not in notes['pages'][last]['source_section_ids']:
                                                item['subtopic_reanchors']+=1
                                            else:
                                                assert page['id']==notes['pages'][last]['id'],'Matching bookmark was discarded'
                                            context=retrieve_context(db,curriculum['chapter_slug'],unit['id'],
                                                subject=curriculum['subject'],class_level=curriculum['class_level'],
                                                content_version=entry['content_version'],page_id=page['id'],page_number=page['page_number'],
                                                subtopic_ids=page['source_section_ids'],proficiency=proficiency,style=style)
                                            assert context['matched_sections']==[page['id']],'Ask AI opened a different page'
                                            assert context['provenance']==page['source_refs'],'Ask AI returned unrelated provenance'
                                            assert context['source_types']==page['source_types'],'Ask AI reported an unused source'
                                            assert all((b.get('expanded_text') or b['text']) in context['context'] for b in page['blocks'])
                                            item['subtopic_tutor_checks']+=1
                                        canonical=_read(client,curriculum,unit['id'])
                                        assert canonical['saved_page_id']==reopened['saved_page_id'],'Deep link overwrote the saved bookmark'
                            except Exception as exc:
                                db.rollback();report['errors'].append({'chapter_number':curriculum['chapter_number'],'topic':unit['id'],
                                    'proficiency':proficiency,'style':style,'phase':'variant','reason':str(exc)})
                item['page_counts']=dict(sorted(item['page_counts'].items()))
            assert db.query(TestHistory).count()==0,'Reading awarded assessment/mastery'
            report['summary']={'chapters':len(report['chapters']),'units':sum(c['units'] for c in report['chapters']),
                'subtopics':sum(c['subtopics'] for c in report['chapters']),
                'link_requests':sum(c['link_requests'] for c in report['chapters']),
                'ambiguous_links_with_choices':sum(c['choices'] for c in report['chapters']),
                'verified_choice_options':sum(c['choice_options_opened'] for c in report['chapters']),
                'variants':sum(c['variants'] for c in report['chapters']),
                'progress_roundtrips':sum(c['progress_roundtrips'] for c in report['chapters']),
                'subtopic_bookmark_checks':sum(c['subtopic_bookmark_checks'] for c in report['chapters']),
                'subtopic_reanchors':sum(c['subtopic_reanchors'] for c in report['chapters']),
                'subtopic_tutor_checks':sum(c['subtopic_tutor_checks'] for c in report['chapters']),
                'persisted_variants':db.query(AdaptiveNoteLesson).count(),
                'errors':len(report['errors']),'elapsed_seconds':round(time.perf_counter()-started,2)}
    engine.dispose()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'tmp/notes_endpoint_audit.json')
    args=parser.parse_args()
    report=audit()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({**report['summary'],'report':str(args.report)},indent=2))
    return 1 if report['errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
