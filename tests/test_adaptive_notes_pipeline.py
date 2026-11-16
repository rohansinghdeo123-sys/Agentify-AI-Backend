"""Source retrieval, persisted variants and learner positions through real services."""
from copy import deepcopy
from datetime import date
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.security import verify_firebase_user
from database import Base, get_db
from models import AdaptiveNoteLesson, AdminAuditLog, AICoachProfile, ContentChapter, ContentConcept, StudyRouteState, TopicDurationOverride
from services.adaptive_notes_service import compiled_lesson, notes_response, reading_progress, session_portion
from services.notes_config_service import apply_topic_durations, set_topic_duration
from services.notes_material_service import retrieve_material
from services.notes_scope_service import resolve_notes_curriculum, resolve_notes_topics


@pytest.fixture
def db():
    engine = create_engine('sqlite://',poolclass=StaticPool,connect_args={'check_same_thread':False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def curriculum():
    return {'curriculum_key':'published_class10_mathematics_reviewed_quadratics',
        'chapter_slug':'reviewed_quadratics','chapter_title':'Reviewed Quadratic Equations',
        'class_level':'Class 10','subject':'Mathematics','chapter_number':4,
        'source':{'type':'published_chapter','authority':'approved_content_pipeline'},
        'units':[{'id':'quadratic_unit','title':'Quadratic equations','primary_topic_id':'quadratic_definition',
            'estimated_minutes':{'min':30,'max':40},
            'ncert_subtopics':[{'id':'quadratic_definition','title':'Definition','section_id':'4.1'},
                              {'id':'quadratic_formula','title':'Formula','section_id':'4.2'}]}]}


def source_section(subtopic='quadratic_definition',**changes):
    return {'id':subtopic,'title':'Quadratic equations','unit_id':'quadratic_unit',
        'subtopic_ids':[subtopic],'source_pages':[1],'reviewed':True,'complete':True,
        'text':'A quadratic equation has degree two. Its leading coefficient must be nonzero.',
        'teaching_format':'concept',**changes}


def install(db,curriculum,sections):
    chapter, _ = published(db)
    for section in sections:
        concept_id = section['subtopic_ids'][0]
        concept = db.query(ContentConcept).filter_by(chapter_id=chapter.id, concept_id=concept_id).first()
        if not concept:
            concept = ContentConcept(chapter_id=chapter.id, concept_id=concept_id, title=section['title'], source_pages=[1])
            db.add(concept)
        if section.get('teaching_format') == 'numerical':
            concept.examples = [*(concept.examples or []), section['text']]
        else:
            concept.definition = section['text']
    db.commit()


def published(db,*,class_level='10',subject='Mathematics',slug='published_reviewed_quadratics',teaching=True):
    existing = db.query(ContentChapter).filter_by(slug=slug).first()
    if existing:
        return existing, db.query(ContentConcept).filter_by(chapter_id=existing.id).first()
    chapter = ContentChapter(slug=slug,board='NCERT',status='published',class_level=class_level,subject=subject,
        chapter_number=4,chapter_name='Reviewed Quadratic Equations',source_hash='approved-v1',
        published_source_hash='approved-v1',page_count=2,extracted_page_count=2)
    db.add(chapter);db.flush()
    concept = ContentConcept(chapter_id=chapter.id,concept_id=slug+'_definition',title='Quadratic equations',
        definition='A quadratic equation has degree two.' if teaching else '',
        core_explanation='The leading coefficient must be nonzero.' if teaching else '',
        source_pages=[1],key_points=['The highest power of the unknown is two.'],
        properties=['The leading coefficient is nonzero.'],applications=['Quadratic equations model some area problems.'])
    db.add(concept);db.commit()
    return chapter,concept






@pytest.mark.parametrize('class_level,subject',[('10','Mathematics'),('9','Biology'),('12','Physics')])
def test_generic_published_scopes_use_existing_catalog_ids_and_teaching(db,class_level,subject):
    chapter,concept=published(db,class_level=class_level,subject=subject)
    curriculum=resolve_notes_curriculum(db,chapter.slug,subject=subject,class_level='Class '+class_level)
    assert curriculum and curriculum['subject']==subject
    unit=curriculum['units'][0]
    assert unit['ncert_subtopics'][0]['id']==concept.concept_id
    assert resolve_notes_topics(db,curriculum,concept.concept_id)==[unit]
    assert resolve_notes_topics(db,curriculum,'outside-topic')==[]
    assert resolve_notes_curriculum(db,chapter.slug,subject='Wrong subject',class_level=class_level) is None
    assert resolve_notes_curriculum(db,chapter.slug,subject=subject,class_level='Class 8') is None
    material=retrieve_material(db,curriculum,unit)
    assert material['missing_subtopics']==[] and material['sections'][0]['source_type']=='NCERT'
    payload,_=compiled_lesson(db,curriculum,unit,material=material)
    assert payload['pages'] and concept.definition in '\n'.join(p['text'] for p in payload['pages'])
    assert db.query(ContentConcept).count()==1


@pytest.mark.parametrize('wrong_scope',[{'subject':'Physics','class_level':'Class 10'},
                                       {'subject':'Mathematics','class_level':'Class 8'}])
def test_notes_endpoint_legacy_fallback_cannot_escape_explicit_class_subject(db,wrong_scope):
    from fastapi import HTTPException
    from routers.study import get_chapter_notes
    chapter,concept=published(db)
    with pytest.raises(HTTPException) as error:
        get_chapter_notes(chapter.slug,topic_id=concept.concept_id,content_version=None,db=db,
                          current_user={'uid':'learner'},**wrong_scope)
    assert error.value.status_code==404


def test_approved_keypoints_properties_and_applications_are_not_blank_notes(db):
    chapter,concept=published(db,teaching=False)
    curriculum=resolve_notes_curriculum(db,chapter.slug,subject=chapter.subject,class_level=chapter.class_level)
    material=retrieve_material(db,curriculum,curriculum['units'][0])
    assert material['missing_subtopics']==[]
    text='\n'.join(s['text'] for s in material['sections'])
    assert all(value in text for value in [*concept.key_points,*concept.properties,*concept.applications])
    assert compiled_lesson(db,curriculum,curriculum['units'][0],material=material)[0]['pages']


def test_persisted_cache_reuses_variant_and_invalidates_on_source_or_duration_changes(db,curriculum):
    unit=curriculum['units'][0]
    original=[source_section(),source_section('quadratic_formula',text='The discriminant is b² − 4ac.')]
    install(db,curriculum,original)
    first,_=compiled_lesson(db,curriculum,unit);db.commit()
    with patch('services.adaptive_content_compiler.compile_adaptive_lesson',side_effect=AssertionError('Cache missed')):
        cached,_=compiled_lesson(db,curriculum,unit)
    assert cached==first and db.query(AdaptiveNoteLesson).count()==1
    changed=deepcopy(original);changed[0]['text']+=' For example, x² − 4 = 0 is quadratic.'
    install(db,curriculum,changed)
    refreshed,_=compiled_lesson(db,curriculum,unit,requested_version=first['content_version']);db.commit()
    assert refreshed['content_version']!=first['content_version']
    assert refreshed['source_fingerprint']!=first['source_fingerprint']
    set_topic_duration(db,curriculum=curriculum,unit_id=unit['id'],minimum=45,maximum=55,actor_uid='admin');db.commit()
    configured=apply_topic_durations(db,curriculum)
    duration_changed,_=compiled_lesson(db,configured,configured['units'][0],requested_version=refreshed['content_version']);db.commit()
    assert duration_changed['content_version']!=refreshed['content_version']
    assert duration_changed['estimated_minutes']=={'min':45,'max':55}
    assert curriculum['units'][0]['estimated_minutes']=={'min':30,'max':40}
    assert db.query(AdaptiveNoteLesson).count()==3


def test_proficiency_and_style_are_distinct_persisted_variants(db,curriculum):
    install(db,curriculum,[source_section(text='A quadratic equation has degree two. Its leading coefficient must be nonzero.\n\nFor example, x² − 4 = 0 has degree two because its highest power is two.'),
        source_section('quadratic_formula',text='The discriminant is b² − 4ac.')])
    payloads=[compiled_lesson(db,curriculum,curriculum['units'][0],proficiency=p)[0] for p in
              ['new_to_it','know_the_basics','mostly_confident']]
    concise=compiled_lesson(db,curriculum,curriculum['units'][0],style='concise')[0]
    db.commit()
    assert len({p['content_version'] for p in [*payloads,concise]})==4
    assert [p['adaptation']['level'] for p in payloads]==[1,2,3]
    assert all('b² − 4ac' in '\n'.join(page['text'] for page in p['pages']) for p in payloads)
    assert all(p['estimated_minutes']=={'min':30,'max':40} for p in payloads)
    assert db.query(AdaptiveNoteLesson).count()==4


def test_requested_version_cannot_pin_obsolete_proficiency_style_or_compiler(db,curriculum):
    unit=curriculum['units'][0]
    install(db,curriculum,[source_section(),source_section('quadratic_formula',text='The discriminant is b² − 4ac.')])
    first,_=compiled_lesson(db,curriculum,unit);db.commit()
    confident,_=compiled_lesson(db,curriculum,unit,proficiency='mostly_confident',requested_version=first['content_version'])
    detailed,_=compiled_lesson(db,curriculum,unit,style='detailed',requested_version=first['content_version'])
    assert confident['proficiency']=='mostly_confident' and confident['content_version']!=first['content_version']
    assert detailed['explanation_style']=='detailed' and detailed['content_version']!=first['content_version']
    with patch('services.adaptive_content_compiler.COMPILER_VERSION','reviewed_compiler_v_next'):
        upgraded,_=compiled_lesson(db,curriculum,unit,requested_version=first['content_version'])
    assert upgraded['content_version']!=first['content_version']
    assert upgraded['compiler_version']=='reviewed_compiler_v_next'


def test_dynamic_pages_persist_exact_position_completion_and_ignore_stale_navigation(db):
    from services.study_route_service import record_heartbeat,StudyRouteError
    chapter,concept=published(db)
    curriculum=resolve_notes_curriculum(db,chapter.slug,subject=chapter.subject,class_level=chapter.class_level)
    unit=curriculum['units'][0]
    install(db,curriculum,[source_section(id='worked-'+str(i),title='Worked example '+str(i),unit_id=unit['id'],
        subtopic_ids=[concept.concept_id],teaching_format='numerical',
        text=f'Worked example {i}: Solve x² = {i*i}. Taking square roots therefore gives x = ±{i}. Substitution checks each root.') for i in range(1,15)])
    payload,_=compiled_lesson(db,curriculum,unit);db.commit()
    assert len(payload['pages'])>10
    state=StudyRouteState(user_id='learner',route_date=date.today(),class_level=chapter.class_level,subject=chapter.subject,
        active_chapter_slug=chapter.slug,active_curriculum_key=curriculum['curriculum_key'],current_unit_id=unit['id'],
        selections_json=[{'chapter_ref':chapter.slug,'chapter_proficiency':'new_to_it'}],runtime_json={'mastery_evidence':{'unchanged':True}})
    db.add(state);db.commit()
    def position(index,sequence,**changes):
        return {'unit_id':unit['id'],'notes_topic':unit['id'],'notes_page':index,'notes_page_id':payload['pages'][index]['id'],
            'notes_total_pages':len(payload['pages']),'notes_version':payload['content_version'],
            'notes_completed_page_ids':[payload['pages'][index]['id']],'notes_navigation_id':'reader-tab','notes_navigation_seq':sequence,**changes}
    assert record_heartbeat(db,user_id='learner',last_position=position(12,2),compact=True)=={'saved':True}
    db.expire_all();state=db.query(StudyRouteState).one()
    progress=reading_progress(state,unit,payload)
    assert progress['saved_page']==12 and progress['completed_page_ids']==[payload['pages'][12]['id']]
    record_heartbeat(db,user_id='learner',last_position=position(2,1),compact=True)
    progress=reading_progress(state,unit,payload)
    assert progress['saved_page']==12
    assert set(progress['completed_page_ids'])=={payload['pages'][2]['id'],payload['pages'][12]['id']}
    assert state.runtime_json['mastery_evidence']=={'unchanged':True}
    for invalid in [{'notes_page':len(payload['pages'])},{'notes_total_pages':999},{'notes_page_id':'outside-page'},
                    {'notes_completed_page_ids':['another-edition-page']}]:
        with pytest.raises(StudyRouteError):
            record_heartbeat(db,user_id='learner',last_position=position(0,3,**invalid),compact=True)
        db.rollback()
    before=deepcopy(unit['estimated_minutes'])
    portion=session_portion(payload['pages'],unit,start_page=12,session_minutes=5)
    assert portion['start_page']==12 and portion['allocated_minutes']==5
    assert unit['estimated_minutes']==before


def test_daily_session_portion_is_frozen_from_first_open_after_advance_and_reload(db):
    from datetime import timedelta
    from services.study_route_service import start_route,record_heartbeat,_get_state,_today
    chapter,concept=published(db)
    curriculum=resolve_notes_curriculum(db,chapter.slug,subject=chapter.subject,class_level=chapter.class_level)
    unit=curriculum['units'][0]
    install(db,curriculum,[source_section(id='worked-'+str(i),unit_id=unit['id'],subtopic_ids=[concept.concept_id],
        teaching_format='numerical',text=f'Worked example {i}: Solve x² = {i*i}. Therefore x = ±{i}. Squaring either root gives {i*i}.') for i in range(1,15)])
    start_route(db,user_id='learner',class_level='Class 10',subject=chapter.subject,
        selections=[{'chapter_ref':chapter.slug,'chapter_proficiency':'mostly_confident'}],study_time_today='15')
    db.add(AICoachProfile(user_id='learner',coach_id='generic-notes',study_preferences={'preferred_style':'detailed'}));db.commit()
    first=notes_response(db,chapter.slug,unit['id'],'learner',subject=chapter.subject,class_level='Class 10');db.commit()
    assert first['proficiency']=='mostly_confident' and first['explanation_style']=='detailed'
    assert first['session_portion']['start_page']==0
    assert first['can_save_position'] is True and len(first['pages'])>10
    version=first['content_version']
    record_heartbeat(db,user_id='learner',compact=True,last_position={'unit_id':unit['id'],'notes_topic':unit['id'],
        'notes_page':4,'notes_page_id':first['pages'][4]['id'],'notes_version':version,'notes_total_pages':len(first['pages']),
        'notes_completed_page_ids':[first['pages'][3]['id']]})
    db.expire_all()
    reopened=notes_response(db,chapter.slug,unit['id'],'learner',subject=chapter.subject,class_level='Class 10',content_version=version)
    assert reopened['saved_page']==4
    assert reopened['session_portion']==first['session_portion']
    today=_today()
    with patch('services.study_route_service._today',return_value=today+timedelta(days=1)):
        tomorrow=notes_response(db,chapter.slug,unit['id'],'learner',subject=chapter.subject,class_level='Class 10',content_version=version)
    assert tomorrow['saved_page']==4 and tomorrow['session_portion']['start_page']==4
    assert _get_state(db,'learner').runtime_json['notes_session_date']==(today+timedelta(days=1)).isoformat()
    assert tomorrow['completed_page_ids']==[first['pages'][3]['id']]


def test_source_changed_progress_rebases_without_claiming_new_page_completion(db,curriculum):
    unit=curriculum['units'][0]
    install(db,curriculum,[source_section(),source_section('quadratic_formula',text='The discriminant is b² − 4ac.')])
    first,_=compiled_lesson(db,curriculum,unit);db.commit()
    anchor=first['pages'][-1]
    state=StudyRouteState(user_id='learner',route_date=date.today(),runtime_json={
        'notes_reading_by_version':{first['content_version']:{unit['id']:{'page_id':anchor['id'],'completed_page_ids':[anchor['id']]}}},
        'notes_anchor_by_unit':{unit['id']:{'content_anchor':anchor['content_anchor'],'source_ids':[r['id'] for r in anchor['source_refs']]}}})
    install(db,curriculum,[source_section(text=source_section()['text']+' A missing square term is not quadratic.'),
                           source_section('quadratic_formula',text='The discriminant is b² − 4ac.')])
    current,_=compiled_lesson(db,curriculum,unit)
    progress=reading_progress(state,unit,current)
    assert progress['progress_rebased'] is True and progress['completed_page_ids']==[]
    assert current['pages'][progress['saved_page']]['content_anchor']==anchor['content_anchor']


def test_legacy_v1_bookmark_rebases_to_its_mapped_subtopic_without_false_completion(db):
    from Logic.planning.curriculum_registry import load_planning_curriculum
    from services.teacher_notes_service import _load_lesson
    curriculum=load_planning_curriculum();unit=curriculum['units'][1]
    legacy=_load_lesson(unit['id']);old_index=8
    old_subtopics=set(legacy['pages'][old_index]['source_section_ids'])
    current,_=compiled_lesson(db,curriculum,unit)
    state=StudyRouteState(user_id='legacy-learner',route_date=date.today(),runtime_json={
        'notes_positions_by_version':{'ncert_teacher_v1':{unit['id']:old_index}},
        'notes_positions':{unit['id']:31},'mastery_evidence':{'unchanged':True}})
    progress=reading_progress(state,unit,current)
    expected=next(i for i,p in enumerate(current['pages']) if old_subtopics.intersection(p['source_section_ids']))
    assert progress['saved_page']==expected and progress['progress_rebased'] is True
    assert progress['completed_page_ids']==[]
    assert state.runtime_json['notes_positions_by_version']['ncert_teacher_v1'][unit['id']]==old_index
    assert state.runtime_json['notes_positions'][unit['id']]==31
    assert state.runtime_json['mastery_evidence']=={'unchanged':True}


def test_source_identity_or_subtopic_anchor_survives_changed_page_layout(db,curriculum):
    unit=curriculum['units'][0]
    install(db,curriculum,[source_section(),source_section('quadratic_formula',text='The discriminant is b² − 4ac.')])
    current,_=compiled_lesson(db,curriculum,unit)
    last=current['pages'][-1]
    for anchor in [{'content_anchor':'old-layout','source_ids':[r['id'] for r in last['source_refs']]},
                   {'content_anchor':'old-layout','source_ids':['removed-source'],'subtopic_ids':['quadratic_formula']}]:
        state=StudyRouteState(user_id='learner',route_date=date.today(),runtime_json={'notes_anchor_by_unit':{unit['id']:anchor}})
        progress=reading_progress(state,unit,current)
        assert progress['progress_rebased'] is True and progress['completed_page_ids']==[]
        assert 'quadratic_formula' in current['pages'][progress['saved_page']]['source_section_ids']


def test_duration_override_requires_admin_and_persists_audited_existing_scope(db):
    from routers.notes_admin import router
    chapter,_=published(db)
    curriculum=resolve_notes_curriculum(db,chapter.slug,subject=chapter.subject,class_level=chapter.class_level)
    unit=curriculum['units'][0]
    params={'chapter_ref':chapter.slug,'subject':chapter.subject,'class_level':'Class 10'}
    payload={**params,'unit_id':unit['id'],'minimum_minutes':45,'maximum_minutes':55}
    token={'uid':'student','email':'student@example.test'}
    app=FastAPI();app.include_router(router)
    app.dependency_overrides[get_db]=lambda:db
    app.dependency_overrides[verify_firebase_user]=lambda:token
    with TestClient(app) as client:
        assert client.put('/admin/notes/topic-duration',json=payload).status_code==404
        assert client.get('/admin/notes/topic-durations',params=params).status_code==404
        assert db.query(TopicDurationOverride).count()==0
        token.update(uid='admin',admin=True)
        response=client.put('/admin/notes/topic-duration',json=payload)
        assert response.status_code==200,response.text
        with Session(db.bind) as independent:
            persisted=independent.query(TopicDurationOverride).one()
            assert (persisted.minimum_minutes,persisted.maximum_minutes,persisted.configured_by)==(45,55,'admin')
        assert db.query(AdminAuditLog).one().action=='notes_topic_duration'
        assert client.get('/admin/notes/topic-durations',params=params).json()['topics'][0]['estimated_minutes']=={'min':45,'max':55}
        assert client.put('/admin/notes/topic-duration',json={**payload,'unit_id':'wrong-unit'}).status_code==400
        assert client.put('/admin/notes/topic-duration',json={**payload,'minimum_minutes':56}).status_code==400
        assert client.put('/admin/notes/topic-duration',json={**payload,'minimum_minutes':True}).status_code==422
        assert db.query(AdminAuditLog).count()==1


def test_exact_six_chapter_one_duration_ranges_remain_authoritative(db):
    from Logic.planning.curriculum_registry import load_planning_curriculum
    curriculum=load_planning_curriculum()
    assert [(u['estimated_minutes']['min'],u['estimated_minutes']['max']) for u in curriculum['units']]==[
        (30,35),(45,50),(45,50),(40,45),(60,75),(75,90)]
    for unit in curriculum['units']:
        with pytest.raises(ValueError,match='must remain unchanged'):
            set_topic_duration(db,curriculum=curriculum,unit_id=unit['id'],minimum=5,maximum=6,actor_uid='admin')
    assert db.query(TopicDurationOverride).count()==0


def test_notes_migration_creates_usable_unique_cache_and_duration_tables_idempotently():
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect
    from sqlalchemy.exc import IntegrityError
    filename=Path(__file__).resolve().parents[1]/'migrations/versions/20261108_0018_adaptive_notes.py'
    spec=importlib.util.spec_from_file_location('adaptive_notes_migration',filename)
    migration=importlib.util.module_from_spec(spec);spec.loader.exec_module(migration)
    engine=create_engine('sqlite://')
    with engine.begin() as connection:
        migration.op=Operations(MigrationContext.configure(connection))
        migration.upgrade();migration.upgrade()
        assert set(inspect(connection).get_table_names())=={'adaptive_note_lessons','topic_duration_overrides'}
    with Session(engine) as session:
        session.add(TopicDurationOverride(curriculum_key='generic',unit_id='unit',minimum_minutes=30,maximum_minutes=40,configured_by='admin'))
        session.add(AdaptiveNoteLesson(content_version='version',curriculum_key='generic',chapter_slug='chapter',unit_id='unit',
            proficiency='new_to_it',explanation_style='balanced',source_fingerprint='a'*64,minimum_minutes=30,maximum_minutes=40,payload_json={'pages':[]}))
        session.commit()
        session.add(TopicDurationOverride(curriculum_key='generic',unit_id='unit',minimum_minutes=20,maximum_minutes=30,configured_by='admin'))
        with pytest.raises(IntegrityError):session.commit()
        session.rollback()
        assert session.query(TopicDurationOverride).one().maximum_minutes==40
        assert session.query(AdaptiveNoteLesson).one().payload_json=={'pages':[]}
    with engine.begin() as connection:
        migration.op=Operations(MigrationContext.configure(connection));migration.downgrade()
        assert inspect(connection).get_table_names()==[]
    engine.dispose()



def test_locked_route_refresh_preserves_other_session_bookmark_changes(db):
    from services.study_route_service import _get_state
    state=StudyRouteState(user_id='concurrent-reader',route_date=date.today(),runtime_json={'old':True})
    db.add(state);db.commit()
    cached=_get_state(db,'concurrent-reader')
    with Session(db.bind) as other:
        row=other.query(StudyRouteState).filter_by(user_id='concurrent-reader').one()
        row.runtime_json={'notes_reading_by_version':{'edition':{'unit':{'completed_page_ids':['p1']}}}}
        other.commit()
    latest=_get_state(db,'concurrent-reader',lock=True)
    assert latest is cached
    assert latest.runtime_json['notes_reading_by_version']['edition']['unit']['completed_page_ids']==['p1']
