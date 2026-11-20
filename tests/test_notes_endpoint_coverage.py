"""Every released Chemistry unit must survive real HTTP serialization and resume."""
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.security import verify_firebase_user
from database import Base, get_db
from models import TestHistory as History
from routers.planning import router as planning_router
from routers.study import router as study_router
from scripts.audit_notes_endpoints import audit
from services.adaptive_content_compiler import PROFICIENCIES
from services.study_route_service import start_route
from services.teacher_notes_service import teacher_curricula


def test_every_released_chemistry_unit_opens_and_resumes_through_actual_endpoints():
    report=audit(proficiencies=('new_to_it',),styles=('balanced',),aliases=False)
    assert report['errors']==[],report['errors']
    assert report['summary']['chapters']==9
    assert report['summary']['units']==72
    assert report['summary']['variants']==72
    assert report['summary']['progress_roundtrips']==72
    assert report['summary']['subtopic_bookmark_checks']==report['summary']['subtopics']==129
    assert report['summary']['subtopic_tutor_checks']==129
    page_counts={count for chapter in report['chapters'] for count in chapter['page_counts']}
    assert len(page_counts)>1 and max(page_counts)>10,'Pages were forced into a fixed ten-page layout'
    assert all(chapter['subtopics']>0 and chapter['variants']==chapter['units'] for chapter in report['chapters'])


def test_timed_notes_continue_past_first_topic_and_save_next_topic_at_expiry():
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    curriculum = teacher_curricula()[0]
    first, second, third = [u['id'] for u in curriculum['units'][:3]]
    path = '/notes/' + curriculum['chapter_slug']
    app = FastAPI()
    app.include_router(study_router)
    app.include_router(planning_router)
    clock = datetime(2026, 11, 16, 10)
    with Session(engine, autoflush=False) as db, TestClient(app) as client, patch('app.config.STUDY_LAB_PLANNING_INTEGRATION', True), \
         patch('services.study_route_service._now', side_effect=lambda: clock):
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[verify_firebase_user] = lambda: {'uid': 'timed-reader'}
        start_route(db, user_id='timed-reader', class_level='Class 11', subject='Chemistry',
                    selections=[{'chapter_ref': curriculum['chapter_slug'], 'chapter_proficiency': 'know_a_little'}],
                    study_time_today='15')

        def open_topic(topic, restart=False):
            response = client.get(path, params={'topic_id': topic, 'planned': True, 'start_session': restart})
            assert response.status_code == 200, response.text[:300]
            return response.json()

        def finish(notes, *, full=True):
            pages = notes['pages']
            return client.post('/planning/route/heartbeat/timed-reader?compact=true', json={'last_position': {
                'unit_id': notes['topic_id'], 'notes_topic': notes['topic_id'], 'notes_version': notes['content_version'],
                'notes_page': len(pages) - 1, 'notes_page_id': pages[-1]['id'], 'notes_total_pages': len(pages),
                'notes_completed_page_ids': [p['id'] for p in pages] if full else [], 'finish_notes': True,
            }})

        notes = open_topic(first, restart=True)
        assert notes['study_session']['remaining_seconds'] == 900
        assert notes['source_types'] == ['NCERT'] and notes['proficiency'] == 'new_to_it'
        assert finish(notes, full=False).status_code != 200
        db.rollback()
        clock += timedelta(minutes=6)
        assert finish(notes).json()['next_notes']['topic_id'] == second
        assert finish(notes).json()['next_notes']['topic_id'] == second
        notes = open_topic(second)
        assert notes['study_session']['remaining_seconds'] == 540
        clock += timedelta(minutes=10)
        assert finish(notes).json() == {'saved': True, 'notes_outcome': 'time_up'}
        route = client.get('/planning/route/current/timed-reader').json()
        assert route['current_step']['current_unit_id'] == third
        assert route['read_unit_ids'] == [first, second]
        assert open_topic(third)['study_session']['remaining_seconds'] == 0
        assert open_topic(third, restart=True)['study_session']['remaining_seconds'] == 900
        assert db.query(History).count() == 0
    engine.dispose()


def test_switching_notes_mode_honors_selection_even_with_high_assessment_scores():
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    curriculum = teacher_curricula()[0]
    unit = curriculum['units'][0]
    app = FastAPI()
    app.include_router(study_router)
    learner = 'changing-mode'
    evidence = {'topic_heatmap': [{'topic': unit['primary_topic_id'], 'attempts': 4, 'accuracy': 100}]}
    with Session(engine, autoflush=False) as db, TestClient(app) as client, \
         patch('Logic.planning.portfolio_engine.get_user_analytics', return_value=evidence):
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[verify_firebase_user] = lambda: {'uid': learner}
        versions, teaching_views = set(), set()
        for mode in reversed(PROFICIENCIES):
            route = start_route(db, user_id=learner, class_level='Class 11', subject='Chemistry',
                                selections=[{'chapter_ref': curriculum['chapter_slug'], 'chapter_proficiency': mode}])
            assert route['chapters'][0]['effective_proficiency']['effective'] == 'mostly_confident'
            response = client.get('/notes/' + curriculum['chapter_slug'], params={'topic_id': unit['id']})
            assert response.status_code == 200, response.text[:300]
            notes = response.json()
            assert notes['proficiency'] == mode
            assert notes['pages'] and notes['source_types'] == ['NCERT']
            versions.add(notes['content_version'])
            teaching_views.add(tuple(block['text'] for page in notes['pages'] for block in page['blocks']))
        assert len(versions) == len(teaching_views) == 3
    engine.dispose()
