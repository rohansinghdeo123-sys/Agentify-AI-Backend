"""End-to-end service tests with real SQLite persistence and mastery derivation."""
from datetime import timedelta
from unittest.mock import patch

import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base
from models import StudyRouteState, TestHistory as History, PlanningLearningEvent, TopicPerformance, UserProgress
from schemas import StudyRouteStateResponse, StudyLabContextResponse, StudyRouteAdvanceResponse
from services import study_route_service as service



def start(db, **kwargs):
    return service.start_route(db, user_id='route-student', class_level='Class 11', subject='Chemistry',
        selections=[{'chapter_ref': 'some_basic_concepts_of_chemistry', 'chapter_proficiency': 'new_to_it'}],
        study_time_today='120_plus', **kwargs)


def check(db, correct=True):
    def generate(**kwargs):
        return {'questions': [{'id': f'Q{i}', 'question': f'Question {i}',
            'options': ['A. one', 'B. two', 'C. three', 'D. four'], 'correct': 'B',
            'explanation': 'The approved explanation.'} for i in range(kwargs['count'])]}
    with patch('Logic.section_doubt.generate_structured_mcqs', side_effect=generate), patch('app.security.enforce_user_quota'):
        pack = service.prepare_check(db, user_id='route-student')
    answers = {q['id']: 'B' if correct else 'A' for q in pack['questions']}
    return pack, answers


def advance(db, pack, answers):
    return service.advance_route(db, user_id='route-student', completed_unit_id=pack['unit_id'],
        interaction_id=pack['id'], answers=answers)


class StudyRouteIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_start_notes_resume_refresh_and_next_day(self):
        db = self.db
        first = start(db)
        StudyRouteStateResponse.model_validate(first)
        context = service.build_study_lab_context(db, user_id='route-student')
        StudyLabContextResponse.model_validate(context)
        assert context['key_notes']
        uid = first['current_step']['current_unit_id']
        service.record_heartbeat(db, user_id='route-student', last_position={'unit_id': uid, 'conversation_id': 'saved-conversation'})
        pack, _ = check(db)
        with patch('Logic.section_doubt.generate_structured_mcqs', side_effect=AssertionError('must not regenerate')):
            assert service.prepare_check(db, user_id='route-student')['id'] == pack['id']
        db.expire_all()
        with patch.object(service, '_today', return_value=service._today() + timedelta(days=1)):
            resumed = start(db)
        assert resumed['current_step']['last_position']['conversation_id'] == 'saved-conversation'
        assert resumed['today_route'] == first['today_route']
        assert db.query(StudyRouteState).count() == 1
        assert all('correct' not in q for q in pack['questions'])


    def test_strong_completion_advances_and_is_idempotent(self):
        db = self.db
        original = start(db)
        pack, answers = check(db)
        # A UserProgress row used to make analytics commit prematurely.
        db.add(UserProgress(user_id='route-student'))
        db.commit()
        result = advance(db, pack, answers)
        StudyRouteAdvanceResponse.model_validate(result)
        assert result['completion']['mastered']
        assert result['route']['current_step']['current_unit_id'] != pack['unit_id']
        assert result['route']['completed_unit_ids'] == [pack['unit_id']]
        assert result['route']['today_route'] == original['today_route']
        assert result['route']['aggregate_progress']['mastered_units'] == 1
        proficiency = next(iter(result['route']['effective_proficiency'].values()))
        assert proficiency['effective'] == 'mostly_confident'
        assert proficiency['self_reported'] == 'new_to_it'
        replay = advance(db, pack, answers)
        assert replay['completion'] == result['completion']
        assert db.query(History).count() == 1
        assert db.query(PlanningLearningEvent).count() == 1
        assert db.query(TopicPerformance).first().attempts == len(answers)


    def test_weak_performance_stays_in_review_and_does_not_claim_completion(self):
        db = self.db
        first = start(db)
        pack, answers = check(db, False)
        result = advance(db, pack, answers)
        assert not result['completion']['mastered']
        assert result['route']['current_step']['current_unit_id'] == first['current_step']['current_unit_id']
        assert result['route']['current_step']['unit_status'] == 'needs_review'
        assert result['route']['needs_review_concept_ids']
        assert not result['next_action']['done_for_today']
        assert not result['route']['completed_unit_ids']
        paused = service.complete_today(db, user_id='route-student')
        assert paused['status'] == 'paused'
        assert paused['aggregate_progress']['mastered_units'] == 0


    def test_prerequisite_branch_and_return_to_original_unit(self):
        db = self.db
        start(db)
        pack, answers = check(db)
        mastered = advance(db, pack, answers)
        target = pack['unit_id']
        original_unit = mastered['route']['current_step']['current_unit_id']
        assert original_unit != target
        service.record_heartbeat(db, user_id='route-student', last_position={'unit_id': original_unit, 'conversation_id': 'original-B-session'})
        pack, answers = check(db, False)
        weak = advance(db, pack, answers)
        assert weak['route']['override']['active']
        assert weak['route']['override']['return_unit_id'] == original_unit
        assert weak['route']['current_step']['current_unit_id'] == target
        assert weak['route']['override']['explanation']
        review_pack, review_answers = check(db)
        returned = advance(db, review_pack, review_answers)
        assert not returned['route']['override']['active']
        assert returned['route']['current_step']['current_unit_id'] == original_unit
        assert returned['route']['current_step']['last_position']['conversation_id'] == 'original-B-session'
        assert returned['route']['status'] == 'in_progress'


    def test_rejects_forged_scores_skips_invalid_branches_and_stale_heartbeats(self):
        db = self.db
        route = start(db)
        chapter = route['chapters'][0]
        last = chapter['learning_units'][-1]['id']
        with self.assertRaises(service.StudyRouteError):
            service.advance_route(db, user_id='route-student', completed_unit_id=last,
                interaction_id='forged', correct_answers=99, total_questions=99)
        with self.assertRaises(service.StudyRouteError):
            service.branch_route(db, user_id='route-student', reason='prerequisite_gap', target_unit_id=last)
        with self.assertRaises(service.StudyRouteError):
            service.record_heartbeat(db, user_id='route-student', last_position={'unit_id': last})
        assert db.query(History).count() == 0


    def test_assessment_transaction_rolls_back_all_evidence_on_failure(self):
        db = self.db
        start(db)
        pack, answers = check(db)
        with patch.object(service, '_build_portfolio_from_state', side_effect=[
            service._build_portfolio_from_state(db, service._get_state(db, 'route-student')),
            RuntimeError('simulated routing failure')]):
            with self.assertRaises(RuntimeError):
                advance(db, pack, answers)
        db.rollback()
        assert db.query(History).count() == 0
        assert db.query(PlanningLearningEvent).count() == 0
        assert db.query(TopicPerformance).count() == 0
        assert service._get_state(db, 'route-student').runtime_json['check']['id'] == pack['id']

    def test_authenticated_api_context_and_cross_user_isolation(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app import config
        from app.security import verify_firebase_user
        from database import get_db
        from routers.planning import router
        from routers.coach import router as coach_router
        start(self.db)
        app = FastAPI()
        app.include_router(router)
        app.include_router(coach_router)
        app.dependency_overrides[get_db] = lambda: self.db
        app.dependency_overrides[verify_firebase_user] = lambda: {'uid': 'route-student'}
        with patch.object(config, 'STUDY_LAB_PLANNING_INTEGRATION', True):
            client = TestClient(app)
            response = client.get('/coach/study-lab/context/route-student')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['key_notes'])
            self.assertEqual(client.get('/planning/route/current/another-student').status_code, 403)
            rejected = client.post('/planning/route/advance/route-student', json={
                'completed_unit_id': response.json()['current_unit_id'],
                'interaction_id': 'unverified', 'correct_answers': 1, 'total_questions': 1})
            self.assertEqual(rejected.status_code, 422)

    def test_runtime_migration_preserves_existing_route_and_is_repeatable(self):
        import importlib.util
        from pathlib import Path
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from sqlalchemy import text, inspect
        start(self.db)
        self.db.close()
        with self.engine.begin() as conn:
            conn.execute(text('ALTER TABLE study_route_states DROP COLUMN runtime_json'))
            file = Path(__file__).parents[1] / 'migrations/versions/20261030_0016_route_runtime.py'
            spec = importlib.util.spec_from_file_location('route_runtime_migration', file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with Operations.context(MigrationContext.configure(conn)):
                module.upgrade()
                module.upgrade()
            self.assertIn('runtime_json', {c['name'] for c in inspect(conn).get_columns('study_route_states')})
            self.assertEqual(conn.execute(text('SELECT user_id FROM study_route_states')).scalar(), 'route-student')
        restored = service.read_current_route(self.db, user_id='route-student')
        self.assertTrue(restored['today_route']['items'])

    def test_latest_weak_evidence_survives_recent_session_window(self):
        from Logic.analytics_engine import update_topic_performance
        route = start(self.db)
        topic = route['chapters'][0]['learning_units'][0]['primary_topic_id']
        self.db.add(History(user_id='route-student', topic=topic, date=service._today(),
            score=100, total_questions=100, accuracy_rate=100, session_type='practice'))
        update_topic_performance(self.db, 'route-student', topic, 100, 100)
        pack, answers = check(self.db, False)
        result = advance(self.db, pack, answers)
        self.assertEqual(result['route']['current_step']['unit_status'], 'needs_review')
        for index in range(12):
            self.db.add(History(user_id='route-student', topic=f'unrelated-{index}', date=service._today(),
                score=1, total_questions=1, accuracy_rate=100, session_type='practice'))
        self.db.commit()
        restored = service.read_current_route(self.db, user_id='route-student')
        self.assertEqual(restored['current_step']['unit_status'], 'needs_review')
        self.assertIn(topic, restored['needs_review_concept_ids'])
