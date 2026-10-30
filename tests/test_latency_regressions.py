import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import Base
from services.ttl_cache import TTLCache
from tests.test_study_route_integration import start
from services.study_route_service import record_heartbeat

class LatencyRegressionTests(unittest.TestCase):
 def test_parallel_identical_cache_misses_build_once(self):
  cache=TTLCache(); barrier=threading.Barrier(8); calls=[]
  def run():
   barrier.wait(timeout=3)
   return cache.get_or_build('same',30,lambda:calls.append(1) or {'ok':True})
  with ThreadPoolExecutor(max_workers=8) as pool:
   values=list(pool.map(lambda _:run(),range(8)))
  self.assertEqual(len(calls),1);self.assertEqual(values,[{'ok':True}]*8)
 def test_invalidation_during_build_does_not_cache_stale_value(self):
  cache=TTLCache(); begun=threading.Event(); release=threading.Event()
  def slow():begun.set();release.wait(timeout=3);return 'old'
  with ThreadPoolExecutor(max_workers=1) as pool:
   future=pool.submit(cache.get_or_build,'key',30,slow)
   self.assertTrue(begun.wait(timeout=2));cache.invalidate('key');release.set()
   self.assertEqual(future.result(timeout=2),'old')
  self.assertEqual(cache.get_or_build('key',30,lambda:'new'),'new')
 def test_failed_build_can_retry(self):
  cache=TTLCache()
  with self.assertRaises(ValueError):cache.get_or_build('key',30,lambda:(_ for _ in []).throw(ValueError()))
  self.assertEqual(cache.get_or_build('key',30,lambda:'good'),'good')
 def test_compact_heartbeat_persists_without_portfolio_rebuild(self):
  engine=create_engine('sqlite://');Base.metadata.create_all(engine)
  with Session(engine) as db:
   route=start(db)
   with patch('services.study_route_service.read_current_route',side_effect=AssertionError('unneeded portfolio read')):
    self.assertEqual(record_heartbeat(db,user_id='route-student',last_position={'unit_id':route['current_step']['current_unit_id'],'message_count':5},compact=True),{'saved':True})
   from models import StudyRouteState
   self.assertEqual(db.query(StudyRouteState).first().last_position_json['message_count'],5)
  engine.dispose()
 def test_stream_preflight_is_not_an_async_blocking_handler(self):
  import inspect
  from routers.coach import coach_chat_stream
  self.assertFalse(inspect.iscoroutinefunction(coach_chat_stream))

 def test_analytics_serialization_does_not_reload_rows_after_commit(self):
  from datetime import date
  from sqlalchemy import event
  from models import TestHistory,TopicPerformance,UserProgress
  from Logic.analytics_engine import get_user_analytics
  engine=create_engine('sqlite://');Base.metadata.create_all(engine)
  with Session(engine) as db:
   db.add(UserProgress(user_id='metrics'))
   db.add_all([TestHistory(user_id='metrics',date=date.today(),topic=f'topic-{i}',score=4,total_questions=5,accuracy_rate=80,focus_score=80,time_spent_seconds=60) for i in range(20)])
   db.add_all([TopicPerformance(user_id='metrics',topic=f'topic-{i}',attempts=5,correct=4) for i in range(20)])
   db.commit()
   selects=[]
   def count(conn,cursor,sql,params,context,many):
    if sql.lstrip().upper().startswith('SELECT'):selects.append(sql)
   event.listen(engine,'before_cursor_execute',count)
   payload=get_user_analytics(db,'metrics')
   self.assertEqual(payload['cognitive_metrics']['focus_score'],80)
   self.assertLessEqual(len(selects),4)
   event.remove(engine,'before_cursor_execute',count)
   self.assertEqual(db.query(UserProgress).first().focus_score,80)
  engine.dispose()

 def test_compact_dashboard_keeps_metrics_without_global_board(self):
  from datetime import date
  from models import TestHistory,UserProgress
  from routers.progress import get_dashboard_data
  engine=create_engine('sqlite://');Base.metadata.create_all(engine)
  with Session(engine) as db:
   db.add(UserProgress(user_id='dashboard'))
   db.add(TestHistory(user_id='dashboard',date=date.today(),topic='matter',score=4,total_questions=5,accuracy_rate=80,focus_score=80,time_spent_seconds=60))
   db.commit()
   with patch('routers.progress.build_leaderboard',side_effect=AssertionError('unused board')),patch('routers.progress.get_user_analytics',side_effect=AssertionError('unused evidence')):
    result=get_dashboard_data('dashboard',include_analytics=False,include_leaderboard=False,db=db,current_user={'uid':'dashboard'})
   self.assertEqual(result['progress']['focus_score'],80)
   self.assertGreater(result['progress']['consistency_index'],0)
   self.assertIsNone(result['leaderboard'])
   self.assertEqual(len(result['sessions']),1)
  engine.dispose()

 def test_gzip_preserves_unbuffered_sse(self):
  from fastapi import FastAPI
  from fastapi.testclient import TestClient
  from starlette.middleware.gzip import GZipMiddleware
  from starlette.responses import StreamingResponse
  app=FastAPI();app.add_middleware(GZipMiddleware,minimum_size=1024,compresslevel=5)
  @app.get('/json')
  def data():return {'notes':'source material '*500}
  @app.get('/stream')
  def stream():return StreamingResponse(iter(['data: '+('text '*500)+'\n\n']),media_type='text/event-stream')
  client=TestClient(app)
  self.assertEqual(client.get('/json').headers.get('content-encoding'),'gzip')
  self.assertNotIn('content-encoding',client.get('/stream').headers)

 def test_query_embedding_parallel_misses_share_provider_call(self):
  import os
  from Logic import embeddings
  embeddings.clear_query_cache();barrier=threading.Barrier(8)
  def run():barrier.wait(timeout=3);return embeddings.embed_query('What is matter?')
  with patch.dict(os.environ,{'EMBEDDINGS_API_KEY':'synthetic-test-key'}),patch('Logic.embeddings.embed_texts',return_value=[[1.0,0.0]]) as embed:
   with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(lambda _:run(),range(8)))
   self.assertEqual(embed.call_count,1);self.assertEqual(values,[[1.0,0.0]]*8)
  embeddings.clear_query_cache()

 def test_index_migration_is_repeatable_and_preserves_history(self):
  import importlib.util
  from pathlib import Path
  from alembic.migration import MigrationContext
  from alembic.operations import Operations
  from sqlalchemy import text,inspect
  engine=create_engine('sqlite://');Base.metadata.create_all(engine)
  file=Path(__file__).parents[1]/'migrations/versions/20261030_0017_latency_indexes.py'
  spec=importlib.util.spec_from_file_location('latency_indexes',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  with engine.begin() as conn:
   conn.execute(text("INSERT INTO test_history (user_id, topic) VALUES ('preserved', 'matter')"))
   for name,_,_ in module.INDEXES:conn.execute(text(f'DROP INDEX {name}'))
   with Operations.context(MigrationContext.configure(conn)):
    module.upgrade();module.upgrade()
   for name,table,_ in module.INDEXES:self.assertIn(name,{i['name'] for i in inspect(conn).get_indexes(table)})
   self.assertEqual(conn.execute(text('SELECT user_id FROM test_history')).scalar(),'preserved')
   plan=str(conn.execute(text("EXPLAIN QUERY PLAN SELECT MAX(id) FROM test_history WHERE user_id='preserved' GROUP BY topic")).all())
   self.assertIn('ix_test_history_user_topic_id',plan)
  engine.dispose()

 def test_exam_upload_processing_does_not_block_event_loop(self):
  import asyncio
  from types import SimpleNamespace
  from unittest.mock import AsyncMock
  from routers.exam_papers import upload_paper
  began=threading.Event();release=threading.Event()
  def slow(*args,**kwargs):
   began.set();release.wait(timeout=2)
   return SimpleNamespace(parse_status='analyzed',analysis_json={},extracted_question_count=1,warnings_json=[])
  async def exercise():
   task=asyncio.create_task(upload_paper(file=SimpleNamespace(filename='test.pdf',content_type='application/pdf'),db=object(),current_user={'uid':'u'}))
   await asyncio.to_thread(began.wait,1)
   blocked=task.done()
   release.set()
   result=await task
   self.assertFalse(blocked,'The event loop could not resume during synchronous paper analysis')
   self.assertEqual(result['questions_extracted'],1)
  with patch('routers.exam_papers.enforce_user_quota'),patch('routers.exam_papers._read_capped',new=AsyncMock(return_value=b'%PDF')),patch('routers.exam_papers.paper_service.create_paper_from_upload',side_effect=slow),patch('routers.exam_papers.paper_service.serialize_paper',return_value={}):
   asyncio.run(exercise())

 def test_nullable_recent_history_does_not_hide_older_evidence(self):
  from datetime import date
  from models import TestHistory
  from Logic.analytics_engine import get_user_analytics
  engine=create_engine('sqlite://');Base.metadata.create_all(engine)
  with Session(engine) as db:
   db.add(TestHistory(user_id='nullable',topic='older-weak',date=date.today(),total_questions=5,score=0,accuracy_rate=0))
   db.flush()
   db.add_all([TestHistory(user_id='nullable',topic=None,date=date.today(),total_questions=1,score=1,accuracy_rate=100) for _ in range(12)])
   db.commit()
   payload=get_user_analytics(db,'nullable',persist_metrics=False)
   self.assertIn('older-weak',[row['topic'] for row in payload['topic_evidence']])
  engine.dispose()
