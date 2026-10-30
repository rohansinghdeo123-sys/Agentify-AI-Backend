"""Reproducible local benchmark; synthetic data, no provider or production DB.
Run: python scripts/benchmark_latency.py --baseline-ref 720af30
"""
import argparse,json,statistics,subprocess,sys,time
from datetime import date,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine,event
from sqlalchemy.orm import Session
from database import Base
from models import UserProgress,UserProfile,TestHistory,TopicPerformance
from Logic.analytics_engine import get_user_analytics
from services.leaderboard_service import build_leaderboard
from services.study_route_service import start_route,record_heartbeat

p=argparse.ArgumentParser();p.add_argument('--baseline-ref',default='720af30');args=p.parse_args()
def previous(path,name):
 source=subprocess.check_output(['git','show',f'{args.baseline_ref}:{path}'],text=True,encoding='utf-8')
 namespace={'__name__':'latency_baseline'};exec(compile(source,path,'exec'),namespace);return namespace[name]
old_analytics=previous('Logic/analytics_engine.py','get_user_analytics')
old_board=previous('services/leaderboard_service.py','build_leaderboard')
engine=create_engine('sqlite://');Base.metadata.create_all(engine)
with Session(engine) as db:
 db.add_all([UserProgress(user_id=f'student-{i}',xp=100 if i<100 else 0,total_tests=10 if i<100 else 0) for i in range(3000)])
 db.add_all([UserProfile(user_id=f'student-{i}',display_name=f'Student {i}',class_level='Class 11') for i in range(3000)])
 db.add_all([TestHistory(user_id='student-0',date=date.today(),topic=f'topic-{i%40}',score=4,total_questions=5,accuracy_rate=80,focus_score=80,time_spent_seconds=30) for i in range(2000)])
 db.add_all([TopicPerformance(user_id='student-0',topic=f'topic-{i}',attempts=50,correct=40,last_practiced=datetime.now()) for i in range(40)])
 db.commit()
count=[0]
@event.listens_for(engine,'before_cursor_execute')
def query_count(conn,cursor,statement,params,context,many):
 if statement.lstrip().upper().startswith('SELECT'):count[0]+=1

def measure(fn):
 elapsed=[];queries=[]
 for _ in range(12):
  with Session(engine) as db:
   count[0]=0;t=time.perf_counter();fn(db);elapsed.append((time.perf_counter()-t)*1000);queries.append(count[0])
 return {'median_ms':round(statistics.median(elapsed),3),'selects':int(statistics.median(queries))}
report={'environment':'Local SQLite, synthetic 3000 accounts / 100 active, 2000 sessions, 40 topics; median of 12 requests, same indexed schema for both versions.',
 'analytics_before':measure(lambda db:old_analytics(db,'student-0',persist_metrics=False)),
 'analytics_after':measure(lambda db:get_user_analytics(db,'student-0',persist_metrics=False)),
 'analytics_persisting_before':measure(lambda db:old_analytics(db,'student-0')),
 'analytics_persisting_after':measure(lambda db:get_user_analytics(db,'student-0')),
 'leaderboard_before':measure(lambda db:old_board(db,{'uid':'student-0'})),
 'leaderboard_after':measure(lambda db:build_leaderboard(db,{'uid':'student-0'}))}
with Session(engine) as db:
 route=start_route(db,user_id='reader',class_level='Class 11',subject='Chemistry',selections=[{'chapter_ref':'some_basic_concepts_of_chemistry','chapter_proficiency':'new_to_it'}])
report['heartbeat_before']=measure(lambda db:record_heartbeat(db,user_id='reader',last_position={'message_count':3}))
report['heartbeat_after']=measure(lambda db:record_heartbeat(db,user_id='reader',last_position={'message_count':3},compact=True))
with Session(engine) as db:
 full=record_heartbeat(db,user_id='reader',last_position={'message_count':3})
 compact=record_heartbeat(db,user_id='reader',last_position={'message_count':3},compact=True)
 report['heartbeat_payload_bytes']={'full':len(json.dumps(full,default=str,separators=(',',':')).encode()),'compact':len(json.dumps(compact,separators=(',',':')).encode())}
print(json.dumps(report,indent=2))
