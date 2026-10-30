import gzip,hashlib,json,unittest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import Base
from services.topic_notes_service import mapped_notes, chapter_one_mapping
from services.study_route_service import start_route,record_heartbeat
from Logic.planning.curriculum_registry import resolve_planning_curriculum
from Logic.planning.recommendation_engine import build_planning_roadmap

class TopicNotesTests(unittest.TestCase):
 def setUp(self):
  self.engine=create_engine('sqlite://');Base.metadata.create_all(self.engine);self.db=Session(self.engine)
  self.c=resolve_planning_curriculum(chapter_ref='some_basic_concepts_of_chemistry',subject='Chemistry',class_level='Class 11')
 def tearDown(self):self.db.close();self.engine.dispose()
 def test_manual_ranges_and_order(self):
  expected=[(30,35),(45,50),(45,50),(40,45),(60,75),(75,90)]
  self.assertEqual([(u['estimated_minutes']['min'],u['estimated_minutes']['max']) for u in self.c['units']],expected)
  for proficiency in ['new_to_it','mostly_confident']:
   r=build_planning_roadmap(self.c,chapter_proficiency=proficiency,study_time_today='15')
   self.assertEqual(r['next_step']['estimated_minutes'],{'min':30,'max':35});self.assertLessEqual(r['daily_route']['total_minutes'],15)
 def test_mapping_exact_source_and_no_unassigned_subtopics(self):
  mapping=chapter_one_mapping();source=json.load(gzip.open(mapping['source'],'rt',encoding='utf-8'))
  self.assertEqual(mapping['source_digest'],source['digest'])
  chapter=next(c for c in source['chapters'] if c['chapter']['chapter_number']==1)
  byid={c['concept_id']:c for c in chapter['concepts']}
  identities=[]
  for unit,original in zip(mapping['units'],self.c['units']):
   self.assertEqual(unit['subtopics'],original['ncert_subtopics'])
   for page in unit['pages']:
    paragraph=byid[page['source_concept_id']]['core_explanation'].split('\n\n')[page['source_paragraph']]
    self.assertIn(page['text'],paragraph);self.assertEqual(hashlib.sha256(paragraph.encode()).hexdigest(),page['source_text_sha256'])
    identities.append((page['source_concept_id'],page['source_paragraph'],page['text']))
  self.assertEqual(len(identities),len(set(identities)));self.assertTrue(mapping['review_required'])
 def test_mass_topic_isolation_and_resume(self):
  start_route(self.db,user_id='reader',class_level='Class 11',subject='Chemistry',selections=[{'chapter_ref':self.c['chapter_slug'],'chapter_proficiency':'new_to_it'}])
  mass=self.c['units'][3]
  notes=mapped_notes(self.c['chapter_slug'],mass['id'],self.db,'reader')
  text=' '.join(p['text'] for p in notes['pages'])
  self.assertIn('Average atomic mass',text);self.assertNotIn('Stoichiometry',text);self.assertNotIn('molar mass',text)
  record_heartbeat(self.db,user_id='reader',last_position={'notes_topic':mass['id'],'notes_page':1})
  self.db.expire_all();self.assertEqual(mapped_notes(self.c['chapter_slug'],mass['id'],self.db,'reader')['saved_page'],1)
 def test_other_chapters_do_not_use_chapter_one_mapping(self):
  self.assertIsNone(mapped_notes('structure_of_atom','any',self.db,'reader'))
