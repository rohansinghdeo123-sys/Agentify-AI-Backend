"""All published Chemistry chapters use the same source-bound teaching contract."""
import gzip
import json
from copy import deepcopy
from functools import lru_cache
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database import Base
from models import ContentChapter, ContentConcept, TestHistory as History
from services.catalog_service import resolve_published_planning_curriculum
from services.study_route_service import (
    StudyRouteError, _get_state, start_route, record_heartbeat, prepare_check, advance_route,
)
from services.teacher_notes_service import (
    ROOT, VERSION, teacher_curricula, resolve_teacher_unit, resolve_teacher_curriculum,
    _load_lesson, compile_lesson, retrieve_teacher_context, session_portion,
)
from services.topic_notes_service import mapped_notes


@lru_cache(maxsize=1)
def published_rows():
    payload = json.load(gzip.open(ROOT / "data/releases/ncert_class_11_chemistry_v1.json.gz", "rt", encoding="utf-8"))
    # Keep tests faithful to the published IDs/grouping without restoring the
    # unrelated, large embedding corpus or making model/network calls.
    return [(c["chapter"], c["concepts"]) for c in payload["chapters"]]


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for chapter, concepts in published_rows():
            row = ContentChapter(**{k:v for k,v in chapter.items() if k in ContentChapter.__table__.columns})
            session.add(row)
            session.flush()
            session.add_all([ContentConcept(chapter_id=row.id, **{k:v for k,v in c.items() if k in ContentConcept.__table__.columns}) for c in concepts])
        session.commit()
        yield session
    engine.dispose()


def test_registry_covers_every_published_chapter_without_changing_planning_ids_or_times(db):
    curricula = teacher_curricula()
    assert [c["chapter_number"] for c in curricula] == list(range(1,10))
    assert len({u["id"] for c in curricula for u in c["units"]}) == 72
    for c in curricula[2:]:
        actual = resolve_published_planning_curriculum(db, chapter_ref=c["chapter_slug"], subject="Chemistry", class_level="Class 11")
        assert actual["curriculum_key"] == c["curriculum_key"]
        assert [(u["id"],u["estimated_minutes"]) for u in actual["units"]] == [(u["id"],u["estimated_minutes"]) for u in c["units"]]


def test_deployment_includes_every_required_verified_source():
    docker_rules = (ROOT / ".dockerignore").read_text().splitlines()
    for c in teacher_curricula():
        assert "!" + c["source"]["path"] in docker_rules


@pytest.mark.parametrize("chapter", range(1,10))
def test_every_lesson_has_ten_scoped_meaningful_pages_across_all_adaptations(chapter):
    c = teacher_curricula()[chapter-1]
    for u in c["units"]:
        original = deepcopy(_load_lesson(u["id"]))
        estimates = deepcopy(u["estimated_minutes"])
        for proficiency in ["new_to_it","know_a_little","know_the_basics","mostly_confident"]:
            for style in ["balanced","concise","detailed"]:
                pages = compile_lesson(c,u,proficiency=proficiency,style=style)
                assert [p["page_number"] for p in pages] == list(range(1,11))
                assert len({p["id"] for p in pages}) == 10
                assert {s for p in pages for s in p["source_section_ids"]} == {s["id"] for s in u["ncert_subtopics"]}
                for source, rendered in zip(original["pages"],pages):
                    assert rendered["source_pages"]
                    assert min(rendered["source_pages"]) >= 1
                    assert max(rendered["source_pages"]) <= c["source"]["page_count"]
                    if u.get("source_segments"):
                        assert set(rendered["source_pages"]) <= {s["page"] for s in u["source_segments"]}
                    assert len(rendered["blocks"]) >= 2
                    assert all(b in rendered["blocks"] for b in source["blocks"] if b["level"] == "essential")
                    assert "\ufffd" not in rendered["text"]
                assert 5 <= len(pages[-1]["questions"]) <= 10
                assert all(q["answer"].strip() for q in pages[-1]["questions"])
                assert u["estimated_minutes"] == estimates
        assert _load_lesson(u["id"]) == original, "Per-learner adaptation mutated shared content"
        portion = session_portion(pages,u,session_minutes=5)
        assert portion["additional_practice_needed"]


@pytest.mark.parametrize("chapter", range(2,10))
def test_notes_endpoint_adapter_aliases_and_exact_page_grounding(db,chapter):
    c = teacher_curricula()[chapter-1]
    u = c["units"][0]
    alias = c.get("aliases",[c["chapter_slug"]])[0]
    assert resolve_teacher_curriculum(alias)["curriculum_key"] == c["curriculum_key"]
    notes = mapped_notes(alias,u["id"],db,"reader")
    assert notes["chapter"]["chapter_number"] == chapter
    assert len(notes["pages"]) == 10
    assert notes["topic_id"] == u["id"]
    assert notes["estimated_minutes"] == u["estimated_minutes"]
    assert notes["can_save_position"] is False
    page = notes["pages"][1]
    result = retrieve_teacher_context(alias,u["id"],question="Explain this page",page_number=2,subtopic_ids=page["source_section_ids"])
    assert result["matched_sections"] == [page["id"]]
    assert result["source_pages"] == page["source_pages"]
    assert len(result["provenance"]) == 1
    with pytest.raises(ValueError):
        retrieve_teacher_context(alias,u["id"],subtopic_ids=["another-chapter"])
    with pytest.raises(HTTPException):
        mapped_notes(alias,teacher_curricula()[0]["units"][0]["id"],db,"reader")


def test_published_notes_ask_ai_uses_exact_page_but_revision_keeps_published_retrieval():
    from Logic.coach.retriever import GroundedRetriever
    from Logic.tools.knowledge_search import search_knowledge_base
    c = teacher_curricula()[8]
    u = c["units"][0]
    page = compile_lesson(c,u)[2]
    scope = {"chapter_slug":c["chapter_slug"],"topic":u["id"],"catalog_source":"published","subject":"Chemistry","class_level":"11",
             "notes_content_version":VERSION,"notes_page_number":3,"subtopic_ids":page["source_section_ids"]}
    with patch("Logic.coach.retriever.search_approved_content",side_effect=AssertionError("Page-scoped notes must not retrieve another source")):
        result = GroundedRetriever().retrieve("Explain this",scope)
        assert result.source == "ncert_teacher_notes"
        assert result.matched_sections == [page["id"]]
    result = search_knowledge_base(u["id"],"Explain this",scope=scope)
    assert result["matched_sections"] == [page["id"]]
    revision = {k:v for k,v in scope.items() if not k.startswith("notes_") and k != "subtopic_ids"}
    with patch("Logic.coach.retriever.search_approved_content",return_value={"context":"Original published Revision source","source":"approved_content_pipeline"}):
        assert GroundedRetriever().retrieve("Revise hydrocarbons",revision).source == "approved_content_pipeline"
    assert resolve_teacher_unit(c["chapter_slug"],teacher_curricula()[7]["units"][0]["id"]) is None


def test_tutor_retrieves_the_same_adapted_blocks_the_learner_reads():
    c = teacher_curricula()[1]
    u = c["units"][5]
    for proficiency, style in [("new_to_it","balanced"),("mostly_confident","concise"),("mostly_confident","detailed")]:
        pages = compile_lesson(c,u,proficiency=proficiency,style=style)
        for page in pages:
            result = retrieve_teacher_context(c["chapter_slug"],u["id"],page_number=page["page_number"],
                subtopic_ids=page["source_section_ids"],proficiency=proficiency,style=style)
            assert all(b["text"] in result["context"] for b in page["blocks"])


def test_multi_chapter_positions_persist_without_changing_active_route_or_awarding_mastery(db):
    chapters = [teacher_curricula()[1],teacher_curricula()[8]]
    start_route(db,user_id="portfolio-reader",class_level="Class 11",subject="Chemistry",study_time_today="15",
                selections=[{"chapter_ref":c["chapter_slug"],"chapter_proficiency":"mostly_confident"} for c in chapters])
    state = _get_state(db,"portfolio-reader")
    active, key = state.current_unit_id, state.active_curriculum_key
    for c in chapters:
        u = c["units"][0]
        record_heartbeat(db,user_id="portfolio-reader",last_position={"unit_id":active,"notes_topic":u["id"],"notes_page":4,"notes_version":VERSION})
        db.expire_all()
        notes = mapped_notes(c["chapter_slug"],u["id"],db,"portfolio-reader")
        assert notes["saved_page"] == 4 and notes["can_save_position"]
        assert notes["proficiency"] == "mostly_confident"
    state = _get_state(db,"portfolio-reader")
    assert (state.current_unit_id,state.active_curriculum_key) == (active,key)
    assert db.query(History).count() == 0
    with pytest.raises(StudyRouteError):
        record_heartbeat(db,user_id="portfolio-reader",last_position={"notes_topic":teacher_curricula()[2]["units"][0]["id"],"notes_page":0,"notes_version":VERSION})


def test_published_chapter_learning_check_still_persists_mastery_and_advances(db):
    c = teacher_curricula()[2]
    start_route(db,user_id="assessed-reader",class_level="Class 11",subject="Chemistry",study_time_today="30",
                selections=[{"chapter_ref":c["chapter_slug"],"chapter_proficiency":"new_to_it"}])
    def generated(**kwargs):
        return {"questions":[{"id":f"Q{i}","question":f"Application {i}","options":["A. one","B. two","C. three","D. four"],"correct":"B","explanation":"Reasoned answer."} for i in range(kwargs["count"])]}
    with patch("Logic.section_doubt.generate_structured_mcqs",side_effect=generated),patch("app.security.enforce_user_quota"):
        pack = prepare_check(db,user_id="assessed-reader")
    result = advance_route(db,user_id="assessed-reader",completed_unit_id=pack["unit_id"],interaction_id=pack["id"],answers={q["id"]:"B" for q in pack["questions"]})
    assert result["completion"]["mastered"]
    assert result["route"]["current_step"]["current_unit_id"] != pack["unit_id"]
    assert db.query(History).count() == 1


def test_revision_bookmarks_save_after_route_completion_without_reopening_it(db):
    c = teacher_curricula()[2]
    start_route(db,user_id="finished-reader",class_level="Class 11",subject="Chemistry",study_time_today="15",
                selections=[{"chapter_ref":c["chapter_slug"],"chapter_proficiency":"new_to_it"}])
    state = _get_state(db,"finished-reader")
    state.status = "completed"
    state.last_position_json = {"conversation_id":"retained-conversation","check_answers":{"Q0":"B"}}
    db.commit()
    record_heartbeat(db,user_id="finished-reader",compact=True,last_position={"unit_id":state.current_unit_id,
        "notes_topic":c["units"][0]["id"],"notes_page":5,"notes_version":VERSION})
    db.expire_all()
    state = _get_state(db,"finished-reader")
    assert state.status == "completed"
    assert state.last_position_json == {"conversation_id":"retained-conversation","check_answers":{"Q0":"B"}}
    assert mapped_notes(c["chapter_slug"],c["units"][0]["id"],db,"finished-reader")["saved_page"] == 5
    assert db.query(History).count() == 0


def test_existing_chapter_two_catalog_links_use_teaching_units_or_explicit_scoped_choices(db):
    from routers.study import get_chapter_notes
    from services.catalog_service import build_catalog
    c = teacher_curricula()[1]
    catalog = build_catalog(db)
    chapter = next(ch for s in catalog["subjects"] for ch in s["chapters"] if ch["chapter_number"] == 2)
    assert len(chapter["topics"]) == 10
    for index, topic in enumerate(chapter["topics"]):
        notes = get_chapter_notes(chapter["slug"],topic_id=topic["id"],db=db,current_user={"uid":"archived-reader"})
        if index < 8:
            assert len(notes["pages"]) == 10
            assert notes["topic_id"] == c["units"][[0,1,2,2,3,4,5,6][index]]["id"]
            assert notes["catalog_source"] == "planning_manifest"
        else:
            assert notes['resolution'] == 'topic_choice'
            assert len(notes['topic_options']) >= 2
            assert {o['id'] for o in notes['topic_options']} <= {u['id'] for u in c['units']}


@pytest.mark.parametrize('chapter_number', range(1,10))
def test_every_catalog_source_subtopic_concept_and_alias_opens_meaningful_notes(db,chapter_number):
    from routers.study import get_chapter_notes
    from services.catalog_service import build_catalog
    from services.teacher_notes_service import _topic_aliases, _topic_identity
    c = teacher_curricula()[chapter_number-1]
    raw_chapter, concepts = next(row for row in published_rows() if row[0]['chapter_number'] == chapter_number)
    catalog = build_catalog(db)
    published = next(ch for s in catalog['subjects'] for ch in s['chapters'] if ch['slug'] == raw_chapter['slug'])
    cases = {topic['id'] for topic in published['topics']}
    for u in c['units']:
        cases.update([u['id'],u['primary_topic_id'],*u.get('legacy_topic_ids',[])])
        for field in ['concepts','ncert_subtopics']:
            cases.update(item['id'] for item in u[field])
        cases.update(_topic_identity(s['id']) for s in u['ncert_subtopics'])
    for concept in concepts:
        cases.add(concept['concept_id'])
        cases.update((concept.get('raw_json') or {}).get('source_concept_ids') or [])
    cases.update(_topic_aliases()[c['curriculum_key']]['aliases'])
    allowed = {u['id'] for u in c['units']}
    checked_options = set()
    for topic in sorted(cases):
        notes = get_chapter_notes(raw_chapter['slug'],topic_id=topic,db=db,current_user={'uid':'coverage-reader'})
        assert notes['chapter']['chapter_number'] == chapter_number, topic
        if notes['resolution'] == 'topic_choice':
            ids = [o['id'] for o in notes['topic_options']]
            assert len(ids) >= 2 and len(ids) == len(set(ids)), topic
            assert set(ids) <= allowed and notes['can_save_position'] is False, topic
            assert ids == [u['id'] for u in c['units'] if u['id'] in ids], topic
            for unit_id in set(ids)-checked_options:
                lesson = mapped_notes(c['chapter_slug'],unit_id,db,'coverage-reader')
                assert len(lesson['pages']) == 10
                assert all(p['text'].strip() and len(p['blocks']) >= 2 for p in lesson['pages'])
                checked_options.add(unit_id)
        else:
            assert notes['topic_id'] in allowed and len(notes['pages']) == 10, topic
            assert all(p['text'].strip() and len(p['blocks']) >= 2 for p in notes['pages']), topic
            assert type(notes['saved_page']) is int and 0 <= notes['saved_page'] < 10, topic


def test_screenshot_group_and_final_review_expose_all_related_lessons_without_state_guessing(db):
    from services.teacher_notes_service import resolve_teacher_unit
    c = teacher_curricula()[0]
    start_route(db,user_id='mixed-reader',class_level='Class 11',subject='Chemistry',study_time_today='15',
                selections=[{'chapter_ref':c['chapter_slug'],'chapter_proficiency':'new_to_it'}])
    active = _get_state(db,'mixed-reader').current_unit_id
    for suffix, expected in [('2cb26c9c02',[1,2,3,4,5]),('fc4070d4c3',[1,3,4,5])]:
        old_id = 'unit_ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry_' + suffix
        assert resolve_teacher_unit(c['chapter_slug'],old_id,active) is None
        notes = mapped_notes(c['chapter_slug'],old_id,db,'mixed-reader')
        assert notes['resolution'] == 'topic_choice' and not notes['pages']
        assert [o['id'] for o in notes['topic_options']] == [c['units'][i]['id'] for i in expected]
        assert _get_state(db,'mixed-reader').current_unit_id == active
        assert db.query(History).count() == 0


def test_original_seed_topic_links_remain_compatible(db):
    from services.master_notes_service import CHAPTER_ONE_SEED_GROUPS
    c = teacher_curricula()[0]
    for alias, index in CHAPTER_ONE_SEED_GROUPS.items():
        notes = mapped_notes(c['chapter_slug'],alias,db,'seed-reader')
        assert notes['topic_id'] == c['units'][index]['id']
        assert len(notes['pages']) == 10


def test_direct_subtopic_starts_on_its_own_first_page_and_keeps_parent_lesson_scope(db):
    from services.teacher_notes_service import _topic_identity
    for c in teacher_curricula():
        for u in c['units']:
            for subtopic in u['ncert_subtopics']:
                notes = mapped_notes(c['chapter_slug'],_topic_identity(subtopic['id']),db,'subtopic-reader')
                assert notes['topic_id'] == u['id']
                assert subtopic['id'] in notes['pages'][notes['saved_page']]['source_section_ids']
                assert {s for p in notes['pages'] for s in p['source_section_ids']} <= {s['id'] for s in u['ncert_subtopics']}


def test_changed_catalog_group_hash_resolves_via_exact_approved_members(db):
    from services.catalog_service import build_catalog
    from services.teacher_notes_service import resolve_teacher_topics
    c = teacher_curricula()[5]
    with patch('services.topic_grouping.meaningful_topic_budget',return_value=2):
        catalog = build_catalog(db)
        chapter = next(ch for s in catalog['subjects'] for ch in s['chapters'] if ch['slug'] == c['chapter_slug'])
        grouped_id = chapter['topics'][0]['id']
        assert not resolve_teacher_topics(c['chapter_slug'],grouped_id)[1]
        notes = mapped_notes(c['chapter_slug'],grouped_id,db,'regroup-reader')
        assert notes['resolution'] == 'topic_choice'
        assert {o['id'] for o in notes['topic_options']} == {u['id'] for u in c['units'][:5]}


def test_alias_resolution_never_crosses_explicit_syllabus_boundaries(db):
    from services.teacher_notes_service import resolve_teacher_topics
    c = teacher_curricula()[0]
    for prefix in ['ncert_class_12_chemistry_chapter_1_','ncert_class_11_biology_chapter_1_']:
        assert resolve_teacher_topics(prefix+c['chapter_slug'],c['units'][0]['id']) is None
    for other in teacher_curricula()[1:]:
        with pytest.raises(HTTPException):
            mapped_notes(c['chapter_slug'],other['units'][0]['id'],db,'wrong-chapter-reader')


@pytest.mark.parametrize('position',[None,'NaN','4',1.5,-5,1000,{},[],True])
def test_old_malformed_bookmarks_never_make_a_lesson_blank(db,position):
    c = teacher_curricula()[0]
    u = c['units'][0]
    start_route(db,user_id='old-bookmark-reader',class_level='Class 11',subject='Chemistry',study_time_today='15',
                selections=[{'chapter_ref':c['chapter_slug'],'chapter_proficiency':'new_to_it'}])
    state = _get_state(db,'old-bookmark-reader')
    state.runtime_json = {**state.runtime_json,'notes_positions_by_version':{VERSION:{u['id']:position}},
                          'notes_session_start_positions':{VERSION:{u['id']:position}}}
    db.commit()
    notes = mapped_notes(c['chapter_slug'],u['id'],db,'old-bookmark-reader')
    assert type(notes['saved_page']) is int and 0 <= notes['saved_page'] < 10
    assert notes['pages'][notes['saved_page']]['blocks']
