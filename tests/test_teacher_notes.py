from copy import deepcopy
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database import Base
from models import AICoachProfile, TestHistory as History
from services.master_notes_service import resolve_unit
from services.teacher_notes_service import VERSION, _load_lesson, compile_lesson, session_portion, validate_lesson
from services.topic_notes_service import mapped_notes
from services.study_route_service import start_route, record_heartbeat, _get_state, StudyRouteError

CHAPTER = "some_basic_concepts_of_chemistry"

@pytest.fixture
def curriculum():
    return resolve_unit(CHAPTER, None)[0]

@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_sixty_meaningful_pages_across_all_adaptations(curriculum):
    for unit in curriculum["units"]:
        original = _load_lesson(unit["id"])
        validate_lesson(original, unit)
        for proficiency in ["new_to_it", "know_a_little", "know_the_basics", "mostly_confident"]:
            for style in ["concise", "balanced", "detailed"]:
                pages = compile_lesson(curriculum, unit, proficiency=proficiency, style=style)
                assert [p["page_number"] for p in pages] == list(range(1,11))
                assert len({p["id"] for p in pages}) == 10
                for authored, rendered in zip(original["pages"], pages):
                    assert len(rendered["blocks"]) >= 2
                    for block in authored["blocks"]:
                        if block["level"] == "essential": assert block in rendered["blocks"]
                    assert rendered["source"].startswith("NCERT")
                assert 5 <= len(pages[9]["questions"]) <= 10
                assert {s for p in pages for s in p["source_section_ids"]} == {s["id"] for s in unit["ncert_subtopics"]}


def test_adaptation_changes_support_without_cutting_core_science(curriculum):
    unit = curriculum["units"][1]
    novice = compile_lesson(curriculum, unit, proficiency="new_to_it")
    confident = compile_lesson(curriculum, unit, proficiency="mostly_confident")
    concise = compile_lesson(curriculum, unit, proficiency="mostly_confident", style="concise")
    detailed = compile_lesson(curriculum, unit, proficiency="mostly_confident", style="detailed")
    assert any(b["level"] == "foundation" for p in novice for b in p["blocks"])
    assert not any(b["level"] == "foundation" for p in confident for b in p["blocks"])
    assert not any(b["level"] == "extension" for p in concise for b in p["blocks"])
    assert sum(len(p["blocks"]) for p in detailed) > sum(len(p["blocks"]) for p in concise)
    assert [p["id"] for p in novice] == [p["id"] for p in confident]


def test_session_time_limits_a_portion_not_the_topic_workload(curriculum):
    for unit in curriculum["units"]:
        manual = deepcopy(unit["estimated_minutes"])
        pages = compile_lesson(curriculum, unit)
        short = session_portion(pages, unit, session_minutes=5)
        full = session_portion(pages, unit, session_minutes=manual["min"])
        assert short["end_page"] < 9 and short["additional_practice_needed"]
        assert full["end_page"] == 9 and not full["additional_practice_needed"]
        assert len(pages) == 10 and unit["estimated_minutes"] == manual
        assert session_portion(pages, unit, start_page=9, session_minutes=1)["additional_practice_needed"]


def test_versioned_page_resume_persists_without_mastery_or_legacy_reset(db,curriculum):
    unit = curriculum["units"][0]
    start_route(db,user_id="learner",class_level="Class 11",subject="Chemistry",selections=[{"chapter_ref":CHAPTER,"chapter_proficiency":"new_to_it"}],study_time_today="15")
    record_heartbeat(db,user_id="learner",last_position={"unit_id":unit["id"],"notes_topic":unit["id"],"notes_page":18})
    assert mapped_notes(CHAPTER,unit["id"],db,"learner")["saved_page"] == 0
    record_heartbeat(db,user_id="learner",last_position={"unit_id":unit["id"],"notes_topic":unit["id"],"notes_page":3,"notes_version":VERSION})
    db.expire_all()
    notes = mapped_notes(CHAPTER,unit["id"],db,"learner")
    assert notes["saved_page"] == 3 and len(notes["pages"]) == 10
    assert notes["session_portion"]["start_page"] == 0
    assert notes["proficiency"] == "new_to_it"
    assert notes["estimated_minutes"] == {"min":30,"max":35}
    assert notes["allocated_session_minutes"] <= 15
    assert _get_state(db,"learner").runtime_json["notes_positions"][unit["id"]] == 18
    assert db.query(History).count() == 0
    start_route(db,user_id="learner",class_level="Class 11",subject="Chemistry",selections=[{"chapter_ref":CHAPTER,"chapter_proficiency":"new_to_it"}],study_time_today="30")
    renewed=mapped_notes(CHAPTER,unit["id"],db,"learner")
    assert renewed["saved_page"] == 3
    assert renewed["session_portion"]["start_page"] == 3
    with pytest.raises(StudyRouteError):record_heartbeat(db,user_id="learner",last_position={"notes_topic":unit["id"],"notes_page":10,"notes_version":VERSION})
    db.rollback()


def test_saved_preference_and_proficiency_are_consumed(db,curriculum):
    start_route(db,user_id="learner",class_level="Class 11",subject="Chemistry",selections=[{"chapter_ref":CHAPTER,"chapter_proficiency":"mostly_confident"}],study_time_today="15")
    db.add(AICoachProfile(user_id="learner",coach_id="notes-test",study_preferences={"preferred_style":"detailed"}));db.commit()
    notes = mapped_notes(CHAPTER,curriculum["units"][1]["id"],db,"learner")
    assert notes["proficiency"] == "mostly_confident" and notes["explanation_style"] == "detailed"
    assert any(b["level"] == "foundation" for p in notes["pages"] for b in p["blocks"])


def test_validation_rejects_empty_filler_pages_and_scope_leaks(curriculum):
    unit = curriculum["units"][0]
    for mutate in [lambda d:d["pages"].pop(), lambda d:d["pages"][0].update(blocks=[]), lambda d:d["pages"][0].update(source_section_ids=["1.10"]), lambda d:d.update(questions=[])]:
        data = deepcopy(_load_lesson(unit["id"]));mutate(data)
        with pytest.raises(ValueError):validate_lesson(data,unit)


def test_ncert_notes_and_tutor_do_not_fall_back_to_older_paragraph_generation(curriculum):
    from Logic.coach.retriever import GroundedRetriever
    from Logic.tools.knowledge_search import search_knowledge_base
    unit = curriculum["units"][3]
    scope={"chapter":CHAPTER,"topic":unit["id"],"catalog_source":"planning_manifest"}
    with patch("services.master_notes_service.retrieve_notes",side_effect=AssertionError("Old Master excerpt generator used")):
        first=GroundedRetriever().retrieve("molecular mass",scope)
        second=search_knowledge_base(unit["id"],"molecular mass",scope=scope)
    assert first.source == second["source"] == "ncert_teacher_notes"
    assert "NCERT" in first.context and first.supported


def test_exact_page_and_subtopic_retrieval_do_not_expose_future_or_mixed_revision(curriculum):
    from services.teacher_notes_service import retrieve_teacher_context
    unit = curriculum["units"][3]
    scoped = retrieve_teacher_context(CHAPTER, unit["id"], subtopic_ids=["1.7.4"])
    assert scoped["provenance"]
    assert all(set(p["subtopic_ids"]) <= {"1.7.4"} for p in scoped["provenance"])
    page = compile_lesson(curriculum,unit)[1]
    current = retrieve_teacher_context(CHAPTER,unit["id"],page_number=2,subtopic_ids=page["source_section_ids"])
    assert current["matched_sections"] == [page["id"]]
    assert len(current["provenance"]) == 1


def test_legacy_full_slug_plan_retains_available_time_and_proficiency(db,curriculum):
    start_route(db,user_id="learner",class_level="Class 11",subject="Chemistry",selections=[{"chapter_ref":"ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry","chapter_proficiency":"mostly_confident"}],study_time_today="15")
    notes=mapped_notes(CHAPTER,curriculum["units"][0]["id"],db,"learner")
    assert notes["proficiency"] == "mostly_confident" and notes["allocated_session_minutes"] == 10


def test_question_retrieval_preserves_page_sequence(curriculum):
    from services.teacher_notes_service import retrieve_teacher_context
    result=retrieve_teacher_context(CHAPTER,curriculum["units"][3]["id"],question="revision formula mass sodium chloride")
    pages=compile_lesson(curriculum,curriculum["units"][3])
    numbers={p["id"]:p["page_number"] for p in pages}
    sequence=[numbers[p] for p in result["matched_sections"]]
    assert sequence==sorted(sequence)


def test_new_day_keeps_route_and_reading_position_but_starts_a_new_notes_portion(db,curriculum):
    from datetime import timedelta
    unit=curriculum["units"][0]
    args=dict(user_id="learner",class_level="Class 11",subject="Chemistry",selections=[{"chapter_ref":CHAPTER,"chapter_proficiency":"new_to_it"}],study_time_today="15")
    start_route(db,**args)
    record_heartbeat(db,user_id="learner",last_position={"unit_id":unit["id"],"notes_topic":unit["id"],"notes_page":4,"notes_version":VERSION})
    previous=_get_state(db,"learner")
    previous_id=previous.id
    with patch("services.study_route_service._today",return_value=previous.route_date+timedelta(days=1)):
        start_route(db,**args)
    latest=_get_state(db,"learner")
    assert latest.id == previous_id
    notes=mapped_notes(CHAPTER,unit["id"],db,"learner")
    assert notes["saved_page"] == notes["session_portion"]["start_page"] == 4
    assert db.query(History).count() == 0
