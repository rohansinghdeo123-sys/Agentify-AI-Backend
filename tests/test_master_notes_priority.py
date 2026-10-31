import hashlib
from unittest.mock import patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import Base
from models import ContentChunk
from services.master_notes_service import artifact, install_master_notes, retrieve_notes, lesson_pages, resolve_unit
from services.topic_notes_service import mapped_notes

@pytest.fixture
def db():
    engine=create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        install_master_notes(session)
        yield session
    engine.dispose()

CHAPTER="some_basic_concepts_of_chemistry"
def unit(index):
    from Logic.planning.curriculum_registry import resolve_planning_curriculum
    return resolve_planning_curriculum(chapter_ref=CHAPTER,subject="Chemistry",class_level="Class 11")["units"][index]["id"]

def test_persisted_embedded_idempotent(db):
    rows=db.query(ContentChunk).all()
    assert len(rows)==64
    assert all(r.embedding and len(r.embedding)>100 and r.metadata_json["embedding_model"] for r in rows)
    assert all(hashlib.sha256(r.text.encode()).hexdigest()==r.metadata_json["sha256"] for r in rows)
    install_master_notes(db)
    assert db.query(ContentChunk).count()==64

def test_sufficient_master_stops_ncert(db):
    with patch("services.master_notes_service.ncert_gap",side_effect=AssertionError("NCERT must not run")):
        for i in range(1,6):
            result=retrieve_notes(db,CHAPTER,unit(i))
            assert result["sections"] and not result["missing_subtopics"]
            assert all(s["source"]=="Amit Master Notes" for s in result["sections"])

def test_only_missing_opening_sections_query_ncert(db):
    from services.master_notes_service import ncert_gap
    with patch("services.master_notes_service.ncert_gap",wraps=ncert_gap) as fallback:
        result=retrieve_notes(db,CHAPTER,unit(0))
    assert {call.args[3] for call in fallback.call_args_list}=={"intro.development_of_chemistry","1.1"}
    assert any(s["source"]=="Supplementary NCERT" for s in result["sections"])
    assert len({s["text"] for s in result["sections"]})==len(result["sections"])

def test_exact_subtopic_scope_and_notation(db):
    result=retrieve_notes(db,CHAPTER,unit(3),["1.7.3"])
    text=" ".join(s["text"] for s in result["sections"])
    assert "H₂O = 2 x 1 + 16 = 18 u" in text
    assert "CO₂ = 12 + 2 x 16 = 44 u" in text
    assert "molar mass" not in text.lower()
    with pytest.raises(ValueError):retrieve_notes(db,CHAPTER,unit(3),["1.10"])

def test_conflicting_supplement_is_flagged_and_withheld(db):
    with patch("services.master_notes_service.ncert_gap",return_value=[{"id":"bad","text":"Matter has no mass","subtopic_ids":["1.2"],"source":"Supplementary NCERT"}]):
        result=retrieve_notes(db,CHAPTER,unit(0))
    assert result["conflicts"]
    assert "bad" not in [s["id"] for s in result["sections"]]

def test_content_dependent_structure_and_worked_examples(db):
    laws=lesson_pages(retrieve_notes(db,CHAPTER,unit(2)))
    assert any(b["label"]=="Statement" for p in laws for b in p["blocks"])
    mass=lesson_pages(retrieve_notes(db,CHAPTER,unit(3)))
    assert any(p["teaching_format"]=="numerical" for p in mass)
    mole=lesson_pages(retrieve_notes(db,CHAPTER,unit(4)))
    text=" ".join(p["text"] for p in mole)
    assert "36 / 18 = 2 mol" in text and "Practice:" in text
    concept=lesson_pages(retrieve_notes(db,CHAPTER,unit(0),["1.2"]))
    assert "Why this example works" in concept[0]["text"]
    assert "copper" in concept[0]["text"]

def test_real_coach_and_tool_share_master_first_pipeline(db):
    from Logic.coach.retriever import GroundedRetriever
    from Logic.tools.knowledge_search import search_knowledge_base
    # Real persistent DB retrieval through both public entry points.
    with patch("database.SessionLocal",return_value=db),patch("services.master_notes_service.ncert_gap",side_effect=AssertionError("NCERT queried")):
        result=GroundedRetriever().retrieve("Explain molecular mass",{"chapter":CHAPTER,"topic":unit(3),"catalog_source":"planning_manifest"})
        assert result.source=="amit_master_first" and "18 u" in result.context
        result=search_knowledge_base(unit(3),"molecular mass",scope={"chapter":CHAPTER,"catalog_source":"planning_manifest"})
        assert result["source"]=="amit_master_first"


def test_every_indexed_section_matches_the_actual_pdf():
    import pymupdf
    from services.master_notes_service import ROOT
    doc=pymupdf.open(ROOT / "data/notes/amit_chapter1.pdf")
    lines=[]
    import re
    for page in doc:
        lines.extend(line.strip() for line in page.get_text().splitlines()
            if not line.startswith("Class XI Chemistry") and not re.fullmatch(r"Page \d+",line.strip()))
    original=" ".join(" ".join(lines).split())
    for section in artifact()["sections"]:
        assert " ".join(section["text"].split()) in original, section["id"]


def test_conflicting_formula_in_a_supplement_is_flagged():
    from services.master_notes_service import formula_conflicts
    master=[{"source":"Amit Master Notes","text":"Density = mass / volume."}]
    assert formula_conflicts(master,{"text":"Density = volume / mass."})
    assert not formula_conflicts(master,{"text":"Density = mass / volume."})


def test_supplement_does_not_repeat_existing_master_sentences(db):
    duplicate="Matter is anything that has mass and occupies space."
    extra="This supplementary section is about the development of chemistry."
    with patch("services.master_notes_service.ncert_gap",return_value=[{"id":"supplement","text":duplicate+" "+extra,
        "subtopic_ids":["intro.development_of_chemistry"],"source":"Supplementary NCERT"}]):
        result=retrieve_notes(db,CHAPTER,unit(0))
    supplement=[s for s in result["sections"] if s["source"]=="Supplementary NCERT"]
    assert supplement and all(duplicate not in s["text"] for s in supplement)


def test_legacy_screenshot_url_uses_amit_measuring_matter(db):
    legacy="unit_ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry_9a85991e10"
    result=mapped_notes("ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry",legacy,db,"reader")
    assert result["topic_id"]==unit(1)
    assert all(p["source"]=="Amit Master Notes" for p in result["pages"])
    text=" ".join(p["text"] for p in result["pages"])
    assert "kilo (k) = 10^3" in text
    assert "National standards are maintained" not in text
    assert mapped_notes(CHAPTER,None,db,"reader")["pages"]
    from fastapi import HTTPException
    with pytest.raises(HTTPException):mapped_notes(CHAPTER,"unknown-outdated-topic",db,"reader")

def test_every_archived_concept_link_resolves_to_master_notes(db):
    from services.topic_notes_service import chapter_one_mapping
    for old_id in {p["source_concept_id"] for u in chapter_one_mapping()["units"] for p in u["pages"]}:
        result=mapped_notes(CHAPTER,old_id,db,"reader")
        assert result["pages"] and result["topic_id"].startswith("chem11_u01_")
