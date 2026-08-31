"""Focused tests for content-safe admin evidence contracts."""

import unittest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base
from models import ContentChapter, ContentChunk, ContentConcept, ContentPage, ModelToolTrace
from services.admin_evidence_service import (
    build_activity_evidence,
    build_chapter_evidence,
    build_content_evidence,
    build_evidence_overview,
)


class AdminEvidenceTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        chapter = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            book_name="Chemistry I",
            chapter_number=1,
            chapter_name="Some Basic Concepts of Chemistry",
            slug="chemistry-11-1",
            status="published",
            version="v2",
            source_hash="sha256:source",
            published_source_hash="sha256:source",
            page_count=2,
            extracted_page_count=2,
            coverage_score=0.98,
            extraction_quality=0.97,
            validation_report={
                "blocking_issue_count": 0,
                "issues": [{"severity": "warning", "message": "Non-blocking review note"}],
            },
            published_at=datetime.utcnow(),
        )
        self.db.add(chapter)
        self.db.flush()
        self.chapter_id = chapter.id
        self.db.add_all(
            [
                ContentPage(chapter_id=chapter.id, page_number=1, text="private source text", char_count=19),
                ContentPage(chapter_id=chapter.id, page_number=2, text="more private text", char_count=17),
                ContentConcept(
                    chapter_id=chapter.id,
                    concept_id="mole-concept",
                    title="Mole Concept",
                    definition="must not be returned",
                    core_explanation="must not be returned",
                    key_points=["one"],
                    examples=["one"],
                    source_pages=[1, 2],
                    validation_issues=[],
                ),
                ContentConcept(
                    chapter_id=chapter.id,
                    concept_id="stoichiometry",
                    title="Stoichiometry",
                    definition="hidden",
                    source_pages=[2],
                    validation_issues=[{"code": "needs_example", "message": "Add example"}],
                ),
                ContentChunk(
                    chapter_id=chapter.id,
                    chunk_id="chunk-1",
                    text="must not be returned",
                    page_start=1,
                    page_end=1,
                    embedding=[0.1, 0.2],
                    metadata_json={"embedding_model": "test", "embedding_dimensions": 2},
                ),
                ContentChunk(
                    chapter_id=chapter.id,
                    chunk_id="chunk-2",
                    text="must not be returned",
                    page_start=2,
                    page_end=2,
                    embedding=None,
                ),
                ModelToolTrace(
                    created_at=datetime.utcnow(),
                    user_id="private-user",
                    session_id="private-session",
                    turn_id="private-turn",
                    trace_type="turn",
                    name="coach_turn",
                    status="success",
                    latency_ms=120,
                    metadata_json={
                        "query": {"question": "private prompt"},
                        "quality": {
                            "score": 0.91,
                            "passed": True,
                            "grounding": 0.95,
                            "hallucination_risk": 0.02,
                            "issues": [],
                        },
                        "retrieval": {
                            "policy": "required",
                            "source": "content_db",
                            "paragraphs_found": 2,
                            "supported": True,
                            "source_pages": [1, 2],
                            "gate": {"grounding_status": "grounded"},
                        },
                    },
                ),
            ]
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_content_list_is_paginated_and_has_proof_not_text(self):
        payload = build_content_evidence(
            self.db, class_level="11", subject="chemistry", limit=1, offset=0
        )
        self.assertEqual(payload["pagination"]["total"], 1)
        item = payload["items"][0]
        self.assertEqual(item["counts"]["subtopics"], 2)
        self.assertEqual(item["counts"]["embedded_chunks"], 1)
        self.assertEqual(item["evidence"]["source_page_coverage_percent"], 100.0)
        self.assertTrue(item["source_integrity"]["published_hash_matches"])
        self.assertEqual(item["quality"]["blocking_issue_count"], 0)
        self.assertTrue(item["quality"]["ready"])
        self.assertNotIn("text", str(payload).lower())

    def test_chapter_detail_paginates_subtopics_and_excludes_raw_content(self):
        payload = build_chapter_evidence(self.db, self.chapter_id, limit=1, offset=0)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["pagination"]["total"], 2)
        self.assertEqual(len(payload["subtopics"]), 1)
        self.assertEqual(payload["subtopics"][0]["source_proof"]["page_numbers"], [1, 2])
        self.assertEqual(payload["chapter"]["blocking_issue_count"], 0)
        serialized = str(payload).lower()
        for forbidden in ("must not be returned", "private source text", "embedding\":"):
            self.assertNotIn(forbidden, serialized)

    def test_missing_page_reference_is_not_reported_as_verified_proof(self):
        concept = (
            self.db.query(ContentConcept)
            .filter(ContentConcept.concept_id == "stoichiometry")
            .one()
        )
        concept.source_pages = [2, 3]
        self.db.commit()

        summary = build_content_evidence(self.db)["items"][0]
        self.assertEqual(summary["evidence"]["missing_source_pages"], [3])
        self.assertEqual(summary["evidence"]["source_page_coverage_percent"], 50.0)
        self.assertFalse(summary["quality"]["ready"])

        detail = build_chapter_evidence(self.db, self.chapter_id, limit=10)
        assert detail is not None
        subtopic = next(item for item in detail["subtopics"] if item["concept_id"] == "stoichiometry")
        self.assertEqual(subtopic["source_proof"]["verified_page_numbers"], [2])
        self.assertEqual(subtopic["source_proof"]["missing_page_numbers"], [3])
        self.assertFalse(subtopic["source_proof"]["verified"])

    def test_activity_is_sanitized_and_evidence_backed(self):
        payload = build_activity_evidence(self.db, hours=24, limit=10)
        self.assertEqual(payload["pagination"]["total"], 1)
        item = payload["items"][0]
        self.assertEqual(item["agent"], "coach")
        self.assertEqual(item["grounding"]["status"], "grounded")
        self.assertEqual(item["grounding"]["source_pages"], [1, 2])
        serialized = str(payload)
        for private in ("private-user", "private-session", "private-turn", "private prompt"):
            self.assertNotIn(private, serialized)

    def test_overview_has_agent_quality_and_readiness(self):
        payload = build_evidence_overview(self.db, hours=24)
        self.assertEqual(payload["content"]["chapters"], 1)
        self.assertEqual(payload["quality"]["grounded_rate_percent"], 100.0)
        self.assertEqual(payload["agents"][0]["success_rate_percent"], 100.0)
        agents = {item["agent"]: item for item in payload["agents"]}
        self.assertTrue({"orchestrator", "coach", "tutor", "planner", "revision", "exam"}.issubset(agents))
        self.assertEqual(agents["exam"]["activity_state"], "not_observed")
        self.assertEqual(agents["exam"]["health"], "attention")
        self.assertIn("release", payload["readiness"])
        self.assertIn("semantic_retrieval", payload["readiness"])


if __name__ == "__main__":
    unittest.main()
