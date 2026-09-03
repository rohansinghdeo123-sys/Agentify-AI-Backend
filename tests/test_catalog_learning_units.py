import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base
from Logic.agents.coach_agent import _selected_material_scope
from Logic.content_pipeline import search_approved_content
from Logic.tools.artifact_generator import generate_study_artifacts
from Logic.tools.knowledge_search import search_knowledge_base
from models import ContentChapter, ContentChunk, ContentConcept
from routers.coach import CoachTurnRequest, _resolve_selected_catalog_topic
from schemas import CoachChatRequest
from services.catalog_service import build_catalog, resolve_catalog_topic


class CatalogLearningUnitTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        self.SessionTesting = sessionmaker(bind=engine)
        self.db = self.SessionTesting()

    def tearDown(self):
        self.db.close()

    def _published_chapter(self, count: int = 20):
        chapter = ContentChapter(
            board="NCERT",
            class_level="10",
            subject="Science",
            chapter_name="Life Processes",
            slug="ncert_class_10_science_life_processes",
            status="published",
            page_count=count,
            extracted_page_count=count,
            version="v1",
        )
        self.db.add(chapter)
        self.db.flush()
        for index in range(1, count + 1):
            family = "Nutrition" if index <= count // 2 else "Respiration"
            self.db.add(
                ContentConcept(
                    chapter_id=chapter.id,
                    concept_id=f"life_process_{index}",
                    title=f"{family} concept {index}",
                    definition=f"Grounded syllabus detail {index} about {family.lower()}.",
                    key_points=[f"Coverage point {index}"],
                    source_pages=[index],
                    difficulty_level=2,
                    importance_level="core",
                    raw_json={},
                )
            )
        self.db.commit()
        return chapter

    def test_existing_published_microtopics_are_grouped_in_backend_catalog(self):
        chapter = self._published_chapter(20)

        catalog = build_catalog(self.db)
        science = next(group for group in catalog["subjects"] if group["subject"] == "Science")
        published = next(item for item in science["chapters"] if item["slug"] == chapter.slug)
        topics = published["topics"]

        self.assertEqual(len(topics), 5)
        self.assertCountEqual(
            [concept_id for topic in topics for concept_id in topic["concept_ids"]],
            [f"life_process_{index}" for index in range(1, 21)],
        )

        unit = topics[0]
        resolved = resolve_catalog_topic(
            self.db,
            unit["id"],
            subject="Science",
            chapter=chapter.slug,
            topic=unit["label"],
            class_level="10",
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["section_id"], unit["id"])
        self.assertEqual(resolved["concept_ids"], unit["concept_ids"])

    def test_legacy_microtopic_and_persisted_alias_both_resolve(self):
        chapter = self._published_chapter(8)
        legacy = resolve_catalog_topic(
            self.db,
            "life_process_3",
            subject="Science",
            chapter=chapter.slug,
        )
        self.assertEqual(legacy["section_id"], "life_process_3")

        consolidated = ContentConcept(
            chapter_id=chapter.id,
            concept_id="unit_persisted_nutrition",
            title="Nutrition and its processes",
            definition="Consolidated grounded content.",
            source_pages=[30],
            raw_json={"source_concept_ids": ["old_nutrition_definition", "old_nutrition_examples"]},
        )
        self.db.add(consolidated)
        self.db.commit()

        alias = resolve_catalog_topic(
            self.db,
            "old_nutrition_examples",
            subject="Science",
            chapter=chapter.slug,
        )
        self.assertEqual(alias["section_id"], "unit_persisted_nutrition")
        self.assertEqual(alias["concept_ids"], ["unit_persisted_nutrition"])

    def test_group_scope_retrieves_every_member_without_frontend_truncation(self):
        chapter = self._published_chapter(12)
        topics = build_catalog(self.db)["subjects"][0]["chapters"][0]["topics"]
        unit = topics[0]
        resolved = resolve_catalog_topic(
            self.db,
            unit["id"],
            subject="Science",
            chapter=chapter.slug,
            topic=unit["label"],
        )

        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting), patch(
            "Logic.content_pipeline.embeddings_service.embed_query", return_value=None
        ):
            result = search_approved_content(
                resolved["section_id"],
                f"Teach {resolved['topic']}",
                scope=resolved,
                max_chars=50000,
                limit=20,
            )

        self.assertEqual(result["source"], "approved_content_pipeline")
        self.assertCountEqual(result["matched_sections"], resolved["concept_ids"])

    def test_study_coach_resolves_frontend_group_id_to_published_members(self):
        chapter = self._published_chapter(12)
        unit = build_catalog(self.db)["subjects"][0]["chapters"][0]["topics"][0]
        payload = CoachChatRequest(
            user_id="student-1",
            message="Explain this concept from the basics with one simple example.",
            subject="Science",
            chapter=chapter.chapter_name,
            topic=unit["label"],
            section_id=unit["id"],
            strict_grounding=True,
            retrieval_required=True,
            fallback_to_general_knowledge=False,
            learning_context={
                "scope": "selected_study_material_only",
                "catalog_source": "published",
                "class_level": "Class 10",
                "selected_subject": "Science",
                "selected_chapter_id": chapter.slug,
                "selected_chapter": chapter.chapter_name,
                "selected_topic_id": unit["id"],
                "selected_topic": unit["label"],
            },
        )

        resolved = _resolve_selected_catalog_topic(
            self.db,
            payload,
            {"class_level": "Class 10"},
        )
        request = CoachTurnRequest(
            payload,
            {"class_level": "Class 10"},
            resolved,
        )
        scope = _selected_material_scope(
            request,
            {"learning_context": request.learning_context},
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(scope["chapter_slug"], chapter.slug)
        self.assertEqual(scope["content_version"], "v1")
        self.assertEqual(scope["catalog_source"], "published")
        self.assertCountEqual(scope["concept_ids"], unit["concept_ids"])

    def test_group_scope_remains_usable_by_study_tools(self):
        chapter = self._published_chapter(12)
        unit = build_catalog(self.db)["subjects"][0]["chapters"][0]["topics"][0]
        resolved = resolve_catalog_topic(
            self.db,
            unit["id"],
            subject="Science",
            chapter=chapter.slug,
            topic=unit["label"],
        )

        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting), patch(
            "Logic.content_pipeline.embeddings_service.embed_query", return_value=None
        ):
            artifact = generate_study_artifacts(
                section_id=resolved["section_id"],
                topic=resolved["topic"],
                subject=resolved["subject"],
                chapter=resolved["chapter"],
                content_scope=resolved,
            )

        self.assertTrue(artifact["available"])
        self.assertEqual(artifact["topic"], resolved["topic"])
        self.assertTrue(artifact["artifacts"])

    def test_group_retrieval_keeps_chunk_when_member_page_is_inside_its_range(self):
        chapter = self._published_chapter(3)
        member = self.db.query(ContentConcept).filter(
            ContentConcept.chapter_id == chapter.id,
            ContentConcept.concept_id == "life_process_2",
        ).one()
        member.source_pages = [150]
        self.db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="life_process_spanning_chunk",
                text="Chlorophyll bridge evidence belongs to this learning unit.",
                page_start=100,
                page_end=200,
                lexical_terms=["chlorophyll", "bridge", "evidence"],
            )
        )
        self.db.commit()
        scope = {
            "subject": "Science",
            "chapter": chapter.slug,
            "topic": "Nutrition",
            "concept_ids": ["life_process_2"],
            "catalog_source": "published",
        }

        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting), patch(
            "Logic.content_pipeline.embeddings_service.embed_query", return_value=None
        ):
            result = search_approved_content(
                "nutrition_unit",
                "Explain the chlorophyll bridge evidence",
                scope=scope,
                max_chars=50000,
                limit=10,
            )

        self.assertIn("life_process_spanning_chunk", result["matched_sections"])

    def test_group_retrieval_ranks_query_match_before_member_order(self):
        chapter = self._published_chapter(4)
        first = self.db.query(ContentConcept).filter(
            ContentConcept.chapter_id == chapter.id,
            ContentConcept.concept_id == "life_process_1",
        ).one()
        target = self.db.query(ContentConcept).filter(
            ContentConcept.chapter_id == chapter.id,
            ContentConcept.concept_id == "life_process_4",
        ).one()
        first.definition = "Unrelated material. " * 400
        target.definition = "Chlorophyll bridge evidence"
        self.db.commit()
        scope = {
            "subject": "Science",
            "chapter": chapter.slug,
            "topic": "Plant processes",
            "concept_ids": [
                "life_process_1",
                "life_process_2",
                "life_process_3",
                "life_process_4",
            ],
            "catalog_source": "published",
        }

        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting), patch(
            "Logic.content_pipeline.embeddings_service.embed_query", return_value=None
        ):
            result = search_approved_content(
                "unit_life_processes",
                "Explain chlorophyll bridge evidence",
                scope=scope,
                max_chars=300,
                limit=6,
            )

        self.assertEqual(result["matched_sections"][0], "life_process_4")
        self.assertIn("Chlorophyll bridge evidence", result["context"])

    def test_complex_unit_fits_normal_study_retrieval_limit(self):
        chapter = self._published_chapter(77)
        topics = build_catalog(self.db)["subjects"][0]["chapters"][0]["topics"]
        unit = max(topics, key=lambda item: len(item["concept_ids"]))
        resolved = resolve_catalog_topic(
            self.db,
            unit["id"],
            subject="Science",
            chapter=chapter.slug,
            topic=unit["label"],
        )

        self.assertLessEqual(len(resolved["concept_ids"]), 8)
        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting), patch(
            "Logic.content_pipeline.embeddings_service.embed_query", return_value=None
        ):
            result = search_knowledge_base(
                resolved["section_id"],
                resolved["topic"],
                # Tutor calls can request only five paragraphs; a published
                # grouped unit must still keep all of its scoped members in
                # the candidate/result limit.
                max_paragraphs=5,
                scope=resolved,
            )

        self.assertCountEqual(result["matched_sections"], resolved["concept_ids"])


if __name__ == "__main__":
    unittest.main()
