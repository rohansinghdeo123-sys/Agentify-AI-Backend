import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.request_models import SectionAIRequest
from app.serializers import serialize_coach_conversation
from database import Base
from Logic.content_pipeline import search_approved_content
from Logic.agents.coach_agent import _selected_material_scope
from Logic.coach.retriever import GroundedRetriever
from Logic.tools.knowledge_search import search_knowledge_base
from models import AICoachInteraction, ContentChapter, ContentChunk, ContentConcept
from routers.study import section_ai
from services.catalog_service import resolve_catalog_topic


NCERT_CHAPTER_SLUG = (
    "ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry"
)


class PublishedRevisionRetrievalTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        self.SessionTesting = sessionmaker(bind=engine)
        self.db = self.SessionTesting()

        chapter = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            chapter_number=1,
            chapter_name="Some Basic Concepts Of Chemistry",
            slug=NCERT_CHAPTER_SLUG,
            status="published",
            version="v3",
        )
        self.db.add(chapter)
        self.db.flush()
        self.chapter_id = chapter.id
        self.db.add(
            ContentConcept(
                chapter_id=chapter.id,
                concept_id="atomic_mass",
                title="Atomic Mass of an Element",
                definition=(
                    "Atomic mass expresses the mass of an atom relative to one twelfth "
                    "of the mass of a carbon-12 atom. "
                    + "This published explanation includes isotope context and worked detail. "
                    * 100
                ),
                core_explanation="Atomic masses are relative values expressed in unified mass units.",
                source_pages=[16, 17],
            )
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_catalog_topic_resolves_by_id_and_display_title(self):
        common_scope = {
            "subject": "Chemistry",
            "chapter": "Some Basic Concepts of Chemistry",
            # Profile values use "Class 11" while ingested NCERT rows use "11".
            "class_level": "Class 11",
        }

        by_id = resolve_catalog_topic(self.db, "atomic_mass", **common_scope)
        by_title = resolve_catalog_topic(
            self.db,
            "Atomic Mass of an Element",
            topic="Atomic Mass of an Element",
            **common_scope,
        )

        for resolved in (by_id, by_title):
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved["section_id"], "atomic_mass")
            self.assertEqual(resolved["topic"], "Atomic Mass of an Element")
            self.assertEqual(resolved["chapter_slug"], NCERT_CHAPTER_SLUG)
            self.assertEqual(resolved["content_version"], "v3")

    def test_exact_atomic_mass_match_is_truncated_not_discarded(self):
        scope = {
            "section_id": "atomic_mass",
            "subject": "Chemistry",
            "chapter": "Some Basic Concepts of Chemistry",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "topic": "Atomic Mass of an Element",
            "class_level": "Class 11",
            "content_version": "v3",
        }

        with (
            patch("Logic.content_pipeline.SessionLocal", self.SessionTesting),
            patch("Logic.content_pipeline.embeddings_service.embed_query", return_value=None),
        ):
            result = search_approved_content(
                "Atomic Mass of an Element",
                "Explain atomic mass clearly",
                scope=scope,
                max_chars=4000,
            )

        self.assertEqual(result["source"], "approved_content_pipeline")
        self.assertIn("Atomic Mass of an Element", result["context"])
        self.assertIn("Atomic mass expresses", result["context"])
        self.assertLessEqual(len(result["context"]), 4000)
        self.assertEqual(result["matched_sections"], ["atomic_mass"])

    def test_exact_topic_precedes_a_high_scoring_semantic_distractor(self):
        self.db.add(
            ContentChunk(
                chapter_id=self.chapter_id,
                chunk_id="semantic-distractor",
                text=("Atomic mass practice distractor. " * 300),
                page_start=99,
                page_end=99,
                section_title="Unrelated practice",
                lexical_terms=["atomic", "mass"],
                embedding=[1.0, 0.0],
            )
        )
        self.db.commit()
        scope = {
            "section_id": "atomic_mass",
            "subject": "Chemistry",
            "chapter": "Some Basic Concepts of Chemistry",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "topic": "Atomic Mass of an Element",
            "class_level": "11",
            "content_version": "v3",
            "catalog_source": "published",
        }

        with (
            patch("Logic.content_pipeline.SessionLocal", self.SessionTesting),
            patch(
                "Logic.content_pipeline.embeddings_service.embed_query",
                return_value=[1.0, 0.0],
            ),
            patch(
                "Logic.content_pipeline.embeddings_service.similarity",
                return_value=0.99,
            ),
        ):
            result = search_approved_content(
                "atomic_mass",
                "Explain atomic mass clearly",
                scope=scope,
                max_chars=700,
            )

        self.assertEqual(result["matched_sections"], ["atomic_mass"])
        self.assertIn("Atomic Mass of an Element", result["context"])
        self.assertNotIn("Unrelated practice", result["context"])

    def test_profile_other_does_not_hide_a_unique_published_topic(self):
        resolved = resolve_catalog_topic(
            self.db,
            "atomic_mass",
            subject="Chemistry",
            chapter="Some Basic Concepts of Chemistry",
            topic="Atomic Mass of an Element",
            class_level="Other",
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["section_id"], "atomic_mass")
        self.assertEqual(resolved["class_level"], "11")

    def test_duplicate_unscoped_topic_is_not_resolved_arbitrarily(self):
        self.db.add(
            ContentConcept(
                chapter_id=self.chapter_id,
                concept_id="properties",
                title="Properties",
                definition="Properties in chapter one.",
            )
        )
        second_chapter = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            chapter_number=2,
            chapter_name="Structure of Atom",
            slug="ncert_class_11_chemistry_chapter_2_structure_of_atom",
            status="published",
            version="v1",
        )
        self.db.add(second_chapter)
        self.db.flush()
        self.db.add(
            ContentConcept(
                chapter_id=second_chapter.id,
                concept_id="properties",
                title="Properties",
                definition="Properties in chapter two.",
            )
        )
        self.db.commit()

        self.assertIsNone(resolve_catalog_topic(self.db, "properties"))

    def test_published_scope_never_falls_back_to_bundled_material(self):
        scope = {
            "catalog_source": "published",
            "chapter_slug": "missing",
            "class_level": "12",
            "content_version": "v9",
        }
        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting):
            result = search_knowledge_base(
                "alkanes",
                "Explain alkanes",
                scope=scope,
            )

        self.assertEqual(result["context"], "")
        self.assertEqual(result["paragraphs_found"], 0)
        self.assertEqual(result["error"], "material_not_found")
        self.assertNotIn(result.get("source"), {"markdown", "knowledge_graph"})

    def test_tutor_keeps_published_scope_on_initial_and_retry_searches(self):
        import Logic.agents.tutor_agent as tutor

        scope = {
            "catalog_source": "published",
            "section_id": "atomic_mass",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "class_level": "11",
            "content_version": "v3",
        }
        request = SimpleNamespace(
            question="Explain atomic mass.",
            section_id="atomic_mass",
            session_id="scope-regression",
            difficulty="medium",
            content_scope=scope,
        )
        retrieval = {
            "context": "Published atomic mass context.",
            "section_id": "atomic_mass",
            "paragraphs_found": 1,
            "keywords_used": ["atomic", "mass"],
            "basics_context": "",
            "source": "approved_content_pipeline",
        }

        with (
            patch.object(tutor, "search_knowledge_base", side_effect=[retrieval, retrieval]) as search,
            patch.object(tutor, "knowledge_graph") as graph,
            patch.object(tutor, "event_bus"),
            patch.object(tutor, "_load_session_memory", return_value=[]),
            patch.object(tutor, "_save_turn"),
            patch.object(tutor.model_gateway, "complete", side_effect=["First answer", "Retry answer"]),
            patch.object(
                tutor,
                "evaluate_answer_quality",
                side_effect=[
                    {"passed": False, "score": 0.2},
                    {"passed": True, "score": 0.9},
                ],
            ),
        ):
            graph.concepts = {}
            result = tutor.tutor_agent(request)

        self.assertEqual(result["answer"], "Retry answer")
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args_list[0].kwargs["scope"], scope)
        self.assertEqual(search.call_args_list[1].kwargs["scope"], scope)

    def test_revision_published_prompt_excludes_the_legacy_graph(self):
        import Logic.agents.revision_agent as revision

        scope = {
            "catalog_source": "published",
            "section_id": "atomic_mass",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "class_level": "11",
            "content_version": "v3",
        }
        request = SimpleNamespace(
            question="Revise atomic mass.",
            section_id="atomic_mass",
            content_scope=scope,
        )
        retrieval = {
            "context": "CURRENT PUBLISHED ATOMIC MASS",
            "section_id": "atomic_mass",
            "paragraphs_found": 1,
            "keywords_used": ["atomic", "mass"],
            "basics_context": "",
            "source": "approved_content_pipeline",
        }

        with (
            patch.object(revision, "search_knowledge_base", return_value=retrieval),
            patch.object(revision, "knowledge_graph") as graph,
            patch.object(revision, "event_bus"),
            patch.object(revision.model_gateway, "complete", return_value="Grounded revision") as complete,
            patch.object(
                revision,
                "evaluate_answer_quality",
                return_value={"passed": True, "score": 0.95},
            ),
        ):
            graph.concepts = {"atomic_mass": {"definition": "STALE GRAPH CONTENT"}}
            graph.get_concept.return_value = {
                "concept_id": "atomic_mass",
                "title": "Atomic Mass",
                "definition": "STALE GRAPH CONTENT",
                "core_explanation": "STALE GRAPH CONTENT",
            }
            result = revision.revision_agent(request)

        prompt = complete.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(result["answer"], "Grounded revision")
        self.assertIn("CURRENT PUBLISHED ATOMIC MASS", prompt)
        self.assertNotIn("STALE GRAPH CONTENT", prompt)
        self.assertEqual(result["metadata"]["concepts_from_graph"], 0)

    def test_published_revision_uses_grounded_source_when_model_is_unavailable(self):
        import Logic.agents.revision_agent as revision

        scope = {
            "catalog_source": "published",
            "section_id": "atomic_mass",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "class_level": "11",
            "content_version": "v3",
        }
        request = SimpleNamespace(
            question="Revise atomic mass.",
            section_id="atomic_mass",
            content_scope=scope,
            required_not_found_response="material missing",
        )
        retrieval = {
            "context": (
                "## Atomic Mass of an Element\n"
                "Source: NCERT Class 11 Chemistry, Some Basic Concepts of Chemistry, "
                "page(s): 16, type: concept\n"
                "Atomic mass expresses the mass of an atom relative to carbon-12.\n"
                "Atomic masses are relative values expressed in unified mass units."
            ),
            "section_id": "atomic_mass",
            "paragraphs_found": 1,
            "keywords_used": ["atomic", "mass"],
            "basics_context": "",
            "source": "approved_content_pipeline",
        }

        with (
            patch.object(revision, "search_knowledge_base", return_value=retrieval),
            patch.object(revision, "event_bus") as events,
            patch.object(
                revision.model_gateway,
                "complete",
                side_effect=RuntimeError("provider temporarily unavailable"),
            ),
        ):
            result = revision.revision_agent(request, revision_type="explain")

        self.assertEqual(result["metadata"]["status"], "grounded_fallback")
        self.assertIn("Atomic Mass of an Element", result["answer"])
        self.assertIn("relative to carbon-12", result["answer"])
        self.assertIn("NCERT Class 11 Chemistry", result["answer"])
        self.assertNotIn("AI service encountered an error", result["answer"])
        completion_statuses = [
            call.args[2].get("status")
            for call in events.emit.call_args_list
            if len(call.args) >= 3 and call.args[1] == "task_complete"
        ]
        self.assertIn("degraded", completion_statuses)
        self.assertNotIn("failed", completion_statuses)

    def test_low_quality_published_revision_does_not_make_a_second_model_call(self):
        import Logic.agents.revision_agent as revision

        scope = {
            "catalog_source": "published",
            "section_id": "atomic_mass",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "class_level": "11",
            "content_version": "v3",
        }
        request = SimpleNamespace(
            question="Revise atomic mass.",
            section_id="atomic_mass",
            content_scope=scope,
        )
        retrieval = {
            "context": "Current approved atomic mass explanation.",
            "section_id": "atomic_mass",
            "paragraphs_found": 1,
            "keywords_used": ["atomic", "mass"],
            "basics_context": "",
            "source": "approved_content_pipeline",
        }

        with (
            patch.object(revision, "search_knowledge_base", return_value=retrieval) as search,
            patch.object(revision, "knowledge_graph") as graph,
            patch.object(revision, "event_bus"),
            patch.object(revision.model_gateway, "complete", return_value="Usable grounded answer") as complete,
            patch.object(
                revision,
                "evaluate_answer_quality",
                return_value={"passed": False, "score": 0.45},
            ),
        ):
            graph.concepts = {}
            result = revision.revision_agent(request, revision_type="explain")

        self.assertEqual(result["answer"], "Usable grounded answer")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(complete.call_count, 1)

    def test_conversation_serialization_preserves_syllabus_scope(self):
        learning_context = {
            "scope": "selected_study_material_only",
            "catalog_source": "published",
            "selected_subject": "Chemistry",
            "selected_chapter_id": NCERT_CHAPTER_SLUG,
            "selected_chapter": "Some Basic Concepts of Chemistry",
            "section_id": "atomic_mass",
            "selected_topic": "Atomic Mass of an Element",
        }
        rows = [
            AICoachInteraction(
                id=1,
                user_id="student-1",
                role="user",
                message="Explain atomic mass.",
                metadata_json={"session_id": "coach-student-1-study-1", "learning_context": learning_context},
                created_at=datetime(2026, 7, 29, 10, 0, 0),
            ),
            AICoachInteraction(
                id=2,
                user_id="student-1",
                role="assistant",
                message="Grounded answer.",
                metadata_json={"session_id": "coach-student-1-study-1", "learning_context": learning_context},
                created_at=datetime(2026, 7, 29, 10, 1, 0),
            ),
        ]

        payload = serialize_coach_conversation("coach-student-1-study-1", rows)

        self.assertEqual(payload["scope"]["source"], "syllabus")
        self.assertEqual(payload["scope"]["catalogSource"], "published")
        self.assertEqual(payload["scope"]["chapterId"], NCERT_CHAPTER_SLUG)
        self.assertEqual(payload["scope"]["topicId"], "atomic_mass")

    def test_study_coach_carries_published_scope_and_ignores_other_class(self):
        request = SimpleNamespace(
            subject="Chemistry",
            chapter="Some Basic Concepts of Chemistry",
            topic="Atomic Mass of an Element",
            section_id="atomic_mass",
        )
        adaptive_context = {
            "learning_context": {
                "scope": "selected_study_material_only",
                "catalog_source": "published",
                "selected_chapter_id": NCERT_CHAPTER_SLUG,
                "class_level": "Other",
            }
        }

        scope = _selected_material_scope(request, adaptive_context)

        self.assertEqual(scope["catalog_source"], "published")
        self.assertEqual(scope["chapter_slug"], NCERT_CHAPTER_SLUG)
        self.assertEqual(scope["class_level"], "")

    def test_study_retriever_preserves_scope_on_approved_fallback_boundary(self):
        scope = {
            "catalog_source": "published",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "section_id": "atomic_mass",
        }
        empty = {"context": "", "source": "content_pipeline", "paragraphs_found": 0}
        missing = {
            "context": "",
            "source": "approved_content_pipeline",
            "paragraphs_found": 0,
            "error": "material_not_found",
        }

        with (
            patch("Logic.coach.retriever.search_approved_content", return_value=empty),
            patch("Logic.coach.retriever.search_knowledge_base", return_value=missing) as search,
        ):
            result = GroundedRetriever()._retrieve_section(
                "atomic_mass",
                "Explain atomic mass",
                scope,
            )

        self.assertFalse(result.supported)
        self.assertEqual(result.error, "material_not_found")
        self.assertEqual(search.call_args.kwargs["scope"], scope)

    def test_study_retriever_preserves_semantic_support_evidence(self):
        scope = {
            "catalog_source": "published",
            "chapter_slug": NCERT_CHAPTER_SLUG,
            "section_id": "atomic_mass",
            "concept_ids": ["atomic_mass"],
        }
        approved = {
            "context": "Published explanation phrased differently from the doubt.",
            "source": "approved_content_pipeline",
            "section_id": "atomic_mass",
            "paragraphs_found": 1,
            "matched_sections": ["atomic_mass"],
            "retrieval_mode": "hybrid",
            "semantic_matches": 2,
            "best_semantic_similarity": 0.41,
        }

        with patch(
            "Logic.coach.retriever.search_approved_content",
            return_value=approved,
        ):
            result = GroundedRetriever()._retrieve_section(
                "atomic_mass",
                "Why are relative values used?",
                scope,
            )

        self.assertTrue(result.supported)
        self.assertEqual(result.matched_sections, ["atomic_mass"])
        self.assertEqual(result.retrieval_mode, "hybrid")
        self.assertEqual(result.semantic_matches, 2)
        self.assertEqual(result.best_semantic_similarity, 0.41)

    def test_section_ai_threads_resolved_scope_to_revision(self):
        request = SectionAIRequest(
            question="Explain atomic mass with clear notes.",
            section_id="atomic_mass",
            session_id="revision-student-1-atomic-mass-summary",
            mode="summary",
            subject="Chemistry",
            chapter="Some Basic Concepts of Chemistry",
            topic="Atomic Mass of an Element",
            strict_grounding=True,
        )

        with (
            patch("routers.study.enforce_user_quota"),
            patch(
                "routers.study.profile_learning_context",
                return_value={"class_level": "Class 11"},
            ),
            patch("routers.study.section_doubt", return_value="Grounded revision") as doubt,
        ):
            response = section_ai(
                request,
                db=self.db,
                current_user={"uid": "student-1", "email": "student@example.com"},
            )

        self.assertEqual(response, {"answer": "Grounded revision"})
        call = doubt.call_args.kwargs
        self.assertEqual(call["section_id"], "atomic_mass")
        self.assertTrue(call["strict_grounding"])
        self.assertEqual(call["content_scope"]["subject"], "Chemistry")
        self.assertEqual(
            call["content_scope"]["chapter_slug"], NCERT_CHAPTER_SLUG
        )
        self.assertEqual(call["content_scope"]["topic"], "Atomic Mass of an Element")
        self.assertEqual(call["content_scope"]["class_level"], "11")
        self.assertEqual(call["content_scope"]["content_version"], "v3")
        self.assertEqual(call["class_level"], "11")

    def test_section_ai_returns_atomic_mass_revision_when_provider_is_down(self):
        """Exercise route, profile scope, retrieval, agent routing and fallback."""
        import Logic.agents.revision_agent as revision

        request = SectionAIRequest(
            question="Teach Atomic Mass of an Element thoroughly.",
            section_id="atomic_mass",
            session_id="revision-student-1-atomic-mass-explain",
            mode="explain",
            subject="Chemistry",
            chapter="Some Basic Concepts of Chemistry",
            topic="Atomic Mass of an Element",
            strict_grounding=True,
            retrieval_required=True,
        )

        with (
            patch("routers.study.enforce_user_quota"),
            patch(
                "routers.study.profile_learning_context",
                return_value={"class_level": "Other"},
            ),
            patch("Logic.content_pipeline.SessionLocal", self.SessionTesting),
            patch("Logic.content_pipeline.embeddings_service.embed_query", return_value=None),
            patch.object(revision, "event_bus"),
            patch.object(
                revision.model_gateway,
                "complete",
                side_effect=RuntimeError("provider temporarily unavailable"),
            ),
        ):
            response = section_ai(
                request,
                db=self.db,
                current_user={"uid": "student-1", "email": "student@example.com"},
            )

        self.assertIn("Atomic Mass of an Element", response["answer"])
        self.assertIn("Atomic mass expresses", response["answer"])
        self.assertNotIn("could not find this", response["answer"].lower())


if __name__ == "__main__":
    unittest.main()
