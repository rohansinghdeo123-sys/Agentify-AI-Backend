import json
import os
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from models import ContentChapter, ContentChunk, ContentConcept, ContentIngestionJob, ContentPage
from routers import health as health_router
from services import content_release_readiness as release_readiness


class ContentReleaseReadinessTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _seed_content(self):
        chapter = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            chapter_number=1,
            chapter_name="Test chapter",
            slug="test_chapter",
            status="published",
        )
        self.db.add(chapter)
        self.db.flush()
        self.db.add(ContentPage(chapter_id=chapter.id, page_number=1, text="Source"))
        self.db.add(ContentConcept(chapter_id=chapter.id, concept_id="concept", title="Concept"))
        self.db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="chunk",
                text="Source",
                page_start=1,
                page_end=1,
                embedding=[1.0],
            )
        )
        self.db.commit()

    def _small_release(self):
        return patch.multiple(
            release_readiness,
            RELEASE_CHAPTER_NUMBERS=(1,),
            RELEASE_EXPECTED_INVENTORY={
                "chapters": 1,
                "pages": 1,
                "concepts": 1,
                "chunks": 1,
                "embedded_chunks": 1,
            },
        )

    def test_exact_published_inventory_and_release_provenance_are_ready(self):
        self._seed_content()
        digest = "sha256:" + "a" * 64
        self.db.add(
            ContentIngestionJob(
                job_id="content_release_test",
                job_type="content_release_restore",
                status="completed",
                summary={
                    "digest": digest,
                    "scope": {
                        "board": "NCERT",
                        "class_level": "11",
                        "subject": "Chemistry",
                        "chapter_numbers": [1],
                    },
                    "inventory": {"chapters": 1, "pages": 1, "concepts": 1, "chunks": 1},
                    "last_result": {"verified_at": "2026-08-30T08:00:00+00:00"},
                },
            )
        )
        # A newer release for another subject must not replace Chemistry's
        # provenance in the public Chemistry readiness signal.
        self.db.add(
            ContentIngestionJob(
                job_id="content_release_newer_maths",
                job_type="content_release_restore",
                status="completed",
                summary={
                    "digest": "sha256:" + "b" * 64,
                    "scope": {
                        "board": "NCERT",
                        "class_level": "11",
                        "subject": "Maths",
                        "chapter_numbers": [1],
                    },
                    "inventory": {"chapters": 1, "pages": 1, "concepts": 1, "chunks": 1},
                },
            )
        )
        self.db.commit()

        with self._small_release():
            result = release_readiness.content_release_readiness(self.db)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["published"], result["expected"])
        self.assertEqual(result["release"]["provenance"], "verified")
        self.assertEqual(result["release"]["digest"], digest)
        self.assertEqual(result["release"]["restored_at"], "2026-08-30T08:00:00+00:00")

    def test_exact_content_without_release_record_is_explicitly_unverified(self):
        self._seed_content()
        with self._small_release():
            result = release_readiness.content_release_readiness(self.db)

        self.assertEqual(result["status"], "content_ready_unverified_provenance")
        self.assertEqual(result["release"]["provenance"], "missing")
        self.assertEqual(result["release"]["digest"], "")

    def test_incomplete_inventory_is_reported_without_content(self):
        with self._small_release():
            result = release_readiness.content_release_readiness(self.db)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["published"]["chapters"], 0)
        self.assertEqual(
            set(result),
            {"status", "scope", "expected", "published", "release"},
        )


class PublicHealthReleaseSignalTests(unittest.TestCase):
    def tearDown(self):
        health_router._database_readiness_cache.clear()

    def test_readiness_exposes_only_safe_release_summary_without_blocking_api(self):
        db = MagicMock()
        release_signal = release_readiness.unavailable_release_readiness("incomplete")
        semantic_signal = {
            "status": "ready",
            "configured": True,
            "configured_model": "model",
            "configured_endpoint_host": "embeddings.example",
            "stored_model": "model",
            "stored_endpoint_host": "embeddings.example",
            "stored_dimensions": 3,
        }
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "configured"}, clear=False),
            patch.object(health_router, "check_db_health", return_value=True),
            patch.object(health_router.security, "firebase_ready", return_value=True),
            patch.object(health_router.knowledge_graph, "list_chapters", return_value=["chapter"]),
            patch.object(health_router, "available_artifact_sections", return_value=["section"]),
            patch.object(health_router, "SessionLocal", return_value=db),
            patch.object(health_router, "retrieval_embedding_status", return_value=semantic_signal),
            patch.object(health_router, "content_release_readiness", return_value=release_signal),
        ):
            health_router._database_readiness_cache.clear()
            response = health_router.readiness_probe()

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["chemistry_release"], release_signal)
        self.assertNotIn("source_path", json.dumps(payload))
        self.assertNotIn("summary", payload["chemistry_release"])
        db.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
