import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from models import ContentChapter, ContentChunk, ContentConcept, ContentIngestionJob, ContentPage
from services.content_release_bundle import (
    ContentReleaseError,
    export_content_release,
    restore_content_release,
    verify_content_release,
)


class ContentReleaseBundleTests(unittest.TestCase):
    def _session(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.addCleanup(engine.dispose)
        db = sessionmaker(bind=engine)()
        self.addCleanup(db.close)
        return db

    @staticmethod
    def _seed(db, *, slug="ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry"):
        chapter = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            book_name="Chemistry Part I",
            chapter_number=1,
            chapter_name="Some Basic Concepts of Chemistry",
            slug=slug,
            pdf_path="local.pdf",
            source_hash="a" * 64,
            published_source_hash="a" * 64,
            status="published",
            version="v1",
            page_count=1,
            extracted_page_count=1,
            chunk_count=1,
            concept_count=1,
            coverage_score=1.0,
            extraction_quality=1.0,
            validation_report={"ready_for_approval": True, "issues": []},
        )
        db.add(chapter)
        db.flush()
        db.add(
            ContentPage(
                chapter_id=chapter.id,
                page_number=1,
                text="Matter",
                char_count=6,
                extraction_quality=1.0,
            )
        )
        db.add(
            ContentConcept(
                chapter_id=chapter.id,
                concept_id="matter",
                title="Matter",
                definition="Matter has mass.",
                source_pages=[1],
                importance_level="essential",
            )
        )
        db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="chapter_1_chunk_1",
                text="Matter has mass.",
                page_start=1,
                page_end=1,
                lexical_terms=["matter", "mass"],
                embedding=[0.6, 0.8, 0.0],
            )
        )
        db.commit()
        return chapter.id

    @staticmethod
    def _export(source, bundle):
        return export_content_release(
            source,
            bundle,
            class_level="11",
            subject="Chemistry",
            chapter_numbers=[1],
            embedding_model="test-embedding-model",
            embedding_dimensions=3,
            embedding_provider="test",
            embedding_endpoint_host="embedding.test.invalid",
        )

    def test_release_round_trip_is_complete_audited_and_idempotent(self):
        source = self._session()
        self._seed(source)
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "chemistry.json.gz"
            exported = self._export(source, bundle)
            self.assertEqual(exported["chapters"], 1)
            self.assertEqual(exported["embedded_chunks"], 1)
            self.assertEqual(verify_content_release(bundle)["digest"], exported["digest"])

            destination = self._session()
            first = restore_content_release(destination, bundle)
            second = restore_content_release(destination, bundle)
            self.assertEqual(len(first["restored"]), 1)
            self.assertFalse(first["skipped"])
            self.assertFalse(second["restored"])
            self.assertEqual(len(second["skipped"]), 1)
            self.assertEqual(destination.query(ContentChapter).count(), 1)
            self.assertEqual(destination.query(ContentPage).count(), 1)
            self.assertEqual(destination.query(ContentConcept).count(), 1)
            self.assertEqual(destination.query(ContentChunk).count(), 1)
            self.assertEqual(destination.query(ContentIngestionJob).count(), 1)
            release_job = destination.query(ContentIngestionJob).one()
            self.assertEqual(len(release_job.summary["initial_result"]["restored"]), 1)
            self.assertEqual(len(release_job.summary["last_result"]["skipped"]), 1)
            chapter = destination.query(ContentChapter).one()
            self.assertTrue(chapter.approved_by.startswith("content-release:"))
            self.assertEqual(
                chapter.validation_report["content_release"]["embedding_model"],
                "test-embedding-model",
            )

    def test_restore_repairs_incomplete_matching_chapter(self):
        source = self._session()
        self._seed(source)
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "chemistry.json.gz"
            self._export(source, bundle)
            destination = self._session()
            restore_content_release(destination, bundle)
            destination.query(ContentChunk).delete()
            destination.commit()
            repaired = restore_content_release(destination, bundle)
            self.assertEqual(len(repaired["restored"]), 1)
            self.assertEqual(destination.query(ContentChunk).count(), 1)

    def test_legacy_slug_is_upgraded_in_place_without_duplicate(self):
        source = self._session()
        self._seed(source)
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "chemistry.json.gz"
            self._export(source, bundle)
            destination = self._session()
            original_id = self._seed(destination, slug="legacy_basic_chemistry")
            restore_content_release(destination, bundle)
            chapters = destination.query(ContentChapter).all()
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].id, original_id)
            self.assertEqual(
                chapters[0].slug,
                "ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry",
            )

    def test_replacing_different_source_keeps_version_monotonic(self):
        source = self._session()
        self._seed(source)
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "chemistry.json.gz"
            self._export(source, bundle)
            destination = self._session()
            self._seed(destination)
            chapter = destination.query(ContentChapter).one()
            chapter.source_hash = "b" * 64
            chapter.published_source_hash = "b" * 64
            chapter.version = "v7"
            destination.commit()
            restore_content_release(destination, bundle)
            self.assertEqual(destination.query(ContentChapter).one().version, "v8")

    def test_tampered_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "bad.json.gz"
            bundle.write_bytes(b"not-a-valid-release")
            with self.assertRaisesRegex(ContentReleaseError, "unreadable"):
                verify_content_release(bundle)


if __name__ == "__main__":
    unittest.main()
