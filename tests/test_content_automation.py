"""Tests for the NCERT content automation orchestrator (no network, no LLM)."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "true")
os.environ["DATABASE_URL"] = ""

from database import Base
from Logic import content_automation as automation
from models import ContentChapter


class DownloadTests(unittest.TestCase):
    def test_discovers_chapters_and_stops_after_two_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_dl(session, url, dest):
                dest.write_bytes(b"%PDF-1.4" + b"0" * 20000)
                return True

            with patch.object(automation, "_remote_pdf_exists", side_effect=lambda s, c, nn: nn <= 3), \
                 patch.object(automation, "_download_pdf", side_effect=fake_dl):
                paths = automation.download_subject(
                    automation._make_session(), "10", "Science", ["jesc1"],
                    dest_root=root, delay_seconds=0, max_chapters=10,
                )
            self.assertEqual([p.name for p in paths], ["chapter_01.pdf", "chapter_02.pdf", "chapter_03.pdf"])

    def test_class_11_chemistry_has_stable_nine_chapter_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloaded = []

            def fake_dl(session, url, dest):
                downloaded.append((url, dest.name))
                dest.write_bytes(b"%PDF-1.4" + b"0" * 20000)
                return True

            with patch.object(automation, "_download_pdf", side_effect=fake_dl), \
                 patch.object(automation, "_pdf_matches_curriculum_title", return_value=True):
                paths = automation.download_subject(
                    automation._make_session(), "11", "Chemistry", ["kech1", "kech2"],
                    dest_root=root, delay_seconds=0, max_chapters=30,
                )

            self.assertEqual(
                [url.rsplit("/", 1)[-1] for url, _ in downloaded],
                [
                    "kech101.pdf", "kech102.pdf", "kech103.pdf",
                    "kech104.pdf", "kech105.pdf", "kech106.pdf",
                    "kech201.pdf", "kech202.pdf", "kech203.pdf",
                ],
            )
            self.assertEqual(
                [path.name for path in paths],
                [chapter.filename for chapter in automation.NCERT_CURRICULUM[("11", "Chemistry")]],
            )

    def test_reuses_semantic_files_without_duplicate_generic_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject_dir = root / "class_11" / "chemistry"
            subject_dir.mkdir(parents=True)
            first = subject_dir / "chapter_01_some_basic_concepts_of_chemistry.pdf"
            second = subject_dir / "chapter_02_structure_of_atom.pdf"
            for path in (first, second):
                path.write_bytes(b"%PDF-1.4" + b"0" * 20000)
            downloaded = []

            def fake_dl(session, url, dest):
                downloaded.append(dest.name)
                dest.write_bytes(b"%PDF-1.4" + b"0" * 20000)
                return True

            with patch.object(automation, "_download_pdf", side_effect=fake_dl), \
                 patch.object(automation, "_pdf_matches_curriculum_title", return_value=True):
                paths = automation.download_subject(
                    automation._make_session(), "11", "Chemistry", ["kech1", "kech2"],
                    dest_root=root, delay_seconds=0, max_chapters=30,
                )
                downloaded.clear()
                resumed = automation.download_subject(
                    automation._make_session(), "11", "Chemistry", ["kech1", "kech2"],
                    dest_root=root, delay_seconds=0, max_chapters=30,
                )

            self.assertEqual(paths[:2], [first, second])
            self.assertEqual([path.name for path in resumed], [path.name for path in paths])
            self.assertEqual(downloaded, [])
            self.assertFalse((subject_dir / "chapter_01.pdf").exists())
            self.assertFalse((subject_dir / "chapter_02.pdf").exists())

    def test_existing_curriculum_pdf_is_reused_only_when_title_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject_dir = Path(tmp)
            candidate = subject_dir / "chapter_03_wrong_material.pdf"
            candidate.write_bytes(b"%PDF-1.4" + b"0" * 20_000)

            with patch.object(
                automation,
                "_pdf_matches_curriculum_title",
                return_value=False,
            ) as title_check:
                existing = automation._existing_chapter_pdf(
                    subject_dir,
                    3,
                    expected_title="Classification of Elements and Periodicity in Properties",
                )

            self.assertIsNone(existing)
            title_check.assert_called_once_with(
                candidate,
                "Classification of Elements and Periodicity in Properties",
            )

    def test_curriculum_title_check_normalizes_case_and_punctuation(self):
        page = MagicMock()
        page.extract_text.return_value = (
            "UNIT 8\nORGANIC CHEMISTRY – SOME BASIC PRINCIPLES AND TECHNIQUES"
        )
        reader = SimpleNamespace(pages=[page])

        with patch("pypdf.PdfReader", return_value=reader):
            self.assertTrue(
                automation._pdf_matches_curriculum_title(
                    Path("chapter_08.pdf"),
                    "Organic Chemistry - Some Basic Principles and Techniques",
                )
            )

    def test_known_curriculum_fails_clearly_when_a_chapter_cannot_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_dl(session, url, dest):
                if url.endswith("kech104.pdf"):
                    return False
                dest.write_bytes(b"%PDF-1.4" + b"0" * 20000)
                return True

            with patch.object(automation, "_download_pdf", side_effect=fake_dl), patch.object(
                automation,
                "_pdf_matches_curriculum_title",
                return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, r"missing chapter\(s\): 4"):
                    automation.download_subject(
                        automation._make_session(), "11", "Chemistry", ["kech1", "kech2"],
                        dest_root=root, delay_seconds=0, max_chapters=30,
                    )

    def test_new_download_is_rejected_when_curriculum_title_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_dl(session, url, dest):
                dest.write_bytes(b"%PDF-1.4" + b"0" * 20000)
                return True

            def title_matches(path, expected_title):
                return not path.name.startswith("chapter_04_")

            with patch.object(automation, "_download_pdf", side_effect=fake_dl), patch.object(
                automation,
                "_pdf_matches_curriculum_title",
                side_effect=title_matches,
            ):
                with self.assertRaisesRegex(RuntimeError, r"missing chapter\(s\): 4"):
                    automation.download_subject(
                        automation._make_session(),
                        "11",
                        "Chemistry",
                        ["kech1", "kech2"],
                        dest_root=root,
                        delay_seconds=0,
                        max_chapters=30,
                    )

            rejected = (
                root
                / "class_11"
                / "chemistry"
                / "chapter_04_chemical_bonding_and_molecular_structure.pdf"
            )
            self.assertFalse(rejected.exists())


class TitleInferenceTests(unittest.TestCase):
    def test_known_curriculum_title_wins_over_ambiguous_first_page_text(self):
        chapter = SimpleNamespace(
            class_level="11",
            subject="Chemistry",
            chapter_number=4,
        )
        self.assertEqual(
            automation._curriculum_title(chapter),
            "Chemical Bonding and Molecular Structure",
        )

    def test_accepts_long_ncert_title_after_unit_marker(self):
        page = SimpleNamespace(
            text=(
                "A sentence-like epigraph appears before the unit heading.\n"
                "UNIT 8\n"
                "ORGANIC CHEMISTRY - SOME BASIC PRINCIPLES AND TECHNIQUES\n"
                "After studying this unit you will be able to"
            )
        )
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = page
        chapter = SimpleNamespace(id=8)

        self.assertEqual(
            automation._infer_title(db, chapter),
            "Organic Chemistry Some Basic Principles And Techniques",
        )


class ProcessChapterTests(unittest.TestCase):
    def _patches(self, publish_side_effect=None):
        chapter = SimpleNamespace(id=1, status="validated", chapter_name="Chapter 1", slug="ncert_11_chem_1")
        db = MagicMock()
        pub = patch.object(automation, "publish_chapter", side_effect=publish_side_effect)
        return chapter, db, pub

    def test_publishes_when_gate_passes(self):
        chapter, db, pub_patch = self._patches(publish_side_effect=lambda *a, **k: None)
        with patch.object(automation, "_chapter_for_pdf", return_value=(None, {}, "")), \
             patch.object(automation, "ingest_pdf_file", return_value=chapter), \
             patch.object(automation, "_infer_title", return_value=""), \
             patch.object(automation, "generate_concepts_for_chapter"), \
             patch.object(automation, "embed_missing_chunks", return_value={"embedded": 5}), \
             pub_patch, \
             patch.object(automation, "serialize_chapter", return_value={"slug": chapter.slug, "coverage_score": 0.8, "concept_count": 5}):
            result = automation.process_chapter(db, Path("x.pdf"))
        self.assertEqual(result["status"], "published")

    def test_needs_review_when_gate_fails(self):
        chapter, db, pub_patch = self._patches(publish_side_effect=ValueError("not ready for approval"))
        chapter.status = "needs_review"
        with patch.object(automation, "_chapter_for_pdf", return_value=(None, {}, "")), \
             patch.object(automation, "ingest_pdf_file", return_value=chapter), \
             patch.object(automation, "_infer_title", return_value=""), \
             patch.object(automation, "generate_concepts_for_chapter"), \
             patch.object(automation, "embed_missing_chunks", return_value={"embedded": 0}), \
             pub_patch, \
             patch.object(automation, "serialize_chapter", return_value={"slug": chapter.slug, "coverage_score": 0.4, "concept_count": 3}):
            result = automation.process_chapter(db, Path("x.pdf"))
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("not ready", result["publish_error"])

    def test_failed_live_replacement_rolls_back_to_published_version(self):
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        live = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            book_name="NCERT Chemistry",
            chapter_number=1,
            chapter_name="Some Basic Concepts Of Chemistry",
            slug="ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry",
            status="published",
            source_hash="old_hash",
            published_source_hash="old_hash",
            concept_count=8,
        )
        db.add(live)
        db.commit()

        def mutate_live_chapter(*args, **kwargs):
            live.status = "validated"
            live.source_hash = "replacement_hash"

        with patch.object(
            automation,
            "_chapter_for_pdf",
            return_value=(live, {}, "replacement_hash"),
        ), patch.object(
            automation,
            "_existing_ingested_chapter",
            return_value=live,
        ), patch.object(
            automation,
            "generate_concepts_for_chapter",
            side_effect=mutate_live_chapter,
        ), patch.object(
            automation,
            "embed_missing_chunks",
            return_value={"embedded": 5},
        ), patch.object(
            automation,
            "publish_chapter",
            side_effect=ValueError("not ready for approval"),
        ):
            with self.assertRaisesRegex(RuntimeError, "previous live chapter was preserved"):
                automation.process_chapter(db, Path("replacement.pdf"))

        db.expire_all()
        preserved = db.get(ContentChapter, live.id)
        self.assertEqual(preserved.status, "published")
        self.assertEqual(preserved.source_hash, "old_hash")
        self.assertEqual(preserved.published_source_hash, "old_hash")
        db.close()
        engine.dispose()

    def test_live_replacement_requires_atomic_auto_publish(self):
        live = SimpleNamespace(id=1, status="published")
        db = MagicMock()
        with patch.object(
            automation,
            "_chapter_for_pdf",
            return_value=(live, {}, "replacement_hash"),
        ), patch.object(automation, "generate_concepts_for_chapter") as generate:
            with self.assertRaisesRegex(ValueError, "cannot be re-ingested with --no-publish"):
                automation.process_chapter(
                    db,
                    Path("replacement.pdf"),
                    auto_publish=False,
                    reuse_ingest=False,
                )
        generate.assert_not_called()


class OrchestratorTests(unittest.TestCase):
    def test_run_summary_counts_and_isolates_failures(self):
        results = [
            {"status": "published", "chapter": {"slug": "s1", "coverage_score": 0.8, "concept_count": 5}, "publish_error": ""},
            ValueError("boom"),  # this chapter raises -> isolated as failed
            {"status": "needs_review", "chapter": {"slug": "s3", "coverage_score": 0.4, "concept_count": 3}, "publish_error": "not ready"},
        ]

        def fake_process(db, path, **kwargs):
            outcome = results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(automation, "robots_allows", return_value=True), \
             patch.object(automation, "download_subject", return_value=[Path("a.pdf"), Path("b.pdf"), Path("c.pdf")]), \
             patch.object(automation, "process_chapter", side_effect=fake_process):
            summary = automation.run_automation(
                classes=["11"], subjects=["Chemistry"], delay_seconds=0,
                db_factory=lambda: MagicMock(),
            )
        self.assertEqual(summary["downloaded"], 3)
        self.assertEqual(summary["sources_ready"], 3)
        self.assertEqual(summary["reused_sources"], 0)
        self.assertEqual(summary["published"], 1)
        self.assertEqual(summary["needs_review"], 1)
        self.assertEqual(len(summary["failed"]), 1)

    def test_skip_completed_skips_published_chapters(self):
        # A resume must not re-process (and re-spend LLM/embedding quota) on a
        # chapter that is already fully published.
        published = {"status": "published", "chapter": {"slug": "s", "coverage_score": 0.8, "concept_count": 5}, "publish_error": ""}
        with patch.object(automation, "robots_allows", return_value=True), \
             patch.object(automation, "download_subject", return_value=[Path("a.pdf"), Path("b.pdf")]), \
             patch.object(automation, "_already_completed", side_effect=lambda db, p: p.name == "a.pdf"), \
             patch.object(automation, "process_chapter", return_value=published) as proc:
            summary = automation.run_automation(
                classes=["11"], subjects=["Chemistry"], delay_seconds=0,
                db_factory=lambda: MagicMock(),
            )
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(proc.call_count, 1)  # only b.pdf processed

    def test_download_only_skips_processing(self):
        with patch.object(automation, "robots_allows", return_value=True), \
             patch.object(automation, "download_subject", return_value=[Path("a.pdf")]), \
             patch.object(automation, "process_chapter") as proc:
            summary = automation.run_automation(classes=["11"], subjects=["Chemistry"], download_only=True, db_factory=lambda: MagicMock())
        proc.assert_not_called()
        self.assertEqual(summary["downloaded"], 1)
        self.assertEqual(summary["sources_ready"], 1)

    def test_exact_chapter_selection_processes_only_requested_chapter(self):
        result = {
            "status": "validated",
            "chapter": {"slug": "chapter_3", "coverage_score": 0.9, "concept_count": 8},
            "publish_error": "",
        }
        with patch.object(automation, "robots_allows", return_value=True), \
             patch.object(
                 automation,
                 "download_subject",
                 return_value=[Path("chapter_01.pdf"), Path("chapter_03.pdf"), Path("chapter_09.pdf")],
             ), \
             patch.object(automation, "process_chapter", return_value=result) as process:
            summary = automation.run_automation(
                classes=["11"], subjects=["Chemistry"], chapter_numbers=[3],
                delay_seconds=0, auto_publish=False, reuse_ingest=False,
                db_factory=lambda: MagicMock(),
            )

        self.assertEqual(summary["selected_chapters"], [3])
        self.assertEqual(process.call_count, 1)
        self.assertEqual(process.call_args.args[1].name, "chapter_03.pdf")
        self.assertFalse(process.call_args.kwargs["reuse_ingest"])

    def test_forced_reingest_never_skips_an_already_published_chapter(self):
        result = {
            "status": "validated",
            "chapter": {"slug": "chapter_8", "coverage_score": 0.9, "concept_count": 8},
            "publish_error": "",
        }
        with patch.object(automation, "robots_allows", return_value=True), \
             patch.object(
                 automation,
                 "download_subject",
                 return_value=[Path("chapter_08.pdf")],
             ), \
             patch.object(automation, "_already_completed", return_value=True) as completed, \
             patch.object(automation, "process_chapter", return_value=result) as process:
            summary = automation.run_automation(
                classes=["11"],
                subjects=["Chemistry"],
                chapter_numbers=[8],
                delay_seconds=0,
                auto_publish=False,
                skip_completed=True,
                reuse_ingest=False,
                db_factory=lambda: MagicMock(),
            )

        completed.assert_not_called()
        self.assertEqual(process.call_count, 1)
        self.assertEqual(summary["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
