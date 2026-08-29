"""Safety checks for the production content-automation CLI preflight."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import automate_content


class PreflightTests(unittest.TestCase):
    def test_exact_chapter_parser_deduplicates_and_validates(self):
        self.assertEqual(automate_content._chapter_csv("9,3,3"), [3, 9])
        with self.assertRaisesRegex(Exception, "positive integers"):
            automate_content._chapter_csv("3,zero")

    def test_download_only_needs_no_database_or_ai_configuration(self):
        result = automate_content.check_preflight(env={}, download_only=True)

        self.assertTrue(result.ok)
        self.assertEqual(result.database_target, "not required (download-only)")
        self.assertFalse(result.generation_enabled)
        self.assertFalse(result.semantic_embeddings_enabled)
        self.assertEqual(result.errors, ())

    def test_full_run_reports_every_missing_requirement(self):
        result = automate_content.check_preflight(env={})

        self.assertFalse(result.ok)
        self.assertEqual(len(result.errors), 3)
        self.assertTrue(any("DATABASE_URL" in error for error in result.errors))
        self.assertTrue(any("GROQ_API_KEY" in error for error in result.errors))
        self.assertTrue(any("EMBEDDINGS_API_KEY" in error for error in result.errors))

    def test_postgres_with_generation_and_embeddings_is_ready_and_sanitized(self):
        result = automate_content.check_preflight(
            env={
                "DATABASE_URL": "postgresql://app:super-secret@db.example.com:5432/agentify?sslmode=require",
                "GROQ_API_KEY": "groq-secret",
                "OPENAI_API_KEY": "embedding-secret",
            }
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.database_target, "postgresql @ db.example.com")
        rendered = " ".join((result.database_target, *result.errors))
        self.assertNotIn("super-secret", rendered)
        self.assertNotIn("agentify", rendered)
        self.assertNotIn("groq-secret", rendered)
        self.assertNotIn("embedding-secret", rendered)

    def test_openrouter_review_route_does_not_require_groq(self):
        result = automate_content.check_preflight(
            env={
                "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
                "COACH_PROVIDER_ORDER": "openrouter",
                "OPENROUTER_API_KEY": "configured",
                "OPENROUTER_REVIEW_MODEL": "provider/reviewer-model",
                "EMBEDDINGS_API_KEY": "configured",
            }
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.generation_enabled)
        self.assertFalse(any("GROQ_API_KEY" in error for error in result.errors))

    def test_openai_review_route_does_not_require_groq(self):
        result = automate_content.check_preflight(
            env={
                "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
                "COACH_LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "configured",
                "OPENAI_REVIEW_MODEL": "reviewer-model",
            }
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.generation_enabled)
        self.assertTrue(result.semantic_embeddings_enabled)

    def test_openai_provider_without_a_reviewer_or_fallback_model_is_blocked(self):
        result = automate_content.check_preflight(
            env={
                "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
                "COACH_PROVIDER_ORDER": "openai",
                "OPENAI_API_KEY": "configured",
            }
        )

        self.assertFalse(result.ok)
        self.assertFalse(result.generation_enabled)
        self.assertTrue(any("REVIEW_MODEL" in error for error in result.errors))

    def test_sqlite_requires_explicit_local_opt_in(self):
        env = {
            "DATABASE_URL": "sqlite:///private/local-file.db",
            "GROQ_API_KEY": "configured",
            "EMBEDDINGS_API_KEY": "configured",
        }

        blocked = automate_content.check_preflight(env=env)
        allowed = automate_content.check_preflight(env=env, allow_local_db=True)

        self.assertFalse(blocked.ok)
        self.assertTrue(any("--allow-local-db" in error for error in blocked.errors))
        self.assertTrue(allowed.ok)
        self.assertEqual(allowed.database_target, "sqlite (local)")

    def test_missing_database_can_use_explicit_local_opt_in(self):
        result = automate_content.check_preflight(
            env={
                "GROQ_API_KEY": "configured",
                "EMBEDDINGS_API_KEY": "configured",
            },
            allow_local_db=True,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.database_target, "sqlite (implicit local opt-in)")

    def test_missing_embedding_key_requires_explicit_lexical_opt_in(self):
        env = {
            "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
            "GROQ_API_KEY": "configured",
        }

        blocked = automate_content.check_preflight(env=env)
        allowed = automate_content.check_preflight(env=env, allow_lexical_only=True)

        self.assertFalse(blocked.ok)
        self.assertTrue(allowed.ok)
        self.assertTrue(allowed.lexical_only)
        self.assertFalse(allowed.semantic_embeddings_enabled)


class MainPreflightTests(unittest.TestCase):
    def test_blocked_preflight_exits_before_loading_or_running_automation(self):
        with patch.dict(automate_content.os.environ, {}, clear=True), \
             patch.object(automate_content, "_load_automation") as loader:
            result = automate_content.main(["--classes", "11", "--subjects", "Chemistry"])

        self.assertEqual(result, 2)
        loader.assert_not_called()

    def test_preflight_only_never_loads_or_runs_automation(self):
        env = {
            "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
            "GROQ_API_KEY": "configured",
            "EMBEDDINGS_API_KEY": "configured",
        }
        with patch.dict(automate_content.os.environ, env, clear=True), \
             patch.object(automate_content, "_load_automation") as loader:
            result = automate_content.main(["--preflight-only"])

        self.assertEqual(result, 0)
        loader.assert_not_called()

    def test_download_only_runs_without_configuration(self):
        summary = {
            "downloaded": 1,
            "published": 0,
            "needs_review": 0,
            "skipped": 0,
            "failed": [],
            "chapters": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "download.json"
            fake_run = unittest.mock.Mock(return_value=summary)
            with patch.dict(automate_content.os.environ, {}, clear=True), \
                 patch.object(automate_content, "_load_automation", return_value=fake_run) as loader:
                result = automate_content.main([
                    "--download-only",
                    "--classes", "11",
                    "--subjects", "Chemistry",
                    "--out", str(output),
                ])

            self.assertEqual(result, 0)
            loader.assert_called_once_with(allow_local_db=True)
            fake_run.assert_called_once()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["downloaded"], 1)

    def test_reingest_flag_forces_fresh_pdf_extraction(self):
        summary = {
            "downloaded": 0,
            "published": 0,
            "needs_review": 1,
            "skipped": 0,
            "failed": [],
            "chapters": [],
        }
        env = {
            "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
            "GROQ_API_KEY": "configured",
            "EMBEDDINGS_API_KEY": "configured",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reingest.json"
            fake_run = unittest.mock.Mock(return_value=summary)
            with patch.dict(automate_content.os.environ, env, clear=True), patch.object(
                automate_content,
                "_load_automation",
                return_value=fake_run,
            ):
                result = automate_content.main([
                    "--classes", "11",
                    "--subjects", "Chemistry",
                    "--chapters", "8",
                    "--reingest",
                    "--out", str(output),
                ])

        self.assertEqual(result, 0)
        self.assertFalse(fake_run.call_args.kwargs["reuse_ingest"])
        self.assertFalse(fake_run.call_args.kwargs["skip_completed"])

    def test_failed_chapter_summary_returns_nonzero_exit_status(self):
        summary = {
            "downloaded": 0,
            "published": 0,
            "needs_review": 0,
            "skipped": 0,
            "failed": [{"file": "chapter_05.pdf", "error": "provider quota"}],
            "chapters": [],
        }
        env = {
            "DATABASE_URL": "postgresql://app:secret@db.example.com/agentify",
            "GROQ_API_KEY": "configured",
            "EMBEDDINGS_API_KEY": "configured",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed.json"
            fake_run = unittest.mock.Mock(return_value=summary)
            with patch.dict(automate_content.os.environ, env, clear=True), patch.object(
                automate_content,
                "_load_automation",
                return_value=fake_run,
            ):
                result = automate_content.main([
                    "--classes", "11",
                    "--subjects", "Chemistry",
                    "--out", str(output),
                ])
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 1)
        self.assertEqual(len(written["failed"]), 1)


if __name__ == "__main__":
    unittest.main()
