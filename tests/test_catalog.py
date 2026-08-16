import os
import unittest

os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "true")
os.environ["DATABASE_URL"] = ""

from fastapi.testclient import TestClient

import main
import routers.study as study_router
from app.security import verify_firebase_user
from database import SessionLocal
from models import ContentChapter, ContentConcept
from services.catalog_service import build_catalog

SLUG = "catalogtest-chapter"


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        main.app.dependency_overrides.clear()
        cls.client.__exit__(None, None, None)

    def setUp(self):
        self.db = SessionLocal()
        self._cleanup()
        study_router._catalog_cache = type(study_router._catalog_cache)(max_entries=2)
        main.app.dependency_overrides[verify_firebase_user] = lambda: {
            "uid": "catalog-user",
            "email": "catalog@example.com",
        }

    def tearDown(self):
        main.app.dependency_overrides.clear()
        self._cleanup()
        self.db.close()

    def _cleanup(self):
        chapter_ids = [
            row.id
            for row in self.db.query(ContentChapter).filter(ContentChapter.slug.like("catalogtest-%"))
        ]
        if chapter_ids:
            self.db.query(ContentConcept).filter(ContentConcept.chapter_id.in_(chapter_ids)).delete(
                synchronize_session=False
            )
            self.db.query(ContentChapter).filter(ContentChapter.id.in_(chapter_ids)).delete(
                synchronize_session=False
            )
        self.db.commit()

    def test_builtin_fallback_when_nothing_published(self):
        catalog = build_catalog(self.db)
        if catalog["source"] == "builtin":
            subjects = catalog["subjects"]
            self.assertEqual(subjects[0]["subject"], "Chemistry")
            chapter_slugs = {chapter["slug"] for chapter in subjects[0]["chapters"]}
            self.assertIn("hydrocarbon", chapter_slugs)
            topics = subjects[0]["chapters"][0]["topics"]
            self.assertTrue(all({"id", "label"} <= set(topic) for topic in topics))

    def test_published_chapters_replace_builtin(self):
        chapter = ContentChapter(
            slug=SLUG,
            subject="Physics",
            class_level="Class 12",
            chapter_name="Waves",
            chapter_number=3,
            status="published",
        )
        self.db.add(chapter)
        self.db.flush()
        self.db.add(
            ContentConcept(chapter_id=chapter.id, concept_id="wave_motion", title="Wave Motion")
        )
        self.db.commit()

        catalog = build_catalog(self.db)
        self.assertEqual(catalog["source"], "published")
        physics = next(group for group in catalog["subjects"] if group["subject"] == "Physics")
        waves = next(item for item in physics["chapters"] if item["slug"] == SLUG)
        self.assertEqual(
            waves["topics"],
            [{"id": "wave_motion", "label": "Wave Motion", "concept_ids": ["wave_motion"]}],
        )

    def test_chapter_without_concepts_falls_back_to_slug_topic(self):
        self.db.add(
            ContentChapter(slug=SLUG, subject="Physics", chapter_name="Optics", status="approved")
        )
        self.db.commit()
        catalog = build_catalog(self.db)
        physics = next(group for group in catalog["subjects"] if group["subject"] == "Physics")
        optics = next(item for item in physics["chapters"] if item["slug"] == SLUG)
        self.assertEqual(optics["topics"], [{"id": SLUG, "label": "Optics"}])

    def test_endpoint_requires_auth_and_returns_catalog(self):
        response = self.client.get("/catalog")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["source"], {"builtin", "published"})
        self.assertTrue(payload["subjects"])

        main.app.dependency_overrides.clear()
        self.assertEqual(self.client.get("/catalog").status_code, 401)


if __name__ == "__main__":
    unittest.main()
