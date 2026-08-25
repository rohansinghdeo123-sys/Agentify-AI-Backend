import os
import unittest

os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "true")
os.environ["DATABASE_URL"] = ""

from fastapi.testclient import TestClient

import main
from app.security import verify_firebase_user

STUDENT = {"uid": "student-1", "email": "student@example.com"}


class RouteSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        main.app.dependency_overrides.clear()
        cls.client.__exit__(None, None, None)

    def tearDown(self):
        main.app.dependency_overrides.clear()

    def _login(self, token=STUDENT):
        main.app.dependency_overrides[verify_firebase_user] = lambda: token

    @staticmethod
    def _planning_portfolio_payload():
        return {
            "class_level": "Class 11",
            "subject": "Chemistry",
            "chapters": [
                {
                    "chapter_ref": "Some Basic Concepts of Chemistry",
                    "chapter_proficiency": "know_a_little",
                },
                {
                    "chapter_ref": "Structure of Atom",
                    "chapter_proficiency": "mostly_confident",
                },
            ],
            "study_time_today": "15",
        }

    def test_health_endpoints_are_public(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_public_pulse_is_public_and_anonymized(self):
        resp = self.client.get("/public/pulse")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("students", "total_xp", "top_streak", "sessions_7d"):
            self.assertIn(key, body)
        for forbidden in ("user_id", "users", "email", "phone", "name", "leaderboard"):
            self.assertNotIn(forbidden, body)

    def test_protected_routes_require_auth(self):
        for method, path in [
            ("get", "/get-progress/u1"),
            ("get", "/coach/conversations/u1"),
            ("post", "/coach/chat"),
            ("get", "/leaderboard"),
            ("get", "/profile/me"),
            ("get", "/admin/me"),
            ("get", "/admin/console"),
            ("get", "/admin/prompts"),
        ]:
            resp = getattr(self.client, method)(path, **({"json": {}} if method == "post" else {}))
            self.assertIn(resp.status_code, (401, 503), f"{path} -> {resp.status_code}")

    def test_non_admin_gets_404_on_admin_routes(self):
        self._login()
        for path in ["/admin/me", "/admin/console", "/admin/overview", "/admin/prompts", "/admin/students"]:
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_cross_user_access_forbidden(self):
        self._login()
        self.assertEqual(self.client.get("/get-progress/other-user").status_code, 403)
        self.assertEqual(self.client.get("/coach/conversations/other-user").status_code, 403)
        self.assertEqual(self.client.get("/sessions/other-user").status_code, 403)

    def test_same_user_access_allowed(self):
        self._login()
        resp = self.client.get("/get-progress/student-1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["user_id"], "student-1")

    def test_planning_portfolio_requires_authentication(self):
        portfolio = self.client.post(
            "/planning/portfolio/student-1",
            json=self._planning_portfolio_payload(),
        )
        self.assertIn(portfolio.status_code, (401, 503), portfolio.text)

    def test_planning_portfolio_rejects_cross_user(self):
        self._login()
        portfolio = self.client.post(
            "/planning/portfolio/other-user",
            json=self._planning_portfolio_payload(),
        )
        self.assertEqual(portfolio.status_code, 403, portfolio.text)

    def test_planning_portfolio_authenticated_response_contract(self):
        self._login()
        request_payload = self._planning_portfolio_payload()
        portfolio = self.client.post(
            "/planning/portfolio/student-1",
            json=request_payload,
        )
        self.assertEqual(portfolio.status_code, 200, portfolio.text)
        body = portfolio.json()

        self.assertEqual(body["portfolio_version"], "planning_portfolio_v1")
        self.assertEqual(body["user_id"], "student-1")
        self.assertEqual(body["class_level"], request_payload["class_level"])
        self.assertEqual(body["subject"], request_payload["subject"])
        self.assertEqual(body["requested_chapter_count"], 2)
        self.assertEqual(body["chapter_count"], 2)
        self.assertEqual(body["deduplicated_chapter_count"], 0)
        self.assertEqual(body["study_time_today"], "15")

        chapters = body["chapters"]
        self.assertEqual(len(chapters), 2)
        self.assertEqual(
            [chapter["chapter_proficiency"] for chapter in chapters],
            ["know_a_little", "mostly_confident"],
        )
        selected = [chapter for chapter in chapters if chapter["selected_for_today"]]
        self.assertEqual(len(selected), 1)

        global_next = body["global_next_step"]
        self.assertEqual(global_next["chapter_slug"], selected[0]["chapter_slug"])
        self.assertEqual(global_next["unit_id"], selected[0]["next_step"]["unit_id"])

        today_route = body["today_route"]
        self.assertEqual(today_route["source"], "student_choice")
        self.assertEqual(today_route["budget_minutes"], 15)
        self.assertEqual(body["today_route"]["total_minutes"], 15)
        self.assertEqual(
            sum(item["minutes"] for item in today_route["items"]),
            today_route["total_minutes"],
        )
        self.assertEqual(
            {item["chapter_slug"] for item in today_route["items"]},
            {selected[0]["chapter_slug"]},
        )
        self.assertTrue(body["selection_factors"])
        self.assertEqual(
            body["aggregate_progress"]["total_units"],
            sum(chapter["progress"]["total_units"] for chapter in chapters),
        )

    def test_foreign_study_session_forbidden(self):
        self._login()
        resp = self.client.post("/section-ai", json={
            "question": "What is matter?",
            "section_id": "matter_definition",
            "session_id": "coach-someone-else-abc",
        })
        self.assertEqual(resp.status_code, 403)

    def test_client_progress_overwrite_disabled_by_default(self):
        self._login()
        resp = self.client.post("/update-progress", json={
            "user_id": "student-1", "total_tests": 1, "total_questions": 1,
            "total_correct": 1, "xp": 99999, "streak": 1,
        })
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
