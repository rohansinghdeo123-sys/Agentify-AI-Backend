import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from Logic.autonomous_study_loop import (
    _estimate_mission_budget,
    _is_fast_track,
    _normalize_mission_profile,
    run_autonomous_study_loop,
)
from schemas import AutonomousStudyRequest


class AutonomousStudyPlanningTests(unittest.TestCase):
    def test_request_contract_omits_retired_planning_fields(self):
        payload = AutonomousStudyRequest.model_validate(
            {
                "current_topic": "atomic_mass",
                "available_minutes": 30,
                "exam_target": "boards",
            }
        )

        self.assertNotIn("available_minutes", AutonomousStudyRequest.model_fields)
        self.assertNotIn("exam_target", AutonomousStudyRequest.model_fields)
        self.assertNotIn("available_minutes", payload.model_dump())
        self.assertNotIn("exam_target", payload.model_dump())

    def test_retired_knowledge_and_style_choices_are_rejected(self):
        with self.assertRaises(ValidationError):
            AutonomousStudyRequest(current_knowledge="weak_basics")

        with self.assertRaises(ValidationError):
            AutonomousStudyRequest(preferred_style="visual_intuition")

    def test_legacy_quick_revision_normalizes_to_fast_track(self):
        request = AutonomousStudyRequest(learning_goal="quick_revision")
        profile = _normalize_mission_profile(learning_goal="quick_revision")

        self.assertEqual(request.learning_goal, "fast_track")
        self.assertEqual(profile["learning_goal"], "fast_track")
        self.assertTrue(_is_fast_track(profile))

    def test_only_fast_track_goal_uses_fast_track_planning(self):
        exam_profile = _normalize_mission_profile(learning_goal="exam")
        fast_profile = _normalize_mission_profile(learning_goal="fast_track")

        self.assertFalse(_is_fast_track(exam_profile))
        self.assertTrue(_is_fast_track(fast_profile))
        self.assertLess(
            _estimate_mission_budget(fast_profile, "building"),
            _estimate_mission_budget(exam_profile, "building"),
        )

    def test_internal_profile_contains_only_active_planning_choices(self):
        profile = _normalize_mission_profile(
            current_knowledge="weak_basics",
            preferred_style="visual_intuition",
        )

        self.assertEqual(profile["current_knowledge"], "some_idea")
        self.assertEqual(profile["preferred_style"], "examples_first")
        self.assertNotIn("available_minutes", profile)
        self.assertNotIn("exam_target", profile)

    def test_generated_mission_uses_clean_profile_contract(self):
        db = SimpleNamespace(commit=lambda: None)
        coach = SimpleNamespace()
        analytics = {
            "summary": {"total_topics": 1, "avg_accuracy": 70, "streak": 2},
            "weak_areas": [],
            "topic_heatmap": [],
        }

        with (
            patch("Logic.autonomous_study_loop.get_user_analytics", return_value=analytics),
            patch("Logic.autonomous_study_loop.get_or_create_coach", return_value=coach),
        ):
            mission = run_autonomous_study_loop(
                db=db,
                user_id="planning-contract-test",
                current_topic="atomic_mass",
                learning_goal="quick_revision",
            )

        self.assertEqual(mission["mode"], "fast_track_mission")
        self.assertEqual(mission["student_state"]["learning_goal"], "fast_track")
        self.assertNotIn("available_minutes", mission["student_state"])
        self.assertNotIn("exam_target", mission["student_state"])
        self.assertNotIn("available_minutes", mission["result"]["metadata"]["profile"])
        self.assertNotIn("exam_target", mission["result"]["metadata"]["profile"])


if __name__ == "__main__":
    unittest.main()
