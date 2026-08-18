import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from Logic.autonomous_study_loop import (
    PlanningChapterNotFoundError,
    _build_chapter_study_plan,
    _estimate_mission_budget,
    _is_fast_track,
    _normalize_mission_profile,
    _resolve_chapter_scope,
    run_autonomous_study_loop,
)
from models import ContentChapter, ContentConcept
from routers.coach import coach_autonomous_study
from schemas import AutonomousStudyRequest, AutonomousStudyResponse
from services.catalog_service import resolve_catalog_chapter_units


def _memory_catalog_session():
    engine = create_engine("sqlite:///:memory:")
    ContentChapter.__table__.create(engine)
    ContentConcept.__table__.create(engine)
    return engine, sessionmaker(bind=engine)()


class AutonomousStudyPlanningTests(unittest.TestCase):
    def test_request_contract_omits_retired_planning_fields(self):
        payload = AutonomousStudyRequest.model_validate(
            {
                "current_chapter": "matter",
                "current_topic": "atomic_mass",
                "class_level": "  Class   11  ",
                "available_minutes": 30,
                "exam_target": "boards",
            }
        )

        self.assertNotIn("available_minutes", AutonomousStudyRequest.model_fields)
        self.assertNotIn("exam_target", AutonomousStudyRequest.model_fields)
        self.assertNotIn("current_topic", AutonomousStudyRequest.model_fields)
        self.assertNotIn("available_minutes", payload.model_dump())
        self.assertNotIn("exam_target", payload.model_dump())
        self.assertNotIn("current_topic", payload.model_dump())
        self.assertEqual(payload.class_level, "Class 11")

    def test_retired_knowledge_and_style_choices_are_rejected(self):
        with self.assertRaises(ValidationError):
            AutonomousStudyRequest(current_chapter="matter", current_knowledge="weak_basics")

        with self.assertRaises(ValidationError):
            AutonomousStudyRequest(current_chapter="matter", preferred_style="visual_intuition")

    def test_legacy_quick_revision_normalizes_to_fast_track(self):
        request = AutonomousStudyRequest(
            current_chapter="matter",
            learning_goal="quick_revision",
        )
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
        chapter_scope = {
            "source": "published",
            "chapter_slug": "matter",
            "chapter_label": "Basic Concepts of Chemistry",
            "subject": "Chemistry",
            "class_level": "Class 11",
            "units": [
                {
                    "id": "unit_measurement",
                    "label": "Measurement and Chemical Laws",
                    "concept_ids": ["measurement", "chemical_laws"],
                },
                {
                    "id": "unit_mole",
                    "label": "Mole Calculations",
                    "concept_ids": ["mole", "molar_mass"],
                },
            ],
        }

        with (
            patch("Logic.autonomous_study_loop.get_user_analytics", return_value=analytics),
            patch("Logic.autonomous_study_loop.get_or_create_coach", return_value=coach),
            patch(
                "Logic.autonomous_study_loop.resolve_catalog_chapter_units",
                return_value=chapter_scope,
            ),
        ):
            mission = run_autonomous_study_loop(
                db=db,
                user_id="planning-contract-test",
                current_chapter="matter",
                learning_goal="quick_revision",
                class_level="Class 11",
            )

        self.assertEqual(mission["mode"], "fast_track_mission")
        self.assertEqual(mission["plan_scope"], "chapter")
        self.assertEqual(mission["chapter"], "Basic Concepts of Chemistry")
        self.assertEqual(mission["target_topic"], "Basic Concepts of Chemistry")
        self.assertEqual(mission["learning_unit_count"], 2)
        self.assertEqual(mission["student_state"]["learning_goal"], "fast_track")
        self.assertEqual(mission["student_state"]["plan_scope"], "chapter")
        self.assertNotIn("available_minutes", mission["student_state"])
        self.assertNotIn("exam_target", mission["student_state"])
        self.assertNotIn("available_minutes", mission["result"]["metadata"]["profile"])
        self.assertNotIn("exam_target", mission["result"]["metadata"]["profile"])

        steps = mission["study_plan"]
        self.assertEqual([step["sequence"] for step in steps], [1, 2])
        self.assertEqual(
            [step["unit_id"] for step in steps],
            ["unit_measurement", "unit_mole"],
        )
        prerequisite_questions = {
            step["prerequisite_check"]["question"] for step in steps
        }
        self.assertEqual(len(prerequisite_questions), 2)
        self.assertTrue(all(step["completion_check"]["question"] for step in steps))
        self.assertIn("do not need to leave Planning", mission["prerequisite_check"]["action"])
        self.assertNotIn("Study Lab", str(mission))
        AutonomousStudyResponse(**mission)

    def test_builtin_micro_units_become_a_compact_chapter_route(self):
        engine, db = _memory_catalog_session()
        try:
            scope = resolve_catalog_chapter_units(
                db,
                chapter_ref="matter",
                subject="Chemistry",
                class_level="Class 11",
            )
            wrong_class = resolve_catalog_chapter_units(
                db,
                chapter_ref="matter",
                subject="Chemistry",
                class_level="Class 10",
            )
        finally:
            db.close()
            engine.dispose()

        self.assertIsNotNone(scope)
        self.assertEqual(scope["chapter_label"], "Basic Concepts of Chemistry")
        self.assertEqual(len(scope["units"]), 4)
        self.assertIsNone(wrong_class)

    def test_unknown_or_missing_chapter_never_selects_an_unrelated_catalog_entry(self):
        with self.assertRaises(ValidationError):
            AutonomousStudyRequest.model_validate({"current_topic": "legacy-only"})

        with patch(
            "Logic.autonomous_study_loop.resolve_catalog_chapter_units",
            return_value=None,
        ) as resolver:
            with self.assertRaises(PlanningChapterNotFoundError) as raised:
                _resolve_chapter_scope(
                    SimpleNamespace(),
                    current_chapter="Unknown Chapter",
                    subject="Chemistry",
                    class_level="Class 10",
                )

        resolver.assert_called_once_with(
            ANY,
            chapter_ref="Unknown Chapter",
            subject="Chemistry",
            class_level="Class 10",
        )
        self.assertIn("Unknown Chapter", str(raised.exception))
        self.assertIn("Choose a chapter from Planning", str(raised.exception))

    def test_route_returns_clear_422_for_an_ungrounded_chapter(self):
        error = PlanningChapterNotFoundError(
            "Unknown Chapter",
            "Chemistry",
            "Class 11",
        )
        with (
            patch("routers.coach.require_same_user_or_admin"),
            patch("routers.coach.enforce_user_quota"),
            patch("routers.coach.profile_learning_context", return_value={"class_level": "Other"}) as profile,
            patch("routers.coach.run_autonomous_study_loop", side_effect=error) as run_loop,
        ):
            with self.assertRaises(HTTPException) as raised:
                coach_autonomous_study(
                    user_id="student-1",
                    payload=AutonomousStudyRequest(
                        current_chapter="Unknown Chapter",
                        class_level="Class 11",
                    ),
                    db=SimpleNamespace(),
                    current_user={"uid": "student-1"},
                )

        profile.assert_not_called()
        self.assertEqual(run_loop.call_args.kwargs["class_level"], "Class 11")
        self.assertEqual(getattr(raised.exception, "status_code", None), 422)
        self.assertIn("Unknown Chapter", str(getattr(raised.exception, "detail", "")))
        self.assertIn(
            "Choose a chapter from Planning",
            str(getattr(raised.exception, "detail", "")),
        )

    def test_route_uses_profile_class_only_when_catalog_class_is_blank(self):
        error = PlanningChapterNotFoundError(
            "Unknown Chapter",
            "Chemistry",
            "Other",
        )
        with (
            patch("routers.coach.require_same_user_or_admin"),
            patch("routers.coach.enforce_user_quota"),
            patch(
                "routers.coach.profile_learning_context",
                return_value={"class_level": "Other"},
            ) as profile,
            patch("routers.coach.run_autonomous_study_loop", side_effect=error) as run_loop,
        ):
            with self.assertRaises(HTTPException):
                coach_autonomous_study(
                    user_id="student-1",
                    payload=AutonomousStudyRequest(current_chapter="Unknown Chapter"),
                    db=SimpleNamespace(),
                    current_user={"uid": "student-1"},
                )

        profile.assert_called_once()
        self.assertEqual(run_loop.call_args.kwargs["class_level"], "Other")

    def test_published_resolution_is_strict_and_queries_only_the_requested_chapter(self):
        engine, db = _memory_catalog_session()
        chemistry = ContentChapter(
            slug="chemistry-shared",
            subject="Chemistry",
            class_level="Class 11",
            chapter_name="Shared Chapter",
            status="approved",
        )
        physics = ContentChapter(
            slug="physics-shared",
            subject="Physics",
            class_level="Class 11",
            chapter_name="Shared Chapter",
            status="approved",
        )
        db.add_all([chemistry, physics])
        db.commit()
        db.add(
            ContentConcept(
                chapter_id=chemistry.id,
                concept_id="chemistry_core",
                title="Chemistry Core",
            )
        )
        db.commit()

        statements = []
        event.listen(
            engine,
            "before_cursor_execute",
            lambda _conn, _cursor, statement, _parameters, _context, _many: statements.append(
                statement.lower()
            ),
        )
        try:
            resolved = resolve_catalog_chapter_units(
                db,
                chapter_ref="Shared Chapter",
                subject="Chemistry",
                class_level="Class 11",
            )
            wrong_subject = resolve_catalog_chapter_units(
                db,
                chapter_ref="Shared Chapter",
                subject="Mathematics",
                class_level="Class 11",
            )
            wrong_class = resolve_catalog_chapter_units(
                db,
                chapter_ref="Shared Chapter",
                subject="Chemistry",
                class_level="Class 10",
            )
        finally:
            db.close()
            engine.dispose()

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["chapter_slug"], "chemistry-shared")
        self.assertIsNone(wrong_subject)
        self.assertIsNone(wrong_class)
        chapter_select = next(
            statement
            for statement in statements
            if statement.lstrip().startswith("select") and "from content_chapters" in statement
        )
        self.assertIn("content_chapters.subject", chapter_select)
        self.assertIn("content_chapters.class_level", chapter_select)
        self.assertIn("content_chapters.slug", chapter_select)
        self.assertIn("limit", chapter_select)

    def test_duplicate_unit_labels_still_receive_distinct_honest_repairs(self):
        scope = {
            "chapter_label": "Sample Chapter",
            "units": [
                {"id": f"unit_{index}", "label": "Core Concepts", "concept_ids": []}
                for index in range(1, 4)
            ],
        }
        profile = _normalize_mission_profile(
            current_knowledge="new",
            learning_goal="deep_understanding",
        )
        plan = _build_chapter_study_plan(scope, "baseline", profile)["study_plan"]

        readiness_questions = [step["prerequisite_check"]["question"] for step in plan]
        self.assertEqual(len(readiness_questions), len(set(readiness_questions)))
        self.assertTrue(all(f"Step {index}" in step["detail"] for index, step in enumerate(plan, 1)))
        self.assertTrue(all("Stay in Planning" in step["prerequisite_check"]["guidance"] for step in plan))
        self.assertNotIn("short explanation and example in this step", str(plan))


if __name__ == "__main__":
    unittest.main()
