import json
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from Logic.autonomous_study_loop import (
    PlanningChapterNotFoundError,
    _build_focus_area_scopes,
    _build_focus_brief,
    _deterministic_focus_brief,
    _focus_ranking_context,
    _is_quick_revision,
    _llm_focus_brief,
    _normalize_mission_profile,
    _resolve_chapter_scope,
    _validate_model_focus_ranking,
    run_autonomous_study_loop,
)
from models import ContentChapter, ContentConcept
from routers.coach import coach_autonomous_study
from schemas import AutonomousStudyRequest, AutonomousStudyResponse
from services.catalog_service import (
    _planning_subtopic_candidates,
    _planning_subtopics,
    resolve_catalog_chapter_units,
)


def _memory_catalog_session():
    engine = create_engine("sqlite:///:memory:")
    ContentChapter.__table__.create(engine)
    ContentConcept.__table__.create(engine)
    return engine, sessionmaker(bind=engine)()


def _chapter_scope(count: int = 4):
    units = []
    for index in range(1, count + 1):
        units.append(
            {
                "id": f"unit_{index}",
                "label": f"Approved Unit {index}",
                "concept_ids": [f"concept_{index}_a", f"concept_{index}_b"],
                "subtopics": [f"Approved Subtopic {index}A", f"Approved Subtopic {index}B"],
                "focus_signals": {
                    "concept_count": 2,
                    "importance_score": 4 if index == 2 else 2,
                    "exam_weightage_score": 3 if index == 2 else 1,
                    "average_difficulty": 4 if index == 2 else 2,
                },
            }
        )
    return {
        "source": "published",
        "chapter_slug": "approved-chapter",
        "chapter_label": "Approved Chapter",
        "subject": "Science",
        "class_level": "Class 10",
        "units": units,
    }


def _valid_model_payload(scope, analytics=None):
    _areas, _signals, _scores, ranked_ids = _focus_ranking_context(
        scope,
        analytics or {},
    )
    return {"focus_ranking": ranked_ids}


class AutonomousStudyPlanningTests(unittest.TestCase):
    def test_request_contract_is_chapter_only_and_ignores_old_payload_fields(self):
        payload = AutonomousStudyRequest.model_validate(
            {
                "current_chapter": "matter",
                "subject": "Chemistry",
                "current_topic": "atomic_mass",
                "class_level": "  Class   11  ",
                "available_minutes": 120,
                "exam_target": "boards",
            }
        )

        self.assertNotIn("current_topic", AutonomousStudyRequest.model_fields)
        self.assertNotIn("available_minutes", AutonomousStudyRequest.model_fields)
        self.assertNotIn("exam_target", AutonomousStudyRequest.model_fields)
        self.assertEqual(payload.class_level, "Class 11")

    def test_retired_choices_are_removed_and_fast_track_maps_to_quick_revision(self):
        request = AutonomousStudyRequest(
            current_chapter="matter",
            subject="Chemistry",
            class_level="Class 11",
            learning_goal="fast_track",
        )
        profile = _normalize_mission_profile(learning_goal="fast_track")
        self.assertNotIn("current_knowledge", AutonomousStudyRequest.model_fields)
        self.assertNotIn("learning_goal", AutonomousStudyRequest.model_fields)
        self.assertNotIn("preferred_style", AutonomousStudyRequest.model_fields)
        self.assertNotIn("study_time_today", AutonomousStudyRequest.model_fields)
        self.assertEqual(request.chapter_proficiency, "mostly_confident")
        self.assertTrue(_is_quick_revision(profile))

    def test_request_requires_nonblank_selected_class_subject_and_chapter(self):
        valid = {
            "current_chapter": "Matter",
            "subject": "Chemistry",
            "class_level": "Class 11",
        }
        for missing in ("current_chapter", "subject", "class_level"):
            payload = dict(valid)
            payload.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValidationError):
                AutonomousStudyRequest.model_validate(payload)

        for blank in ("current_chapter", "subject", "class_level"):
            payload = dict(valid)
            payload[blank] = "   "
            with self.subTest(blank=blank), self.assertRaises(ValidationError):
                AutonomousStudyRequest.model_validate(payload)

    def test_focus_area_aggregation_is_compact_ordered_and_lossless(self):
        scope = _chapter_scope(10)
        areas = _build_focus_area_scopes(scope["units"])

        self.assertEqual(len(areas), 5)
        self.assertEqual(
            [unit_id for area in areas for unit_id in area["unit_ids"]],
            [f"unit_{index}" for index in range(1, 11)],
        )
        self.assertTrue(all(1 <= len(area["subtopics"]) <= 4 for area in areas))
        self.assertTrue(all(len(area["unit_ids"]) == len(area["unit_titles"]) for area in areas))

    def test_deterministic_ranking_has_useful_hierarchy_and_elevates_weakness(self):
        scope = _chapter_scope(4)
        analytics = {
            "weak_areas": [{"topic": "concept_4_a", "accuracy": 20}],
            "topic_heatmap": [],
        }
        brief = _deterministic_focus_brief(scope, analytics)
        by_unit = {
            unit_id: area["focus_level"]
            for area in brief["focus_areas"]
            for unit_id in area["unit_ids"]
        }

        self.assertEqual(set(by_unit.values()), {"high", "medium", "light"})
        self.assertEqual(by_unit["unit_4"], "high")
        self.assertEqual(len(brief["guidance_steps"]), 4)

    def test_consolidated_area_surfaces_late_weakness_that_drives_deep_focus(self):
        scope = _chapter_scope(25)
        for learner_topic in ("concept_5_a", "unit_5", "Approved Unit 5"):
            analytics = {
                "weak_areas": [{"topic": learner_topic, "accuracy": 18}],
                "topic_heatmap": [],
            }

            with self.subTest(learner_topic=learner_topic):
                brief = _deterministic_focus_brief(scope, analytics)
                first_area = brief["focus_areas"][0]

                self.assertEqual(len(first_area["unit_ids"]), 5)
                self.assertEqual(first_area["focus_level"], "high")
                self.assertLessEqual(len(first_area["subtopics"]), 4)
                self.assertIn("Approved Subtopic 5A", first_area["subtopics"])
                self.assertIn("Approved Subtopic 5A", first_area["reason"])
                self.assertEqual(
                    [
                        unit_id
                        for area in brief["focus_areas"]
                        for unit_id in area["unit_ids"]
                    ],
                    [f"unit_{index}" for index in range(1, 26)],
                )

    def test_llm_can_only_rank_immutable_approved_focus_areas(self):
        scope = _chapter_scope(7)
        payload = _valid_model_payload(scope)
        with patch(
            "Logic.autonomous_study_loop.model_gateway.complete",
            return_value=json.dumps(payload),
        ) as complete:
            brief = _llm_focus_brief(scope, {}, _normalize_mission_profile(class_level="Class 10"))

        self.assertEqual(len(brief["focus_areas"]), 5)
        self.assertEqual(brief["focus_areas"][0]["unit_titles"], ["Approved Unit 1", "Approved Unit 2"])
        self.assertEqual(brief["focus_areas"][0]["subtopics"][0], "Approved Subtopic 1A")
        self.assertIn("approved_focus_areas", complete.call_args.args[1][1]["content"])

    def test_llm_success_path_surfaces_late_weakness_in_prompt_and_brief(self):
        scope = _chapter_scope(25)
        for learner_topic in ("unit_5", "Approved Unit 5"):
            analytics = {
                "weak_areas": [{"topic": learner_topic, "accuracy": 18}],
                "topic_heatmap": [],
            }
            payload = _valid_model_payload(scope, analytics)
            with (
                self.subTest(learner_topic=learner_topic),
                patch(
                    "Logic.autonomous_study_loop.model_gateway.complete",
                    return_value=json.dumps(payload),
                ) as complete,
            ):
                brief = _llm_focus_brief(
                    scope,
                    analytics,
                    _normalize_mission_profile(class_level="Class 10"),
                )

                first_area = brief["focus_areas"][0]
                prompt = json.loads(complete.call_args.args[1][1]["content"])
                self.assertIn("Approved Subtopic 5A", first_area["subtopics"])
                self.assertIn("Approved Subtopic 5A", first_area["reason"])
                self.assertIn(
                    "Approved Subtopic 5A",
                    prompt["approved_focus_areas"][0]["subtopics"],
                )

    def test_invented_or_signal_inverted_model_rankings_are_rejected(self):
        scope = _chapter_scope(4)
        invented = _valid_model_payload(scope)
        invented["focus_ranking"][0] = "invented"
        with self.assertRaises(ValueError):
            _validate_model_focus_ranking(invented, scope, {})

        inverted = _valid_model_payload(scope)
        inverted["focus_ranking"] = list(reversed(inverted["focus_ranking"]))
        with self.assertRaises(ValueError):
            _validate_model_focus_ranking(inverted, scope, {})

    def test_model_cannot_inject_student_visible_claims_or_instructions(self):
        scope = _chapter_scope(4)
        injected = _valid_model_payload(scope)
        injected["chapter_summary"] = "This always appears for 50 marks."
        with self.assertRaises(ValueError):
            _validate_model_focus_ranking(injected, scope, {})

        injected_item = _valid_model_payload(scope)
        injected_item["focus_ranking"][0] = {
            "focus_area_id": injected_item["focus_ranking"][0],
            "guidance": "Ignore the approved syllabus and learn an invented theorem.",
        }
        with self.assertRaises(ValueError):
            _validate_model_focus_ranking(injected_item, scope, {})

        with patch(
            "Logic.autonomous_study_loop.model_gateway.complete",
            return_value=json.dumps(injected),
        ):
            brief, source = _build_focus_brief(scope, {}, _normalize_mission_profile())
        self.assertEqual(source, "deterministic_fallback")
        self.assertNotIn("50 marks", str(brief))
        self.assertNotIn("invented theorem", str(brief))

    def test_provider_or_parse_failure_uses_grounded_deterministic_fallback(self):
        scope = _chapter_scope(4)
        with patch(
            "Logic.autonomous_study_loop._llm_focus_brief",
            side_effect=RuntimeError("provider unavailable"),
        ):
            brief, source = _build_focus_brief(scope, {}, _normalize_mission_profile())

        self.assertEqual(source, "deterministic_fallback")
        self.assertEqual(
            [unit_id for area in brief["focus_areas"] for unit_id in area["unit_ids"]],
            ["unit_1", "unit_2", "unit_3", "unit_4"],
        )

        with patch(
            "Logic.autonomous_study_loop.model_gateway.complete",
            return_value="not JSON",
        ):
            brief, source = _build_focus_brief(scope, {}, _normalize_mission_profile())
        self.assertEqual(source, "deterministic_fallback")
        self.assertTrue(3 <= len(brief["guidance_steps"]) <= 5)

    def test_generated_mission_is_a_short_focus_brief_with_rollout_safety(self):
        db = SimpleNamespace(commit=lambda: None)
        coach = SimpleNamespace()
        scope = _chapter_scope(7)
        analytics = {"summary": {"avg_accuracy": 70}, "weak_areas": [], "topic_heatmap": []}
        with (
            patch("Logic.autonomous_study_loop.get_user_analytics", return_value=analytics),
            patch("Logic.autonomous_study_loop.get_or_create_coach", return_value=coach),
            patch("Logic.autonomous_study_loop.resolve_catalog_chapter_units", return_value=scope),
            patch(
                "Logic.autonomous_study_loop.model_gateway.complete",
                side_effect=RuntimeError("provider unavailable"),
            ),
        ):
            mission = run_autonomous_study_loop(
                db=db,
                user_id="student-1",
                current_chapter="approved-chapter",
                subject="Science",
                class_level="Class 10",
            )

        self.assertEqual(mission["brief_version"], "chapter_focus_v1")
        self.assertEqual(mission["coverage"]["unit_count"], 7)
        self.assertEqual(mission["learning_unit_count"], len(mission["focus_areas"]))
        self.assertLessEqual(len(mission["focus_areas"]), 5)
        self.assertTrue(3 <= len(mission["guidance_steps"]) <= 5)
        self.assertEqual(mission["estimated_minutes"], 0)
        self.assertNotIn("duration", str(mission["focus_areas"]))
        self.assertNotIn("prerequisite", str(mission["focus_areas"]).lower())
        self.assertNotIn("Study Lab", str(mission))
        self.assertNotEqual(len(mission["high_priority_concepts"]), len(mission["focus_areas"]))
        # Old deployed clients can validate during the staggered rollout.
        legacy = mission["study_plan"]
        self.assertEqual(mission["learning_unit_count"], len(legacy))
        self.assertEqual(len({step["unit_id"] for step in legacy}), len(legacy))
        self.assertEqual(
            len({step["prerequisite_check"]["question"] for step in legacy}),
            len(legacy),
        )
        self.assertTrue(all(step["duration"] == "Self-paced" for step in legacy))
        self.assertTrue(all(step["prerequisite_check"]["status"] == "ready" for step in legacy))
        self.assertTrue(all(step["completion_check"]["question"] for step in legacy))
        self.assertIn(
            mission["diagnostic_question"]["correct"],
            mission["diagnostic_question"]["options"],
        )
        AutonomousStudyResponse(**mission)

    def test_database_read_transaction_ends_before_external_model_call(self):
        events = []

        class TrackingDb:
            def commit(self):
                events.append("commit")

        scope = _chapter_scope(4)
        payload = _valid_model_payload(scope)

        def model_call(*_args, **_kwargs):
            self.assertEqual(events, ["commit"])
            events.append("model")
            return json.dumps(payload)

        with (
            patch(
                "Logic.autonomous_study_loop.get_user_analytics",
                return_value={"summary": {}, "weak_areas": [], "topic_heatmap": []},
            ),
            patch(
                "Logic.autonomous_study_loop.resolve_catalog_chapter_units",
                return_value=scope,
            ),
            patch(
                "Logic.autonomous_study_loop.get_or_create_coach",
                return_value=SimpleNamespace(),
            ),
            patch(
                "Logic.autonomous_study_loop.model_gateway.complete",
                side_effect=model_call,
            ),
        ):
            mission = run_autonomous_study_loop(
                db=TrackingDb(),
                user_id="transaction-student",
                current_chapter="approved-chapter",
                subject="Science",
                class_level="Class 10",
            )

        self.assertEqual(events, ["commit", "model", "commit"])
        self.assertEqual(mission["result"]["metadata"]["generation_source"], "llm")

    def test_database_connection_is_returned_before_external_model_call(self):
        engine = create_engine(
            "sqlite://",
            poolclass=QueuePool,
            connect_args={"check_same_thread": False},
        )
        db = sessionmaker(bind=engine)()
        scope = _chapter_scope(4)
        payload = _valid_model_payload(scope)

        def resolve_with_read(session, **_kwargs):
            session.execute(text("SELECT 1")).scalar_one()
            self.assertTrue(session.in_transaction())
            return scope

        def model_call(*_args, **_kwargs):
            self.assertFalse(db.in_transaction())
            self.assertEqual(engine.pool.checkedout(), 0)
            return json.dumps(payload)

        try:
            with (
                patch(
                    "Logic.autonomous_study_loop.get_user_analytics",
                    return_value={"summary": {}, "weak_areas": [], "topic_heatmap": []},
                ),
                patch(
                    "Logic.autonomous_study_loop.resolve_catalog_chapter_units",
                    side_effect=resolve_with_read,
                ),
                patch(
                    "Logic.autonomous_study_loop.get_or_create_coach",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "Logic.autonomous_study_loop.model_gateway.complete",
                    side_effect=model_call,
                ),
            ):
                mission = run_autonomous_study_loop(
                    db=db,
                    user_id="transaction-student",
                    current_chapter="approved-chapter",
                    subject="Science",
                    class_level="Class 10",
                )
        finally:
            db.close()
            engine.dispose()

        self.assertEqual(mission["result"]["metadata"]["generation_source"], "llm")

    def test_builtin_catalog_provides_grounded_subtopics_and_focus_metadata(self):
        engine, db = _memory_catalog_session()
        try:
            scope = resolve_catalog_chapter_units(
                db,
                chapter_ref="matter",
                subject="Chemistry",
                class_level="Class 11",
            )
            wrong_subject = resolve_catalog_chapter_units(
                db,
                chapter_ref="matter",
                subject="Science",
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
        self.assertIsNone(wrong_subject)
        self.assertIsNone(wrong_class)
        self.assertEqual(len(scope["units"]), 4)
        self.assertTrue(all(unit["subtopics"] for unit in scope["units"]))
        self.assertTrue(all("focus_signals" in unit for unit in scope["units"]))

    def test_published_chapter_resolution_never_crosses_selected_scope(self):
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
        lower_class = ContentChapter(
            slug="chemistry-shared-10",
            subject="Chemistry",
            class_level="Class 10",
            chapter_name="Shared Chapter",
            status="approved",
        )
        db.add_all([chemistry, physics, lower_class])
        db.commit()
        db.add(
            ContentConcept(
                chapter_id=chemistry.id,
                concept_id="chemistry_core",
                title="Chemistry Core",
            )
        )
        db.commit()
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
                class_level="Class 9",
            )
        finally:
            db.close()
            engine.dispose()

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["chapter_slug"], "chemistry-shared")
        self.assertIsNone(wrong_subject)
        self.assertIsNone(wrong_class)

    def test_published_catalog_aggregates_only_approved_concept_metadata(self):
        engine, db = _memory_catalog_session()
        chapter = ContentChapter(
            slug="forces",
            subject="Science",
            class_level="Class 10",
            chapter_name="Forces",
            status="approved",
        )
        db.add(chapter)
        db.commit()
        db.add(
            ContentConcept(
                chapter_id=chapter.id,
                concept_id="force_core",
                title="Force Core",
                importance_level="core",
                typical_exam_weightage="high",
                difficulty_level=4,
            )
        )
        db.commit()
        try:
            scope = resolve_catalog_chapter_units(
                db,
                chapter_ref="forces",
                subject="Science",
                class_level="Class 10",
            )
        finally:
            db.close()
            engine.dispose()

        self.assertEqual(scope["units"][0]["subtopics"], ["Force Core"])
        self.assertEqual(scope["units"][0]["focus_signals"]["importance_score"], 4)
        self.assertEqual(scope["units"][0]["focus_signals"]["exam_weightage_score"], 3)

    def test_catalog_subtopics_include_late_priority_concept_without_invention(self):
        concepts = [
            {
                "concept_id": f"concept_{index}",
                "title": f"Approved Concept {index}",
                "importance_level": "low",
                "typical_exam_weightage": "low",
                "difficulty_level": 1,
            }
            for index in range(1, 7)
        ]
        concepts[-1].update(
            {
                "importance_level": "essential",
                "typical_exam_weightage": "high",
                "difficulty_level": 5,
            }
        )

        selected = _planning_subtopics(concepts, "Fallback")
        candidates = _planning_subtopic_candidates(concepts, "Fallback")

        self.assertEqual(len(selected), 4)
        self.assertIn("Approved Concept 6", selected)
        self.assertEqual(len(candidates), 6)
        self.assertTrue(set(selected).issubset({concept["title"] for concept in concepts}))

    def test_unknown_chapter_never_fabricates_a_brief(self):
        with patch("Logic.autonomous_study_loop.resolve_catalog_chapter_units", return_value=None) as resolver:
            with self.assertRaises(PlanningChapterNotFoundError):
                _resolve_chapter_scope(
                    SimpleNamespace(),
                    current_chapter="Unknown Chapter",
                    subject="Science",
                    class_level="Class 10",
                )
        resolver.assert_called_once_with(
            ANY,
            chapter_ref="Unknown Chapter",
            subject="Science",
            class_level="Class 10",
        )

    def test_route_preserves_explicit_class_and_returns_clear_422(self):
        error = PlanningChapterNotFoundError("Unknown Chapter", "Science", "Class 10")
        with (
            patch("routers.coach.require_same_user_or_admin"),
            patch("routers.coach.enforce_user_quota"),
            patch("routers.coach.profile_learning_context") as profile,
            patch("routers.coach.run_autonomous_study_loop", side_effect=error) as run_loop,
        ):
            with self.assertRaises(HTTPException) as raised:
                coach_autonomous_study(
                    user_id="student-1",
                    payload=AutonomousStudyRequest(
                        current_chapter="Unknown Chapter",
                        subject="Science",
                        class_level="Class 10",
                    ),
                    db=SimpleNamespace(),
                    current_user={"uid": "student-1"},
                )

        profile.assert_not_called()
        self.assertEqual(run_loop.call_args.kwargs["class_level"], "Class 10")
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("Choose a chapter from Planning", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
