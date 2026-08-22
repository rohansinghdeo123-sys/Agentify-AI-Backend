import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Logic.autonomous_study_loop import run_autonomous_study_loop
from Logic.planning.curriculum_registry import (
    PlanningCurriculumError,
    _validate_manifest,
    _verify_source_file,
    load_planning_curriculum,
    resolve_planning_curriculum,
)
from Logic.planning.recommendation_engine import (
    TIME_BUDGETS,
    build_planning_roadmap,
    derive_unit_statuses,
)
from Logic.tools.knowledge_search import search_knowledge_base
from models import ContentChapter, ContentConcept, PlanningLearningEvent
from schemas import AutonomousStudyRequest, AutonomousStudyResponse
from services.catalog_service import build_catalog, resolve_catalog_topic
from services.planning_progress_service import (
    confirm_study_answer_event,
    planning_learning_states,
    record_study_answer_event,
)


def _catalog_session():
    engine = create_engine("sqlite:///:memory:")
    ContentChapter.__table__.create(engine)
    ContentConcept.__table__.create(engine)
    return engine, sessionmaker(bind=engine)()


def _planning_event_session():
    engine = create_engine("sqlite:///:memory:")
    PlanningLearningEvent.__table__.create(engine)
    return engine, sessionmaker(bind=engine)()


class PlanningRoadmapV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curriculum = load_planning_curriculum()

    def test_manifest_locks_complete_ncert_order_and_dependencies(self):
        curriculum = self.curriculum
        units = curriculum["units"]

        self.assertTrue(curriculum["content_order_locked"])
        self.assertEqual([unit["order"] for unit in units], list(range(1, 11)))
        self.assertEqual(len(units), 10)
        self.assertEqual(units[0]["ncert_sections"][-1]["id"], "1.1")
        self.assertEqual(units[-1]["ncert_sections"][0]["id"], "1.10")
        self.assertEqual(
            [
                section["id"]
                for unit in units
                for section in unit["ncert_sections"]
                if section["id"] in {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10"}
            ],
            [f"1.{index}" for index in range(1, 11)],
        )

        order = {unit["id"]: unit["order"] for unit in units}
        reverse = {unit["id"]: [] for unit in units}
        for unit in units:
            for prerequisite in unit["prerequisite_unit_ids"]:
                self.assertLess(order[prerequisite], unit["order"])
                reverse[prerequisite].append(unit["id"])
        self.assertEqual(
            {unit["id"]: unit["dependent_unit_ids"] for unit in units},
            reverse,
        )

    def test_manifest_validator_fails_closed_on_reordering_or_bad_dependency(self):
        reordered = deepcopy(self.curriculum)
        reordered["units"][1]["order"] = 8
        with self.assertRaises(PlanningCurriculumError):
            _validate_manifest(reordered)

        bad_dependency = deepcopy(self.curriculum)
        bad_dependency["units"][0]["prerequisite_unit_ids"] = [bad_dependency["units"][1]["id"]]
        with self.assertRaises(PlanningCurriculumError):
            _validate_manifest(bad_dependency)

        bad_source = deepcopy(self.curriculum["source"])
        bad_source["sha256"] = "0" * 64
        with self.assertRaises(PlanningCurriculumError):
            _verify_source_file(bad_source)

        colliding_alias = deepcopy(self.curriculum)
        colliding_alias["units"][0]["legacy_topic_ids"] = ["nature_of_matter"]
        with self.assertRaises(PlanningCurriculumError):
            _validate_manifest(colliding_alias)

    def test_scope_resolver_accepts_historical_and_published_slugs_but_not_wrong_scope(self):
        self.assertIsNotNone(
            resolve_planning_curriculum(
                chapter_ref="matter",
                subject="Chemistry",
                class_level="Class 11",
            )
        )
        self.assertIsNotNone(
            resolve_planning_curriculum(
                chapter_ref="ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry",
                subject="Chemistry",
                class_level="11",
            )
        )
        self.assertIsNone(
            resolve_planning_curriculum(
                chapter_ref="some_basic_concepts_of_chemistry",
                subject="Physics",
                class_level="Class 11",
            )
        )

    def test_request_time_contract_is_optional_normalized_and_strict(self):
        common = {
            "current_chapter": "some_basic_concepts_of_chemistry",
            "subject": "Chemistry",
            "class_level": "Class 11",
        }
        self.assertEqual(AutonomousStudyRequest(**common).study_time_today, "no_limit")
        self.assertEqual(
            AutonomousStudyRequest(**common, study_time_today="120+").study_time_today,
            "120_plus",
        )
        self.assertEqual(
            AutonomousStudyRequest(**common, learning_goal="quick_revision").learning_goal,
            "fast_track",
        )

    def test_every_time_option_is_bounded_and_starts_at_the_first_ncert_unit(self):
        first_id = self.curriculum["units"][0]["id"]
        for preference, budget in TIME_BUDGETS.items():
            with self.subTest(preference=preference):
                roadmap = build_planning_roadmap(
                    self.curriculum,
                    study_time_today=preference,
                )
                route = roadmap["daily_route"]
                self.assertEqual(roadmap["study_time_today"], preference)
                self.assertEqual(route["time_preference"], preference)
                self.assertEqual(route["budget_minutes"], budget)
                self.assertEqual(route["items"][0]["unit_id"], first_id)
                if budget is not None:
                    self.assertLessEqual(route["total_minutes"], budget)

    def test_ten_student_stress_scenarios_keep_the_route_clear_and_ordered(self):
        units = self.curriculum["units"]
        expected_order = [unit["id"] for unit in units]
        first = units[0]
        first_mastery = {
            "topic_heatmap": [
                {
                    "topic": first["primary_topic_id"],
                    "value": 90,
                    "attempts": first["practice"]["minimum_items"],
                }
            ]
        }
        scenarios = [
            ("beginner", {"profile": {"current_knowledge": "new", "learning_goal": "deep_understanding"}}),
            ("knows_basics", {"profile": {"current_knowledge": "know_basics", "learning_goal": "fast_track"}}),
            ("thirty_minutes", {"study_time_today": "30"}),
            ("one_hour", {"study_time_today": "60"}),
            ("exam_tomorrow", {"profile": {"current_knowledge": "some_idea", "learning_goal": "exam"}}),
            ("deep_learning", {"profile": {"current_knowledge": "some_idea", "learning_goal": "deep_understanding"}}),
            ("struggling_prerequisite", {"analytics": {"topic_heatmap": [{"topic": first["primary_topic_id"], "value": 35, "attempts": 2}]}}),
            ("masters_quickly", {"analytics": first_mastery}),
            ("large_chapter_scan", {"study_time_today": "15"}),
            ("strong_dependency_chain", {"analytics": first_mastery, "study_time_today": "120_plus"}),
        ]
        for name, kwargs in scenarios:
            with self.subTest(scenario=name):
                roadmap = build_planning_roadmap(self.curriculum, **kwargs)
                self.assertEqual(
                    [unit["id"] for unit in roadmap["learning_units"]],
                    expected_order,
                )
                self.assertIn(roadmap["next_step"]["unit_id"], expected_order)
                self.assertTrue(roadmap["next_step"]["reason"].strip())
                self.assertGreaterEqual(len(roadmap["daily_route"]["items"]), 1)
                budget = roadmap["daily_route"]["budget_minutes"]
                if budget is not None:
                    self.assertLessEqual(roadmap["daily_route"]["total_minutes"], budget)

    def test_importance_never_reorders_the_learning_sequence(self):
        curriculum = deepcopy(self.curriculum)
        curriculum["units"][0]["importance"] = "low"
        curriculum["units"][-1]["importance"] = "very_high"
        roadmap = build_planning_roadmap(curriculum, study_time_today="60")

        self.assertEqual(
            [unit["id"] for unit in roadmap["learning_units"]],
            [unit["id"] for unit in curriculum["units"]],
        )
        self.assertEqual(roadmap["next_step"]["unit_id"], curriculum["units"][0]["id"])
        self.assertNotEqual(
            roadmap["next_step"]["unit_id"],
            curriculum["units"][-1]["id"],
        )

    def test_personalization_changes_workload_and_wording_not_order(self):
        beginner = build_planning_roadmap(
            self.curriculum,
            profile={"current_knowledge": "new", "learning_goal": "deep_understanding"},
        )
        basics = build_planning_roadmap(
            self.curriculum,
            profile={"current_knowledge": "know_basics", "learning_goal": "fast_track"},
        )
        exam = build_planning_roadmap(
            self.curriculum,
            profile={"current_knowledge": "some_idea", "learning_goal": "exam"},
        )

        expected_order = [unit["id"] for unit in self.curriculum["units"]]
        for roadmap in (beginner, basics, exam):
            self.assertEqual([unit["id"] for unit in roadmap["learning_units"]], expected_order)
        self.assertGreaterEqual(
            beginner["daily_route"]["total_minutes"],
            basics["daily_route"]["total_minutes"],
        )
        self.assertIn("essential", basics["daily_route"]["items"][0]["activity"])
        self.assertIn("school-exam", exam["daily_route"]["items"][0]["activity"])

    def test_analytics_mastery_requires_canonical_score_and_evidence_boundaries(self):
        first = self.curriculum["units"][0]
        required = first["practice"]["minimum_items"]
        weak = {
            "topic_heatmap": [
                {"topic": first["primary_topic_id"], "value": 35, "attempts": 4}
            ]
        }
        self.assertEqual(derive_unit_statuses(self.curriculum, weak)[first["id"]], "needs_review")

        cases = [
            (required - 1, 100, "practising"),
            (required, 79, "practising"),
            (required, 80, "mastered"),
        ]
        for attempts, accuracy, expected in cases:
            with self.subTest(attempts=attempts, accuracy=accuracy):
                analytics = {
                    "topic_heatmap": [
                        {
                            "topic": first["primary_topic_id"],
                            "value": accuracy,
                            "attempts": attempts,
                        }
                    ]
                }
                self.assertEqual(
                    derive_unit_statuses(self.curriculum, analytics)[first["id"]],
                    expected,
                )

        noncanonical = {
            "topic_heatmap": [
                {
                    "topic": first["legacy_topic_ids"][0],
                    "value": 100,
                    "attempts": 20,
                }
            ]
        }
        self.assertEqual(
            derive_unit_statuses(self.curriculum, noncanonical)[first["id"]],
            "practising",
        )

    def test_verified_canonical_mastery_moves_to_next_unit_without_skipping(self):
        first, second = self.curriculum["units"][:2]
        analytics = {
            "topic_heatmap": [
                {
                    "topic": first["primary_topic_id"],
                    "value": 86,
                    "attempts": 5,
                }
            ]
        }
        roadmap = build_planning_roadmap(self.curriculum, analytics=analytics)

        self.assertEqual(roadmap["learning_units"][0]["status"], "mastered")
        self.assertEqual(roadmap["next_step"]["unit_id"], second["id"])
        self.assertEqual(roadmap["progress"]["percentage"], 10)

    def test_v2_run_is_deterministic_and_v1_alias_client_order_stays_ncert_ordered(self):
        db = SimpleNamespace(commit=lambda: None)
        coach = SimpleNamespace()
        analytics = {"summary": {}, "weak_areas": [], "topic_heatmap": []}
        with (
            patch("Logic.autonomous_study_loop.get_user_analytics", return_value=analytics),
            patch("Logic.autonomous_study_loop.get_or_create_coach", return_value=coach),
            patch("Logic.autonomous_study_loop.planning_learning_states", return_value=[]),
            patch("Logic.autonomous_study_loop.model_gateway.complete") as complete,
        ):
            mission = run_autonomous_study_loop(
                db=db,
                user_id="roadmap-student",
                current_chapter="some_basic_concepts_of_chemistry",
                subject="Chemistry",
                class_level="Class 11",
                study_time_today="30",
            )

        complete.assert_not_called()
        self.assertEqual(mission["roadmap_version"], "planning_roadmap_v2")
        self.assertEqual(mission["coverage"]["unit_count"], 10)
        canonical_order = mission["coverage"]["included_unit_ids"]
        self.assertEqual(
            [unit["id"] for unit in mission["learning_units"]],
            canonical_order,
        )
        self.assertTrue(
            all(area["focus_level"] == "medium" for area in mission["focus_areas"])
        )
        deployed_v1_order = [
            unit_id
            for area in mission["focus_areas"]
            for unit_id in area["unit_ids"]
        ]
        self.assertEqual(deployed_v1_order, canonical_order)
        AutonomousStudyResponse(**mission)

    def test_planning_catalog_is_additive_and_explicit_handoff_uses_manifest(self):
        engine, db = _catalog_session()
        try:
            catalog = build_catalog(db)
            chemistry = next(
                group
                for group in catalog["subjects"]
                if group["subject"] == "Chemistry" and group["class_level"] == "Class 11"
            )
            starter_chapter = next(
                item
                for item in chemistry["chapters"]
                if item["slug"] == "matter"
            )
            planning_chapter = next(
                item
                for item in catalog["planning_chapters"]
                if item["canonical_slug"] == "some_basic_concepts_of_chemistry"
            )
            resolved = resolve_catalog_topic(
                db,
                self.curriculum["units"][0]["primary_topic_id"],
                subject="Chemistry",
                chapter="matter",
                class_level="Class 11",
                catalog_source="planning_manifest",
            )
        finally:
            db.close()
            engine.dispose()

        self.assertEqual(len(starter_chapter["topics"]), 12)
        self.assertIn("matter", planning_chapter["aliases"])
        self.assertEqual(resolved["catalog_source"], "planning_manifest")
        self.assertEqual(resolved["chapter_slug"], "some_basic_concepts_of_chemistry")

        unit_one = self.curriculum["units"][0]
        self.assertTrue(
            set(unit_one["legacy_topic_ids"]).issubset(resolved["concept_ids"])
        )

    def test_published_catalog_stays_intact_while_planning_scope_is_separate(self):
        engine, db = _catalog_session()
        try:
            chapter_row = ContentChapter(
                slug="matter",
                subject="Chemistry",
                class_level="Class 11",
                chapter_name="Matter",
                chapter_number=1,
                status="published",
            )
            db.add(chapter_row)
            db.flush()
            db.add(
                ContentConcept(
                    chapter_id=chapter_row.id,
                    concept_id="chemistry_definition",
                    title="Definition of Chemistry",
                )
            )
            db.commit()
            catalog = build_catalog(db)
            chemistry = next(
                group
                for group in catalog["subjects"]
                if group["subject"] == "Chemistry" and group["class_level"] == "Class 11"
            )
            published_chapters = [
                chapter
                for chapter in chemistry["chapters"]
                if chapter["slug"] == "matter"
            ]
            planning_chapters = catalog["planning_chapters"]
        finally:
            db.close()
            engine.dispose()

        self.assertEqual(len(published_chapters), 1)
        self.assertEqual(published_chapters[0]["topics"][0]["id"], "chemistry_definition")
        self.assertEqual(
            [chapter["canonical_slug"] for chapter in planning_chapters],
            ["some_basic_concepts_of_chemistry"],
        )

    def test_historical_topic_ids_resolve_to_their_canonical_manifest_units(self):
        engine, db = _catalog_session()
        try:
            expected = {
                **{
                    alias: self.curriculum["units"][0]
                    for alias in self.curriculum["units"][0]["legacy_topic_ids"]
                },
                **{
                    alias: self.curriculum["units"][1]
                    for alias in self.curriculum["units"][1]["legacy_topic_ids"]
                },
            }
            for alias, unit in expected.items():
                with self.subTest(alias=alias):
                    resolved = resolve_catalog_topic(
                        db,
                        alias,
                        subject="Chemistry",
                        chapter="matter",
                        class_level="Class 11",
                        catalog_source="planning_manifest",
                    )
                    self.assertEqual(resolved["section_id"], unit["primary_topic_id"])
                    self.assertEqual(resolved["chapter_slug"], self.curriculum["chapter_slug"])
        finally:
            db.close()
            engine.dispose()

    def test_every_learning_unit_retrieves_only_its_registered_pdf_slices(self):
        for unit in self.curriculum["units"]:
            with self.subTest(unit=unit["id"]):
                result = search_knowledge_base(
                    unit["primary_topic_id"],
                    unit["title"],
                    scope={
                        "catalog_source": "planning_manifest",
                        "chapter_slug": self.curriculum["chapter_slug"],
                        "subject": self.curriculum["subject"],
                        "class_level": self.curriculum["class_level"],
                        "planning_unit_id": unit["id"],
                        "section_id": unit["primary_topic_id"],
                    },
                )
                allowed_pages = {segment["page"] for segment in unit["source_segments"]}
                self.assertNotIn("error", result)
                self.assertEqual(result["unit_id"], unit["id"])
                self.assertEqual(result["section_id"], unit["primary_topic_id"])
                self.assertTrue(set(result["source_pages"]).issubset(allowed_pages))
                self.assertGreater(result["paragraphs_found"], 0)
                self.assertTrue(result["context"].strip())

    def test_study_answer_evidence_is_server_owned_idempotent_and_never_mastery(self):
        engine, db = _planning_event_session()
        unit = self.curriculum["units"][0]
        scope = {
            "catalog_source": "planning_manifest",
            "chapter_slug": self.curriculum["chapter_slug"],
            "subject": self.curriculum["subject"],
            "class_level": self.curriculum["class_level"],
            "planning_unit_id": unit["id"],
            "section_id": unit["primary_topic_id"],
        }
        try:
            first = record_study_answer_event(
                db,
                user_id="student-1",
                interaction_id="study_answer:42",
                scope=scope,
                source_session_id="study-student-1-abc",
            )
            duplicate = record_study_answer_event(
                db,
                user_id="student-1",
                interaction_id="study_answer:42",
                scope=scope,
                source_session_id="study-student-1-abc",
            )
            confirmed = confirm_study_answer_event(
                db,
                user_id="student-1",
                interaction_id="study_answer:42",
            )
            spoofed = record_study_answer_event(
                db,
                user_id="student-1",
                interaction_id="client-made-up",
                scope={**scope, "catalog_source": "builtin"},
            )
            states = planning_learning_states(
                db,
                user_id="student-1",
                curriculum_key=self.curriculum["curriculum_key"],
                valid_unit_ids=[item["id"] for item in self.curriculum["units"]],
            )
        finally:
            db.close()
            engine.dispose()

        self.assertTrue(first["recorded"])
        self.assertEqual(first["event_count"], 1)
        self.assertTrue(duplicate["idempotent"])
        self.assertEqual(duplicate["event_count"], 1)
        self.assertTrue(confirmed["idempotent"])
        self.assertIsNone(spoofed)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].status, "learning")
        self.assertIsNone(states[0].mastery_score)


if __name__ == "__main__":
    unittest.main()
