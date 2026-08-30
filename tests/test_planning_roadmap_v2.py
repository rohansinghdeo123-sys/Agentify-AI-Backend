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
    load_planning_curricula,
    load_planning_curriculum,
    resolve_planning_curriculum,
)
from Logic.planning.recommendation_engine import (
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
    resolve_learning_event_scope,
)


PROFICIENCIES = (
    "new_to_it",
    "know_a_little",
    "know_the_basics",
    "mostly_confident",
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


def _mastery_signals(units):
    heatmap = []
    evidence = []
    for unit in units:
        topic = unit["primary_topic_id"]
        heatmap.append(
            {
                "topic": topic,
                "value": 90,
                "attempts": max(2, int(unit["practice"]["minimum_items"])),
            }
        )
        evidence.extend(
            [
                {
                    "topic": topic,
                    "session_type": "revision",
                    "confidence_after": 82,
                },
                {
                    "topic": topic,
                    "session_type": "exam",
                    "confidence_after": 86,
                },
            ]
        )
    return {"topic_heatmap": heatmap, "topic_evidence": evidence}


def _aliases(curriculum):
    return {
        alias: unit["id"]
        for unit in curriculum["units"]
        for alias in unit.get("legacy_topic_ids") or []
    }


class PlanningRoadmapV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curriculum = load_planning_curriculum()
        cls.curricula = load_planning_curricula()
        cls.structure = next(item for item in cls.curricula if item["chapter_number"] == 2)

    def test_manifests_group_subtopics_without_losing_ncert_order(self):
        unit_one = self.curriculum
        self.assertTrue(unit_one["content_order_locked"])
        self.assertEqual(len(unit_one["units"]), 6)
        self.assertEqual(
            [unit["order"] for unit in unit_one["units"]],
            list(range(1, 7)),
        )
        top_level = [
            subtopic["id"]
            for unit in unit_one["units"]
            for subtopic in unit["ncert_subtopics"]
            if subtopic["id"] in {f"1.{index}" for index in range(1, 11)}
        ]
        self.assertEqual(top_level, [f"1.{index}" for index in range(1, 11)])
        self.assertTrue(all(unit["ncert_subtopics"] for unit in unit_one["units"]))

        self.assertEqual(len(self.structure["units"]), 7)
        structure_ids = {
            subtopic["id"]
            for unit in self.structure["units"]
            for subtopic in unit["ncert_subtopics"]
        }
        for required in ("2.1", "2.1.4", "2.2.5", "2.3.3", "2.4.2", "2.5.2", "2.6.6"):
            self.assertIn(required, structure_ids)

        for curriculum in self.curricula:
            order = {unit["id"]: unit["order"] for unit in curriculum["units"]}
            reverse = {unit["id"]: [] for unit in curriculum["units"]}
            for unit in curriculum["units"]:
                for prerequisite in unit["prerequisite_unit_ids"]:
                    self.assertLess(order[prerequisite], unit["order"])
                    reverse[prerequisite].append(unit["id"])
            self.assertEqual(
                {unit["id"]: unit["dependent_unit_ids"] for unit in curriculum["units"]},
                reverse,
            )

    def test_manifest_validator_fails_closed_on_reordering_dependency_or_source_drift(self):
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

        bad_subtopic = deepcopy(self.curriculum)
        bad_subtopic["units"][0]["ncert_subtopics"][0]["section_id"] = "missing"
        with self.assertRaises(PlanningCurriculumError):
            _validate_manifest(bad_subtopic)

    def test_registry_resolves_both_chapters_and_rejects_wrong_scope(self):
        cases = (
            ("matter", "some_basic_concepts_of_chemistry"),
            ("Structure of Atom", "structure_of_atom"),
        )
        for chapter_ref, expected_slug in cases:
            with self.subTest(chapter=chapter_ref):
                resolved = resolve_planning_curriculum(
                    chapter_ref=chapter_ref,
                    subject="Chemistry",
                    class_level="Class 11",
                )
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved["chapter_slug"], expected_slug)
        self.assertIsNone(
            resolve_planning_curriculum(
                chapter_ref="structure_of_atom",
                subject="Physics",
                class_level="Class 11",
            )
        )

    def test_request_exposes_proficiency_student_time_choice_and_optional_session_state(self):
        common = {
            "current_chapter": "some_basic_concepts_of_chemistry",
            "subject": "Chemistry",
            "class_level": "Class 11",
        }
        for retired in (
            "available_time",
            "exam_target",
            "current_knowledge",
            "learning_goal",
            "preferred_style",
            "prerequisite_confidence",
        ):
            self.assertNotIn(retired, AutonomousStudyRequest.model_fields)

        for proficiency in PROFICIENCIES:
            with self.subTest(proficiency=proficiency):
                request = AutonomousStudyRequest(**common, chapter_proficiency=proficiency)
                self.assertEqual(request.chapter_proficiency, proficiency)

        self.assertIsNone(AutonomousStudyRequest(**common).session_duration_minutes)
        self.assertIsNone(AutonomousStudyRequest(**common).study_time_today)
        for raw, expected in (
            ("15", "15"),
            (30, "30"),
            ("60", "60"),
            (120, "120_plus"),
            ("120+", "120_plus"),
            ("2 hours", "120_plus"),
            ("2+ hours", "120_plus"),
            ("no_limit", "no_limit"),
            ("unlimited", "no_limit"),
        ):
            with self.subTest(time=raw):
                self.assertEqual(
                    AutonomousStudyRequest(**common, study_time_today=raw).study_time_today,
                    expected,
                )
        self.assertEqual(
            AutonomousStudyRequest(**common, session_duration_minutes=15).session_duration_minutes,
            15,
        )
        self.assertEqual(
            AutonomousStudyRequest(**common, session_duration_minutes=60).session_duration_minutes,
            60,
        )
        for invalid_duration in (14, 121):
            with self.subTest(duration=invalid_duration), self.assertRaises(ValueError):
                AutonomousStudyRequest(**common, session_duration_minutes=invalid_duration)
        with self.assertRaises(ValueError):
            AutonomousStudyRequest(**common, study_time_today="45")
        self.assertEqual(
            AutonomousStudyRequest(**common, learning_goal="fast_track").chapter_proficiency,
            "mostly_confident",
        )
        self.assertEqual(
            AutonomousStudyRequest(**common, learning_goal="quick_revision").chapter_proficiency,
            "mostly_confident",
        )

    def test_time_is_a_ceiling_and_never_padding_target(self):
        units = self.curriculum["units"]
        for mastered_count in range(len(units)):
            analytics = _mastery_signals(units[:mastered_count])
            with self.subTest(mastered=mastered_count):
                roadmap = build_planning_roadmap(self.curriculum, analytics=analytics)
                route = roadmap["daily_route"]
                self.assertEqual(route["source"], "default_focus")
                self.assertLessEqual(route["total_minutes"], 30)
                self.assertIsNone(roadmap["study_time_today"])
                self.assertIsNone(roadmap["session_duration_minutes"])
                self.assertEqual(route["budget_minutes"], 30)
                self.assertEqual(
                    route["total_minutes"],
                    sum(item["minutes"] for item in route["items"]),
                )
                self.assertEqual({item["unit_id"] for item in route["items"]}, {roadmap["next_step"]["unit_id"]})
                self.assertEqual(roadmap["next_step"]["estimated_minutes"], route["estimated_minutes"])

        one_hour = build_planning_roadmap(
            self.curriculum,
            chapter_proficiency="know_the_basics",
            session_duration_minutes=60,
        )
        self.assertEqual(one_hour["daily_route"]["source"], "session_state")
        self.assertEqual(one_hour["daily_route"]["budget_minutes"], 60)
        self.assertLessEqual(one_hour["daily_route"]["total_minutes"], 60)
        self.assertLess(one_hour["daily_route"]["total_minutes"], 60)
        self.assertEqual(one_hour["session_duration_minutes"], 60)

        for duration in (47, 48, 49):
            with self.subTest(ambient_duration=duration):
                ambient = build_planning_roadmap(
                    self.curriculum,
                    session_duration_minutes=duration,
                )
                self.assertEqual(ambient["session_duration_minutes"], duration)
                self.assertEqual(ambient["daily_route"]["source"], "session_state")
                self.assertEqual(ambient["daily_route"]["budget_minutes"], 45)
                self.assertLessEqual(ambient["daily_route"]["budget_minutes"], duration)
                self.assertLessEqual(ambient["daily_route"]["total_minutes"], duration)

    def test_every_student_time_choice_is_a_ceiling_and_choice_wins_over_session_state(self):
        budgets = {"15": 15, "30": 30, "60": 60, "120_plus": 120, "no_limit": None}
        first_id = self.curriculum["units"][0]["id"]
        for preference, budget in budgets.items():
            with self.subTest(preference=preference):
                roadmap = build_planning_roadmap(
                    self.curriculum,
                    study_time_today=preference,
                    session_duration_minutes=60,
                )
                route = roadmap["daily_route"]
                self.assertEqual(roadmap["study_time_today"], preference)
                self.assertEqual(roadmap["session_duration_minutes"], 60)
                self.assertEqual(route["source"], "student_choice")
                self.assertEqual(route["budget_minutes"], budget)
                self.assertEqual({item["unit_id"] for item in route["items"]}, {first_id})
                self.assertEqual(route["total_minutes"], sum(item["minutes"] for item in route["items"]))
                if budget is not None:
                    self.assertLessEqual(route["total_minutes"], budget)

        selected = build_planning_roadmap(
            self.curriculum,
            study_time_today="15",
            session_duration_minutes=60,
        )
        self.assertEqual(selected["daily_route"]["source"], "student_choice")
        self.assertEqual(selected["daily_route"]["budget_minutes"], 15)
        self.assertEqual(selected["daily_route"]["total_minutes"], 15)

    def test_fifteen_minute_content_stays_fifteen_for_all_large_or_unlimited_choices(self):
        compact = deepcopy(self.curriculum)
        unit = deepcopy(compact["units"][0])
        unit["estimated_minutes"] = {"min": 15, "max": 15}
        unit["prerequisite_unit_ids"] = []
        unit["dependent_unit_ids"] = []
        compact["units"] = [unit]

        for preference in (None, "15", "30", "60", "120_plus", "no_limit"):
            with self.subTest(preference=preference):
                roadmap = build_planning_roadmap(compact, study_time_today=preference)
                route = roadmap["daily_route"]
                self.assertEqual(route["total_minutes"], 15)
                self.assertEqual(route["estimated_minutes"], {"min": 15, "max": 15})
                self.assertEqual(roadmap["next_step"]["estimated_minutes"], {"min": 15, "max": 15})
                self.assertEqual(sum(item["minutes"] for item in route["items"]), 15)
                self.assertEqual(route["items"][0]["scope"], "full_unit")

    def test_fifteen_minute_cap_on_broad_unit_is_truthfully_partial(self):
        roadmap = build_planning_roadmap(self.curriculum, study_time_today="15")
        route = roadmap["daily_route"]
        self.assertEqual(route["total_minutes"], 15)
        self.assertEqual(route["estimated_minutes"], {"min": 15, "max": 15})
        self.assertEqual([item["minutes"] for item in route["items"]], [10, 5])
        self.assertEqual(route["items"][0]["scope"], "partial")
        self.assertEqual(route["items"][1]["role"], "quick_check")

    def test_no_limit_remains_one_content_sized_unit_and_completed_recall_stays_short(self):
        unlimited = build_planning_roadmap(self.curriculum, study_time_today="no_limit")
        route = unlimited["daily_route"]
        first = self.curriculum["units"][0]
        self.assertIsNone(route["budget_minutes"])
        self.assertEqual(route["source"], "student_choice")
        self.assertEqual({item["unit_id"] for item in route["items"]}, {first["id"]})
        self.assertGreaterEqual(route["total_minutes"], first["estimated_minutes"]["min"])
        self.assertLessEqual(route["total_minutes"], first["estimated_minutes"]["max"])

        complete = build_planning_roadmap(
            self.curriculum,
            study_time_today="120_plus",
            analytics=_mastery_signals(self.curriculum["units"]),
        )
        self.assertEqual(complete["daily_route"]["total_minutes"], 10)
        self.assertEqual(complete["daily_route"]["estimated_minutes"], {"min": 5, "max": 10})
        self.assertEqual(complete["next_step"]["estimated_minutes"], {"min": 5, "max": 10})

    def test_all_proficiencies_change_strategy_without_reordering_curriculum(self):
        expected_order = [unit["id"] for unit in self.curriculum["units"]]
        activities = {}
        approaches = {}
        for proficiency in PROFICIENCIES:
            roadmap = build_planning_roadmap(
                self.curriculum,
                chapter_proficiency=proficiency,
            )
            self.assertEqual(roadmap["chapter_proficiency"], proficiency)
            self.assertEqual(
                [unit["id"] for unit in roadmap["learning_units"]],
                expected_order,
            )
            activities[proficiency] = roadmap["daily_route"]["items"][0]["activity"]
            approaches[proficiency] = tuple(roadmap["next_step"]["approach"])
        self.assertEqual(len(set(activities.values())), 4)
        self.assertEqual(len(set(approaches.values())), 4)
        self.assertIn("guided", activities["new_to_it"].lower())
        self.assertIn("diagnose", activities["mostly_confident"].lower())

    def test_exam_emphasis_requires_no_exam_target_and_never_reorders(self):
        high_exam_index = next(
            index
            for index, unit in enumerate(self.curriculum["units"])
            if unit["exam_relevance"] == "very_high"
        )
        analytics = _mastery_signals(self.curriculum["units"][:high_exam_index])
        roadmap = build_planning_roadmap(
            self.curriculum,
            chapter_proficiency="know_the_basics",
            analytics=analytics,
        )
        next_unit = roadmap["learning_units"][high_exam_index]
        self.assertEqual(roadmap["next_step"]["unit_id"], next_unit["id"])
        self.assertEqual(roadmap["next_step"]["importance"], next_unit["importance"])
        self.assertEqual(next_unit["exam_relevance"], "very_high")
        self.assertIn("Application", roadmap["next_step"]["approach"])
        self.assertNotIn("exam_target", AutonomousStudyRequest.model_fields)

    def test_performance_overrides_self_report_and_repairs_prerequisites(self):
        first, second = self.curriculum["units"][:2]
        weak = {
            "topic_heatmap": [
                {"topic": first["primary_topic_id"], "value": 35, "attempts": 4}
            ]
        }
        confident_but_struggling = build_planning_roadmap(
            self.curriculum,
            chapter_proficiency="mostly_confident",
            analytics=weak,
        )
        self.assertEqual(confident_but_struggling["next_step"]["unit_id"], first["id"])
        self.assertEqual(confident_but_struggling["learning_units"][0]["status"], "needs_review")
        self.assertIn("Review", confident_but_struggling["daily_route"]["items"][0]["activity"])
        self.assertEqual(
            confident_but_struggling["daily_route"]["items"][0]["role"],
            "main_focus",
        )

        beginner_but_strong = build_planning_roadmap(
            self.curriculum,
            chapter_proficiency="new_to_it",
            analytics=_mastery_signals([first]),
        )
        self.assertEqual(beginner_but_strong["learning_units"][0]["status"], "mastered")
        self.assertEqual(beginner_but_strong["next_step"]["unit_id"], second["id"])

    def test_mastery_requires_multiple_evidence_dimensions_and_can_regress(self):
        first = self.curriculum["units"][0]
        attempts = first["practice"]["minimum_items"]
        score_only = {
            "topic_heatmap": [
                {"topic": first["primary_topic_id"], "value": 92, "attempts": attempts}
            ]
        }
        self.assertEqual(
            derive_unit_statuses(self.curriculum, score_only)[first["id"]],
            "practising",
        )

        low_confidence_is_not_rescaled = {
            **score_only,
            "topic_evidence": [
                {
                    "topic": first["primary_topic_id"],
                    "session_type": "study",
                    "confidence_after": 1,
                }
            ],
        }
        self.assertEqual(
            derive_unit_statuses(
                self.curriculum,
                low_confidence_is_not_rescaled,
            )[first["id"]],
            "practising",
        )

        production_exam_evidence = {
            **score_only,
            "topic_evidence": [
                {
                    "topic": first["primary_topic_id"],
                    "session_type": "study_exam",
                    "confidence_after": 1,
                }
            ],
        }
        self.assertEqual(
            derive_unit_statuses(
                self.curriculum,
                production_exam_evidence,
            )[first["id"]],
            "mastered",
        )

        demonstrated = _mastery_signals([first])
        self.assertEqual(
            derive_unit_statuses(self.curriculum, demonstrated)[first["id"]],
            "mastered",
        )

        persisted_mastery = [
            SimpleNamespace(
                unit_id=first["id"],
                status="mastered",
                evidence_count=max(2, attempts),
                mastery_score=90,
            )
        ]
        newer_struggle = {
            "topic_heatmap": [
                {"topic": first["primary_topic_id"], "value": 42, "attempts": attempts}
            ]
        }
        self.assertEqual(
            derive_unit_statuses(self.curriculum, newer_struggle, persisted_mastery)[first["id"]],
            "needs_review",
        )

    def test_recommended_next_current_and_route_state_never_contradict(self):
        roadmap = build_planning_roadmap(self.curriculum)
        next_id = roadmap["next_step"]["unit_id"]
        recommended = [unit for unit in roadmap["learning_units"] if unit["status"] == "recommended"]
        self.assertEqual([unit["id"] for unit in recommended], [next_id])
        self.assertEqual(roadmap["progress"]["recommended_units"], 1)
        self.assertEqual(roadmap["daily_route"]["items"][0]["unit_id"], next_id)
        self.assertEqual(
            {item["role"] for item in roadmap["daily_route"]["items"][:2]},
            {"main_focus", "quick_check"},
        )
        self.assertTrue(all(item["scope"] != "complete" for item in roadmap["daily_route"]["items"]))
        self.assertTrue(roadmap["next_step"]["outcome"])
        self.assertGreaterEqual(len(roadmap["next_step"]["approach"]), 2)

    def test_rapid_mastery_moves_forward_and_completed_chapter_returns_recall(self):
        units = self.curriculum["units"]
        nearly_done = build_planning_roadmap(
            self.curriculum,
            analytics=_mastery_signals(units[:-1]),
        )
        self.assertEqual(nearly_done["next_step"]["unit_id"], units[-1]["id"])

        complete = build_planning_roadmap(
            self.curriculum,
            analytics=_mastery_signals(units),
        )
        self.assertEqual(complete["progress"]["percentage"], 100)
        self.assertEqual(complete["daily_route"]["items"][0]["role"], "quick_check")
        self.assertNotIn("complete", complete["daily_route"]["items"][0]["scope"])

    def test_completion_guidance_is_derived_for_each_chapter(self):
        for curriculum in (self.curriculum, self.structure):
            with self.subTest(chapter=curriculum["chapter_slug"]):
                roadmap = build_planning_roadmap(curriculum)
                guidance = " ".join(roadmap["completion_criteria"])
                self.assertIn(curriculum["units"][0]["title"], guidance)
                self.assertIn(curriculum["units"][-1]["title"], guidance)

        structure_guidance = " ".join(
            build_planning_roadmap(self.structure)["completion_criteria"]
        ).lower()
        self.assertNotIn("stoichiometry", structure_guidance)

    def test_registered_run_is_deterministic_and_response_contract_validates(self):
        db = SimpleNamespace(commit=lambda: None)
        coach = SimpleNamespace()
        analytics = {"summary": {}, "weak_areas": [], "topic_heatmap": [], "topic_evidence": []}
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
                chapter_proficiency="mostly_confident",
                study_time_today="15",
                session_duration_minutes=60,
            )

        complete.assert_not_called()
        self.assertEqual(mission["roadmap_version"], "planning_roadmap_v2")
        self.assertEqual(mission["chapter_proficiency"], "mostly_confident")
        self.assertEqual(mission["study_time_today"], "15")
        self.assertEqual(mission["session_duration_minutes"], 60)
        self.assertEqual(mission["daily_route"]["source"], "student_choice")
        self.assertEqual(mission["daily_route"]["total_minutes"], 15)
        self.assertEqual(mission["estimated_minutes"], 15)
        self.assertEqual(mission["student_state"]["study_time_today"], "15")
        self.assertEqual(mission["student_state"]["planned_minutes"], 15)
        self.assertEqual(coach.last_recommendation["study_time_today"], "15")
        self.assertEqual(coach.last_recommendation["session_duration_minutes"], 60)
        self.assertEqual(
            [unit["id"] for unit in mission["learning_units"]],
            mission["coverage"]["included_unit_ids"],
        )
        AutonomousStudyResponse(**mission)

    def test_catalog_is_additive_and_lists_both_planning_chapters(self):
        engine, db = _catalog_session()
        try:
            catalog = build_catalog(db)
            planning_slugs = [item["canonical_slug"] for item in catalog["planning_chapters"]]
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

        self.assertEqual(
            planning_slugs,
            ["some_basic_concepts_of_chemistry", "structure_of_atom"],
        )
        self.assertEqual(resolved["catalog_source"], "planning_manifest")
        self.assertEqual(resolved["chapter_slug"], "some_basic_concepts_of_chemistry")

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
        finally:
            db.close()
            engine.dispose()

        published = [chapter for chapter in chemistry["chapters"] if chapter["slug"] == "matter"]
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["topics"][0]["id"], "chemistry_definition")

    def test_every_learning_unit_retrieves_only_registered_pdf_slices(self):
        for curriculum in self.curricula:
            for unit in curriculum["units"]:
                with self.subTest(chapter=curriculum["chapter_number"], unit=unit["id"]):
                    result = search_knowledge_base(
                        unit["primary_topic_id"],
                        unit["title"],
                        scope={
                            "catalog_source": "planning_manifest",
                            "chapter_slug": curriculum["chapter_slug"],
                            "subject": curriculum["subject"],
                            "class_level": curriculum["class_level"],
                            "planning_unit_id": unit["id"],
                            "section_id": unit["primary_topic_id"],
                        },
                    )
                    allowed_pages = {segment["page"] for segment in unit["source_segments"]}
                    self.assertNotIn("error", result)
                    self.assertEqual(result["unit_id"], unit["id"])
                    self.assertTrue(set(result["source_pages"]).issubset(allowed_pages))
                    self.assertGreater(result["paragraphs_found"], 0)

    def test_study_evidence_is_idempotent_and_legacy_unit_events_migrate(self):
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
            direct_scope = resolve_learning_event_scope(scope)
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

            legacy_unit_id = next(
                alias
                for alias in unit["legacy_topic_ids"]
                if alias.startswith("chem11_u01_lu")
            )
            db.add(
                PlanningLearningEvent(
                    user_id="student-1",
                    curriculum_key=self.curriculum["curriculum_key"],
                    unit_id=legacy_unit_id,
                    interaction_id="legacy-study-answer",
                    event_type="study_answer",
                    source_session_id="legacy",
                )
            )
            db.commit()
            states = planning_learning_states(
                db,
                user_id="student-1",
                curriculum_key=self.curriculum["curriculum_key"],
                valid_unit_ids=[item["id"] for item in self.curriculum["units"]],
                unit_aliases=_aliases(self.curriculum),
            )
        finally:
            db.close()
            engine.dispose()

        self.assertTrue(first["recorded"])
        self.assertEqual(direct_scope["unit_id"], unit["id"])
        self.assertEqual(first["event_count"], 1)
        self.assertTrue(duplicate["idempotent"])
        self.assertTrue(confirmed["idempotent"])
        state = next(item for item in states if item.unit_id == unit["id"])
        self.assertEqual(state.evidence_count, 2)
        self.assertEqual(state.status, "learning")
        self.assertIsNone(state.mastery_score)


if __name__ == "__main__":
    unittest.main()
