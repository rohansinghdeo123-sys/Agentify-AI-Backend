import unittest
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Logic.planning.curriculum_registry import resolve_planning_curriculum
from Logic.planning.portfolio_engine import (
    PlanningPortfolioError,
    build_planning_portfolio,
)
from Logic.planning.recommendation_engine import build_planning_roadmap
from models import ContentChapter, ContentConcept, PlanningLearningEvent
from schemas import (
    AutonomousStudyRequest,
    PlanningPortfolioRequest,
    PlanningPortfolioResponse,
)
from services.planning_progress_service import (
    confirm_study_answer_event,
    record_study_answer_event,
)


EMPTY_ANALYTICS = {
    "summary": {},
    "topic_heatmap": [],
    "topic_evidence": [],
    "weak_areas": [],
}


def _selection(chapter_ref, proficiency="know_a_little"):
    return {
        "chapter_ref": chapter_ref,
        "chapter_proficiency": proficiency,
    }


class PlanningPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        PlanningLearningEvent.__table__.create(self.engine)
        ContentChapter.__table__.create(self.engine)
        ContentConcept.__table__.create(self.engine)
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _build(self, selections, **kwargs):
        analytics = kwargs.pop("analytics", EMPTY_ANALYTICS)
        with patch(
            "Logic.planning.portfolio_engine.get_user_analytics",
            return_value=analytics,
        ):
            return build_planning_portfolio(
                self.db,
                user_id="portfolio-student",
                class_level="Class 11",
                subject="Chemistry",
                selections=selections,
                **kwargs,
            )

    def test_request_validates_bounded_chapters_proficiency_and_time(self):
        common = {
            "class_level": "  Class   11 ",
            "subject": " Chemistry ",
        }
        with self.assertRaises(ValidationError):
            PlanningPortfolioRequest(**common, chapters=[])
        with self.assertRaises(ValidationError):
            PlanningPortfolioRequest(
                **common,
                chapters=[_selection(f"chapter-{index}") for index in range(7)],
            )
        with self.assertRaises(ValidationError):
            PlanningPortfolioRequest(
                **common,
                chapters=[_selection("Structure of Atom", "expert")],
            )

        payload = PlanningPortfolioRequest(
            **common,
            chapters=[_selection("Structure of Atom", "New to It")],
            study_time_today="2 hours",
            session_duration_minutes=60,
        )
        self.assertEqual(payload.class_level, "Class 11")
        self.assertEqual(payload.subject, "Chemistry")
        self.assertEqual(payload.chapters[0].chapter_proficiency, "new_to_it")
        self.assertEqual(payload.study_time_today, "120_plus")

        for raw, expected in (
            ("15 minutes", "15"),
            ("30 min", "30"),
            ("1 hour", "60"),
        ):
            with self.subTest(time_alias=raw):
                normalized = PlanningPortfolioRequest(
                    **common,
                    chapters=[_selection("Structure of Atom")],
                    study_time_today=raw,
                )
                self.assertEqual(normalized.study_time_today, expected)

    def test_same_canonical_chapter_is_deduplicated_across_aliases(self):
        portfolio = self._build(
            [
                _selection("Some Basic Concepts of Chemistry"),
                _selection("matter"),
            ]
        )
        self.assertEqual(portfolio["requested_chapter_count"], 2)
        self.assertEqual(portfolio["chapter_count"], 1)
        self.assertEqual(portfolio["deduplicated_chapter_count"], 1)
        self.assertEqual(
            portfolio["chapters"][0]["chapter_slug"],
            "some_basic_concepts_of_chemistry",
        )
        PlanningPortfolioResponse.model_validate(portfolio)

    def test_conflicting_duplicate_proficiency_is_rejected(self):
        with self.assertRaisesRegex(PlanningPortfolioError, "different proficiency"):
            self._build(
                [
                    _selection("Structure of Atom", "new_to_it"),
                    _selection("Atomic Structure", "mostly_confident"),
                ]
            )

    def test_each_chapter_keeps_its_own_proficiency_strategy(self):
        portfolio = self._build(
            [
                _selection("Some Basic Concepts of Chemistry", "new_to_it"),
                _selection("Structure of Atom", "mostly_confident"),
            ],
            study_time_today="no_limit",
        )
        chapters = {item["chapter_slug"]: item for item in portfolio["chapters"]}
        foundations = chapters["some_basic_concepts_of_chemistry"]
        confident = chapters["structure_of_atom"]
        self.assertEqual(foundations["chapter_proficiency"], "new_to_it")
        self.assertEqual(confident["chapter_proficiency"], "mostly_confident")
        self.assertIn("Guided example", foundations["next_step"]["approach"])
        self.assertIn("Diagnostic", confident["next_step"]["approach"])
        self.assertEqual(
            [unit["order"] for unit in foundations["learning_units"]],
            list(range(1, len(foundations["learning_units"]) + 1)),
        )
        self.assertEqual(
            [unit["order"] for unit in confident["learning_units"]],
            list(range(1, len(confident["learning_units"]) + 1)),
        )

    def test_global_time_ceiling_is_applied_once_not_per_chapter(self):
        portfolio = self._build(
            [
                _selection("Some Basic Concepts of Chemistry"),
                _selection("Structure of Atom"),
            ],
            study_time_today="15",
            session_duration_minutes=60,
        )
        route = portfolio["today_route"]
        self.assertEqual(route["source"], "student_choice")
        self.assertEqual(route["budget_minutes"], 15)
        self.assertEqual(route["total_minutes"], 15)
        self.assertEqual(sum(item["minutes"] for item in route["items"]), 15)
        self.assertEqual(len({item["chapter_slug"] for item in route["items"]}), 1)
        self.assertEqual(sum(item["selected_for_today"] for item in portfolio["chapters"]), 1)

        one_hour = self._build(
            [
                _selection("Some Basic Concepts of Chemistry"),
                _selection("Structure of Atom"),
            ],
            session_duration_minutes=60,
        )
        self.assertEqual(one_hour["today_route"]["source"], "session_state")
        self.assertLessEqual(one_hour["today_route"]["total_minutes"], 60)
        self.assertEqual(
            one_hour["today_route"]["total_minutes"],
            sum(item["minutes"] for item in one_hour["today_route"]["items"]),
        )

    def test_global_next_step_matches_the_selected_chapter_and_is_explainable(self):
        structure = resolve_planning_curriculum(
            chapter_ref="Structure of Atom",
            subject="Chemistry",
            class_level="Class 11",
        )
        first = structure["units"][0]
        analytics = {
            **EMPTY_ANALYTICS,
            "topic_heatmap": [
                {
                    "topic": first["primary_topic_id"],
                    "value": 35,
                    "attempts": 4,
                }
            ],
        }
        portfolio = self._build(
            [
                _selection("Some Basic Concepts of Chemistry"),
                _selection("Structure of Atom"),
            ],
            analytics=analytics,
        )
        selected = next(item for item in portfolio["chapters"] if item["selected_for_today"])
        global_next = portfolio["global_next_step"]
        self.assertEqual(selected["chapter_slug"], "structure_of_atom")
        self.assertEqual(global_next["chapter_slug"], selected["chapter_slug"])
        self.assertEqual(global_next["unit_id"], selected["next_step"]["unit_id"])
        self.assertEqual(global_next["estimated_minutes"], selected["next_step"]["estimated_minutes"])
        self.assertEqual(
            {item["id"] for item in portfolio["selection_factors"]},
            {
                "status_urgency",
                "chapter_continuity",
                "importance",
                "exam_relevance",
                "effort_sizing",
                "selection_order",
            },
        )
        self.assertTrue(global_next["selection_reason"])
        PlanningPortfolioResponse.model_validate(portfolio)

    def test_existing_single_chapter_contract_and_engine_are_unchanged(self):
        request = AutonomousStudyRequest(
            current_chapter="Structure of Atom",
            subject="Chemistry",
            class_level="Class 11",
            chapter_proficiency="know_the_basics",
            study_time_today="30",
        )
        curriculum = resolve_planning_curriculum(
            chapter_ref=request.current_chapter,
            subject=request.subject,
            class_level=request.class_level,
        )
        roadmap = build_planning_roadmap(
            curriculum,
            chapter_proficiency=request.chapter_proficiency,
            study_time_today=request.study_time_today,
        )
        self.assertEqual(request.current_chapter, "Structure of Atom")
        self.assertEqual(roadmap["roadmap_version"], "planning_roadmap_v2")
        self.assertEqual(roadmap["chapter_slug"], "structure_of_atom")
        self.assertEqual(roadmap["daily_route"]["budget_minutes"], 30)

    def test_published_chapter_without_manifest_joins_multi_chapter_portfolio(self):
        chapter = ContentChapter(
            slug="thermodynamics",
            subject="Chemistry",
            class_level="Class 11",
            chapter_name="Thermodynamics",
            chapter_number=5,
            status="published",
            version="ncert-2026",
            extracted_page_count=32,
        )
        self.db.add(chapter)
        self.db.flush()
        self.db.add_all(
            [
                ContentConcept(
                    chapter_id=chapter.id,
                    concept_id="system_and_surroundings",
                    title="System and Surroundings",
                    source_pages=[1, 2],
                    importance_level="high",
                    typical_exam_weightage="medium",
                    difficulty_level=2,
                ),
                ContentConcept(
                    chapter_id=chapter.id,
                    concept_id="first_law_of_thermodynamics",
                    title="First Law of Thermodynamics",
                    source_pages=[8, 9],
                    importance_level="essential",
                    typical_exam_weightage="high",
                    difficulty_level=4,
                ),
            ]
        )
        self.db.commit()

        portfolio = self._build(
            [
                _selection("Some Basic Concepts of Chemistry", "new_to_it"),
                _selection("thermodynamics", "mostly_confident"),
            ],
            study_time_today="30",
        )

        self.assertEqual(portfolio["chapter_count"], 2)
        published = next(
            item for item in portfolio["chapters"] if item["chapter_slug"] == "thermodynamics"
        )
        self.assertEqual(published["curriculum"]["edition"], "ncert-2026")
        self.assertEqual(published["chapter_proficiency"], "mostly_confident")
        self.assertTrue(published["learning_units"])
        self.assertEqual(
            [unit["order"] for unit in published["learning_units"]],
            list(range(1, len(published["learning_units"]) + 1)),
        )
        PlanningPortfolioResponse.model_validate(portfolio)

        first_unit = published["learning_units"][0]
        receipt = record_study_answer_event(
            self.db,
            user_id="portfolio-student",
            interaction_id="published-coach-turn-1",
            scope={
                "catalog_source": "published",
                "chapter_slug": "thermodynamics",
                "subject": "Chemistry",
                "class_level": "Class 11",
                "section_id": first_unit["primary_topic_id"],
            },
            source_session_id="coach-portfolio-student-thermodynamics",
        )
        self.assertIsNotNone(receipt)
        self.assertTrue(receipt["recorded"])
        self.assertEqual(receipt["unit_id"], first_unit["id"])
        self.assertEqual(
            confirm_study_answer_event(
                self.db,
                user_id="portfolio-student",
                interaction_id="published-coach-turn-1",
            )["unit_id"],
            first_unit["id"],
        )

        refreshed = self._build([_selection("thermodynamics", "mostly_confident")])
        refreshed_unit = refreshed["chapters"][0]["learning_units"][0]
        self.assertEqual(refreshed_unit["status"], "learning")


if __name__ == "__main__":
    unittest.main()
