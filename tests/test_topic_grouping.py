import unittest

from Logic.content_pipeline import consolidate_concept_payloads, validate_concept_payloads
from services.topic_grouping import build_learning_units, covered_concept_ids


def _concept(index: int, *, title: str, page: int, formula: str = ""):
    return {
        "concept_id": f"concept_{index}",
        "title": title,
        "definition": f"Grounded definition for {title}.",
        "core_explanation": f"Complete explanation for {title}.",
        "key_points": [f"Key point {index}"],
        "examples": [f"Example {index}"],
        "formulas": [formula] if formula else [],
        "properties": [f"Property {index}"],
        "applications": [f"Application {index}"],
        "common_mistakes": [{"mistake": f"Mistake {index}"}],
        "prerequisites": [],
        "related_concepts": [],
        "learning_objectives": [f"Objective {index}"],
        "source_pages": [page],
        "difficulty_level": 2 + (index % 3),
        "blooms_taxonomy": "Apply",
        "typical_exam_weightage": "medium",
        "importance_level": "core",
    }


class TopicGroupingTests(unittest.TestCase):
    def test_malformed_source_concept_remains_visible_to_import_validation(self):
        valid = _concept(1, title="Valid foundation", page=1)
        malformed = {
            "concept_id": "malformed_concept",
            "title": "Malformed concept",
            "definition": "This source item must not disappear silently.",
            "source_pages": ["not-a-page"],
        }

        compacted = consolidate_concept_payloads(
            [valid, malformed],
            chapter_key="validation_chapter",
            page_count=2,
        )
        validated, issues = validate_concept_payloads(compacted, available_pages=[1, 2])

        self.assertCountEqual(
            [concept.concept_id for concept in validated],
            ["concept_1", "malformed_concept"],
        )
        self.assertTrue(
            any(
                issue.get("concept_id") == "malformed_concept"
                and "missing_source_pages" in issue.get("issues", [])
                for issue in issues
            )
        )
        self.assertTrue(
            any(item.get("concept_id") == "malformed_concept" for item in compacted)
        )

    def test_cross_batch_duplicate_ids_are_disambiguated_without_losing_coverage(self):
        first = _concept(1, title="Atomic mass foundations", page=1)
        second = _concept(2, title="Relative atomic mass applications", page=9)
        second["concept_id"] = first["concept_id"]

        compacted = consolidate_concept_payloads(
            [first, second],
            chapter_key="some_basic_concepts_of_chemistry",
            page_count=9,
        )

        self.assertEqual(len(compacted), 2)
        self.assertEqual(len({unit["concept_id"] for unit in compacted}), 2)
        self.assertEqual(
            {page for unit in compacted for page in unit["source_pages"]},
            {1, 9},
        )
        self.assertEqual(
            {point for unit in compacted for point in unit["key_points"]},
            {"Key point 1", "Key point 2"},
        )
        self.assertTrue(
            all(first["concept_id"] in unit["source_concept_ids"] for unit in compacted)
        )

    def test_dense_chapter_becomes_compact_units_with_full_coverage(self):
        families = ["Foundations", "Laws", "Measurement", "Applications", "Problems"]
        concepts = [
            _concept(
                index,
                title=f"{families[(index - 1) // 4]} concept {index}",
                page=((index - 1) // 2) + 1,
                formula=f"F{index}" if index % 3 == 0 else "",
            )
            for index in range(1, 21)
        ]

        units = build_learning_units(
            concepts,
            chapter_key="class_10_science_sample_chapter",
            page_count=10,
        )

        self.assertEqual(len(units), 5)  # 20 micro-topics -> 5 study units
        self.assertCountEqual(
            covered_concept_ids(units),
            [concept["concept_id"] for concept in concepts],
        )
        self.assertTrue(all(unit["label"] for unit in units))
        self.assertTrue(all(len(unit["concept_ids"]) >= 2 for unit in units))

    def test_long_group_labels_keep_multiple_syllabus_themes_visible(self):
        themes = [
            "Foundations", "Measurement", "Classification", "Uncertainty",
            "Atomic masses", "Mole calculations", "Stoichiometry", "Concentration",
            "Thermochemistry", "Equilibrium", "Redox reactions", "Hydrocarbons",
        ]
        concepts = [
            _concept(
                index,
                title=f"Detailed NCERT {themes[index - 1]} theme with essential chemistry",
                page=index,
            )
            for index in range(1, 13)
        ]

        units = build_learning_units(concepts, chapter_key="chemistry", page_count=12)

        self.assertEqual(len(units), 4)
        self.assertTrue(all(len(unit["label"]) <= 96 for unit in units))
        self.assertTrue(all(" · " in unit["label"] for unit in units))

    def test_complexity_changes_budget_without_returning_to_micro_topics(self):
        simple = [
            _concept(index, title=f"Cell concept {index}", page=index)
            for index in range(1, 13)
        ]
        complex_chapter = [
            _concept(
                index,
                title=f"Electricity concept {index}",
                page=index,
                formula=f"E{index}=x" if index % 2 else "",
            )
            for index in range(1, 78)
        ]
        extreme_chapter = [
            _concept(
                index,
                title=f"Advanced physics concept {index}",
                page=index,
                formula=f"P{index}=x",
            )
            for index in range(1, 98)
        ]

        simple_units = build_learning_units(simple, chapter_key="cells", page_count=12)
        complex_units = build_learning_units(
            complex_chapter,
            chapter_key="electricity",
            page_count=77,
        )
        extreme_units = build_learning_units(
            extreme_chapter,
            chapter_key="advanced_physics",
            page_count=97,
        )

        self.assertEqual(len(simple_units), 4)
        self.assertEqual(len(complex_units), 10)
        self.assertLess(len(complex_units), len(complex_chapter) // 5)
        complex_sizes = [len(unit["concept_ids"]) for unit in complex_units]
        self.assertLessEqual(max(complex_sizes) - min(complex_sizes), 1)
        self.assertLessEqual(max(complex_sizes), 8)
        self.assertGreater(len(extreme_units), 10)
        self.assertLessEqual(
            max(len(unit["concept_ids"]) for unit in extreme_units),
            8,
        )
        self.assertCountEqual(covered_concept_ids(complex_units), [
            concept["concept_id"] for concept in complex_chapter
        ])
        self.assertCountEqual(covered_concept_ids(extreme_units), [
            concept["concept_id"] for concept in extreme_chapter
        ])

    def test_small_chapter_stays_complete_without_forced_consolidation(self):
        concepts = [
            _concept(index, title=f"Meaningful idea {index}", page=index)
            for index in range(1, 7)
        ]

        units = build_learning_units(concepts, chapter_key="small_chapter", page_count=6)

        self.assertEqual(len(units), 6)
        self.assertCountEqual(
            covered_concept_ids(units),
            [concept["concept_id"] for concept in concepts],
        )

    def test_persisted_consolidation_keeps_all_teaching_material_and_aliases(self):
        concepts = [
            _concept(
                index,
                title=f"Atomic mass idea {index}",
                page=index,
                formula=f"M{index}" if index % 2 == 0 else "",
            )
            for index in range(1, 13)
        ]

        compacted = consolidate_concept_payloads(
            concepts,
            chapter_key="some_basic_concepts_of_chemistry",
            page_count=12,
        )

        self.assertEqual(len(compacted), 4)
        self.assertCountEqual(
            [item for unit in compacted for item in unit["source_concept_ids"]],
            [concept["concept_id"] for concept in concepts],
        )
        self.assertEqual(
            {page for unit in compacted for page in unit["source_pages"]},
            set(range(1, 13)),
        )
        self.assertEqual(
            {point for unit in compacted for point in unit["key_points"]},
            {f"Key point {index}" for index in range(1, 13)},
        )
        self.assertEqual(
            {formula for unit in compacted for formula in unit["formulas"]},
            {f"M{index}" for index in range(2, 13, 2)},
        )


if __name__ == "__main__":
    unittest.main()
