"""Database-agnostic Planning adapter for approved content chapters."""

from __future__ import annotations

import re
from hashlib import sha1
from typing import Any, Dict, Mapping


def _importance(signals: Mapping[str, Any]) -> str:
    score = float(signals.get("importance_score") or 0)
    if score >= 4:
        return "very_high"
    if score >= 3:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


def _exam_relevance(signals: Mapping[str, Any]) -> str:
    score = float(signals.get("exam_weightage_score") or 0)
    if score >= 4:
        return "very_high"
    if score >= 3:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


def _difficulty(signals: Mapping[str, Any]) -> str:
    score = float(signals.get("average_difficulty") or 1)
    if score >= 4:
        return "challenging"
    if score >= 2.5:
        return "steady"
    return "foundation"


def build_published_planning_curriculum(
    chapter_scope: Mapping[str, Any],
) -> Dict[str, Any]:
    """Adapt one approved, source-ordered catalog chapter to roadmap v2."""
    raw_units = list(chapter_scope.get("units") or [])
    chapter_slug = str(chapter_scope["chapter_slug"])
    class_level = str(chapter_scope.get("class_level") or "Class 11")
    subject = str(chapter_scope.get("subject") or "Chemistry")
    scope_key = "_".join(re.findall(r"[a-z0-9]+", f"{class_level}_{subject}".lower()))
    units: list[Dict[str, Any]] = []
    for index, raw_unit in enumerate(raw_units, start=1):
        unit_id = str(raw_unit["id"])
        section_id = f"section_{sha1(unit_id.encode('utf-8')).hexdigest()[:16]}"
        signals = dict(raw_unit.get("focus_signals") or {})
        candidates = list(raw_unit.get("subtopic_candidates") or [])
        concepts: list[Dict[str, Any]] = []
        seen_concepts: set[str] = set()
        for candidate_index, candidate in enumerate(candidates, start=1):
            aliases = [
                str(value)
                for value in candidate.get("concept_ids") or []
                if str(value)
            ]
            concept_id = aliases[0] if aliases else f"{unit_id}_concept_{candidate_index}"
            if concept_id in seen_concepts:
                continue
            seen_concepts.add(concept_id)
            concepts.append(
                {
                    "id": concept_id,
                    "title": str(candidate.get("title") or raw_unit["label"]),
                    "prerequisite_concept_ids": [],
                }
            )
        if not concepts:
            concepts = [
                {
                    "id": unit_id,
                    "title": str(raw_unit["label"]),
                    "prerequisite_concept_ids": [],
                }
            ]
        importance = _importance(signals)
        difficulty = _difficulty(signals)
        exam_relevance = _exam_relevance(signals)
        concept_count = max(1, int(signals.get("concept_count") or len(concepts)))
        minimum = min(30, 10 + (5 * min(concept_count, 4)))
        maximum = min(60, minimum + (10 if difficulty == "challenging" else 5))
        previous_id = str(raw_units[index - 2]["id"]) if index > 1 else ""
        next_id = str(raw_units[index]["id"]) if index < len(raw_units) else ""
        subtopics = [
            {
                "id": str(concept["id"]),
                "title": str(concept["title"]),
                "section_id": section_id,
            }
            for concept in concepts
        ]
        units.append(
            {
                "id": unit_id,
                "order": index,
                "title": str(raw_unit["label"]),
                "short_description": (
                    "A source-ordered learning unit covering "
                    f"{', '.join(item['title'] for item in subtopics[:3])}."
                ),
                "why_it_matters": (
                    "This approved unit preserves the chapter's source order and "
                    "required concept coverage."
                ),
                "primary_topic_id": str(concepts[0]["id"]),
                "importance": importance,
                "difficulty": difficulty,
                "depth": "mastery" if importance in {"very_high", "high"} else "working",
                "exam_relevance": exam_relevance,
                "conceptual_importance": importance,
                "estimated_minutes": {"min": minimum, "max": maximum},
                "ncert_sections": [{"id": section_id, "title": str(raw_unit["label"])}],
                "ncert_subtopics": subtopics,
                "concepts": concepts,
                "skills": ["Explain the central ideas", "Apply the unit in a short check"],
                "learning_types": ["Understand", "Example", "Practice", "Quick Check"],
                "learning_route": ["Understand", "Example", "Practice", "Quick Check"],
                "mastery_criteria": [
                    f"Explain and apply {raw_unit['label']} without relying on notes."
                ],
                "prerequisite_unit_ids": [previous_id] if previous_id else [],
                "dependent_unit_ids": [next_id] if next_id else [],
                "default_status": "not_started",
                "practice": {
                    "minimum_items": 2 if importance in {"very_high", "high"} else 1,
                    "modes": ["recall", "application"],
                    "source_refs": ["Approved NCERT chapter content"],
                },
            }
        )
    return {
        "curriculum_key": f"published_{scope_key}_{chapter_slug}",
        "edition": str(chapter_scope.get("content_version") or "approved_content_pipeline"),
        "board": str(chapter_scope.get("board") or "NCERT"),
        "class_level": class_level,
        "subject": subject,
        "chapter_slug": chapter_slug,
        "chapter_title": str(chapter_scope["chapter_label"]),
        "chapter_number": int(chapter_scope.get("chapter_number") or 1),
        "content_order_locked": True,
        "source": {
            "authority": "approved_content_pipeline",
            "type": "published_chapter",
            "path": chapter_slug,
        },
        "units": units,
    }


__all__ = ["build_published_planning_curriculum"]
