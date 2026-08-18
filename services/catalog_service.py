"""Student-facing course catalog assembled from the content pipeline.

This is the single source of truth for what students can select in Study
Lab, Exam Mode, and Missions. Chapters come from admin-approved/published
content (ContentChapter + ContentConcept). Fine-grained source concepts are
grouped into complete learning units before they reach selectors. Until the
database has published content, the built-in starter catalog keeps every
learning surface working with the same chapters the app launched with.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from Logic.content_pipeline import APPROVED_STATUSES, normalize_key
from models import ContentChapter, ContentConcept
from services.topic_grouping import build_learning_units

# Mirrors the catalog the frontend shipped with, so removing the hard-coded
# frontend lists never leaves a student with empty selectors.
BUILTIN_SUBJECT = "Chemistry"
BUILTIN_CLASS_LEVEL = "Class 11"
BUILTIN_CHAPTERS: List[Dict[str, Any]] = [
    {
        "slug": "hydrocarbon",
        "name": "Hydrocarbons",
        "chapter_number": None,
        "topics": [
            {"id": "alkanes", "label": "Alkanes"},
            {"id": "alkenes", "label": "Alkenes"},
            {"id": "alkynes", "label": "Alkynes"},
            {"id": "aromatics", "label": "Aromatic Hydrocarbons"},
        ],
    },
    {
        "slug": "matter",
        "name": "Basic Concepts of Chemistry",
        "chapter_number": None,
        "topics": [
            {"id": "chemistry_definition", "label": "Definition of Chemistry"},
            {"id": "historical_alchemy", "label": "Alchemy and Iatrochemistry"},
            {"id": "ancient_indian_chemistry", "label": "Ancient Indian Chemistry"},
            {"id": "importance_of_chemistry", "label": "Role and Importance of Chemistry"},
            {"id": "matter_definition", "label": "Matter Definition"},
            {"id": "properties_of_matter", "label": "Properties of Matter"},
            {"id": "states_of_matter", "label": "States of Matter"},
            {"id": "solid_state", "label": "Solid State"},
            {"id": "liquid_state", "label": "Liquid State"},
            {"id": "gaseous_state", "label": "Gaseous State"},
            {"id": "interconversion_of_states", "label": "Interconversion of States"},
            {"id": "classification_of_matter", "label": "Classification of Matter"},
        ],
    },
]


def _builtin_catalog() -> Dict[str, Any]:
    return {
        "source": "builtin",
        "subjects": [
            {
                "subject": BUILTIN_SUBJECT,
                "class_level": BUILTIN_CLASS_LEVEL,
                "chapters": BUILTIN_CHAPTERS,
            }
        ],
    }


def _concept_alias_ids(concept: ContentConcept) -> List[str]:
    raw = concept.raw_json if isinstance(concept.raw_json, dict) else {}
    candidates = [concept.concept_id, *(raw.get("source_concept_ids") or [])]
    aliases: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_key(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(normalized)
    return aliases


def _learning_units_for_chapter(
    chapter: ContentChapter,
    concepts: Sequence[ContentConcept],
) -> List[Dict[str, Any]]:
    units = build_learning_units(
        concepts,
        chapter_key=chapter.slug or chapter.chapter_name or str(chapter.id),
        page_count=int(chapter.extracted_page_count or chapter.page_count or 0),
    )
    result: List[Dict[str, Any]] = []
    for unit in units:
        member_aliases: List[str] = []
        seen: set[str] = set()
        for concept in unit["concepts"]:
            for alias in _concept_alias_ids(concept):
                if alias not in seen:
                    seen.add(alias)
                    member_aliases.append(alias)
        result.append(
            {
                "id": unit["id"],
                "label": unit["label"],
                "concept_ids": member_aliases,
                "concepts": unit["concepts"],
            }
        )
    return result


def _builtin_chapter_units(
    *,
    chapter_ref: str,
    subject: Optional[str],
    class_level: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Resolve one built-in chapter without relaxing an explicit scope."""
    if subject and normalize_key(subject) != normalize_key(BUILTIN_SUBJECT):
        return None
    if class_level and _normalized_class_level(class_level) != _normalized_class_level(
        BUILTIN_CLASS_LEVEL
    ):
        return None

    requested = normalize_key(chapter_ref)
    matches = [
        chapter
        for chapter in BUILTIN_CHAPTERS
        if requested in {normalize_key(chapter["slug"]), normalize_key(chapter["name"])}
    ]
    if len(matches) != 1:
        return None

    chapter = matches[0]
    concepts = [
        {
            "concept_id": topic["id"],
            "title": topic["label"],
            "source_pages": [],
        }
        for topic in chapter["topics"]
    ]
    grouped = build_learning_units(concepts, chapter_key=chapter["slug"])
    units = [
        {
            "id": unit["id"],
            "label": unit["label"],
            "concept_ids": list(unit["concept_ids"]),
        }
        for unit in grouped
    ]
    if not units:
        units = [{"id": chapter["slug"], "label": chapter["name"], "concept_ids": []}]
    return {
        "chapter_slug": chapter["slug"],
        "chapter_label": chapter["name"],
        "subject": BUILTIN_SUBJECT,
        "class_level": BUILTIN_CLASS_LEVEL,
        "source": "builtin",
        "units": units,
    }


def resolve_catalog_chapter_units(
    db: Session,
    *,
    chapter_ref: str,
    subject: Optional[str] = None,
    class_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve only the requested published chapter and its grouped units.

    Planning calls this targeted resolver instead of assembling the complete
    catalog on every plan request. Explicit subject and class values are hard
    boundaries: a similarly named chapter from another syllabus is never used.
    """
    requested = normalize_key(chapter_ref)
    if not requested:
        return None

    raw_ref = str(chapter_ref or "").strip().lower()
    spaced_ref = requested.replace("_", " ")
    query = db.query(ContentChapter).filter(
        ContentChapter.status.in_(APPROVED_STATUSES)
    )
    if str(subject or "").strip():
        query = query.filter(func.lower(ContentChapter.subject) == str(subject).strip().lower())
    if str(class_level or "").strip():
        requested_class = _normalized_class_level(class_level)
        class_values = {
            str(class_level).strip().lower(),
            requested_class.replace("_", " "),
            f"class {requested_class.replace('_', ' ')}",
        }
        query = query.filter(
            func.lower(ContentChapter.class_level).in_(tuple(sorted(class_values)))
        )

    candidates = (
        query.filter(
            or_(
                func.lower(ContentChapter.slug).in_(tuple(sorted({raw_ref, requested}))),
                func.lower(ContentChapter.chapter_name).in_(
                    tuple(sorted({raw_ref, spaced_ref}))
                ),
            )
        )
        .limit(3)
        .all()
    )
    matches = [
        chapter
        for chapter in candidates
        if requested
        in {
            normalize_key(chapter.slug),
            normalize_key(chapter.chapter_name),
        }
    ]
    if len(matches) > 1:
        return None
    if len(matches) == 1:
        chapter = matches[0]
        concepts = (
            db.query(ContentConcept)
            .filter(ContentConcept.chapter_id == chapter.id)
            .order_by(ContentConcept.id)
            .all()
        )
        units = [
            {
                "id": unit["id"],
                "label": unit["label"],
                "concept_ids": unit["concept_ids"],
            }
            for unit in _learning_units_for_chapter(
                chapter,
                [concept for concept in concepts if concept.concept_id],
            )
        ]
        chapter_slug = chapter.slug or requested
        chapter_label = chapter.chapter_name or chapter_slug
        if not units:
            units = [{"id": chapter_slug, "label": chapter_label, "concept_ids": []}]
        return {
            "chapter_slug": chapter_slug,
            "chapter_label": chapter_label,
            "subject": chapter.subject or subject or "",
            "class_level": chapter.class_level or class_level or "",
            "source": "published",
            "units": units,
        }

    # The starter catalog remains available, but it obeys the same explicit
    # subject/class boundary and exact chapter match.
    return _builtin_chapter_units(
        chapter_ref=chapter_ref,
        subject=subject,
        class_level=class_level,
    )


def build_catalog(db: Session) -> Dict[str, Any]:
    """Published catalog grouped by (subject, class_level); builtin fallback."""
    chapters = (
        db.query(ContentChapter)
        .filter(ContentChapter.status.in_(APPROVED_STATUSES))
        .order_by(ContentChapter.subject, ContentChapter.chapter_number, ContentChapter.id)
        .all()
    )
    if not chapters:
        return _builtin_catalog()

    concept_rows = (
        db.query(ContentConcept)
        .filter(ContentConcept.chapter_id.in_([chapter.id for chapter in chapters]))
        .order_by(ContentConcept.chapter_id, ContentConcept.id)
        .all()
    )
    concepts_by_chapter: Dict[int, List[ContentConcept]] = {}
    for concept in concept_rows:
        concepts_by_chapter.setdefault(concept.chapter_id, []).append(concept)

    groups: Dict[tuple, Dict[str, Any]] = {}
    for chapter in chapters:
        key = (chapter.subject or BUILTIN_SUBJECT, chapter.class_level or "")
        group = groups.setdefault(
            key,
            {"subject": key[0], "class_level": key[1], "chapters": []},
        )
        topics = [
            {
                "id": unit["id"],
                "label": unit["label"],
                "concept_ids": unit["concept_ids"],
            }
            for unit in _learning_units_for_chapter(
                chapter,
                [
                    concept
                    for concept in concepts_by_chapter.get(chapter.id, [])
                    if concept.concept_id
                ],
            )
        ]
        if not topics:
            # A chapter without generated concepts is still searchable by its
            # slug, so students can select the whole chapter as one topic.
            topics = [{"id": chapter.slug, "label": chapter.chapter_name or chapter.slug}]
        group["chapters"].append(
            {
                "slug": chapter.slug,
                "name": chapter.chapter_name or chapter.slug,
                "chapter_number": chapter.chapter_number,
                "topics": topics,
            }
        )

    return {"source": "published", "subjects": list(groups.values())}


def _normalized_class_level(value: Any) -> str:
    normalized = normalize_key(value)
    return normalized.removeprefix("class_")


def _chapter_matches_catalog_scope(
    chapter: ContentChapter,
    *,
    subject: Optional[str],
    chapter_ref: Optional[str],
    class_level: Optional[str],
) -> bool:
    requested_subject = normalize_key(subject)
    if requested_subject and requested_subject != normalize_key(chapter.subject):
        return False

    requested_class = _normalized_class_level(class_level)
    if requested_class and requested_class != _normalized_class_level(chapter.class_level):
        return False

    requested_chapter = normalize_key(chapter_ref)
    if requested_chapter:
        chapter_keys = {
            normalize_key(chapter.slug),
            normalize_key(chapter.chapter_name),
        }
        chapter_haystack = normalize_key(
            f"{chapter.slug} {chapter.chapter_name} chapter {chapter.chapter_number or ''}"
        )
        if requested_chapter not in chapter_keys and requested_chapter not in chapter_haystack:
            return False
    return True


def _resolved_catalog_topic(
    chapter: ContentChapter,
    *,
    section_id: str,
    topic: str,
    concept_ids: Sequence[str],
    subject: Optional[str],
    chapter_ref: Optional[str],
    class_level: Optional[str],
) -> Dict[str, Any]:
    return {
        "section_id": section_id,
        "topic": topic,
        "concept_ids": list(concept_ids),
        "subject": chapter.subject or subject or "",
        "chapter": chapter.chapter_name or chapter_ref or "",
        "chapter_slug": chapter.slug or "",
        "class_level": chapter.class_level or class_level or "",
        "content_version": chapter.version or "",
        "catalog_source": "published",
    }


def resolve_catalog_topic(
    db: Session,
    section_id: str,
    *,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    topic: Optional[str] = None,
    class_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve current units and legacy micro-topic aliases safely.

    Current catalog IDs may represent multiple source concepts. The returned
    ``concept_ids`` keep retrieval grounded across the whole unit. Old IDs and
    titles still resolve directly, including IDs consolidated during a later
    content regeneration.
    """
    requested_keys = {
        normalize_key(value)
        for value in (section_id, topic)
        if normalize_key(value)
    }
    if not requested_keys:
        return None

    chapters = (
        db.query(ContentChapter)
        .filter(ContentChapter.status.in_(APPROVED_STATUSES))
        .all()
    )
    chapters = [
        candidate
        for candidate in chapters
        if _chapter_matches_catalog_scope(
            candidate,
            subject=subject,
            chapter_ref=chapter,
            class_level=None,
        )
    ]
    if not chapters:
        return None

    requested_class = _normalized_class_level(class_level)
    if requested_class and requested_class not in {"other", "general", "unspecified"}:
        preferred_chapters = [
            candidate
            for candidate in chapters
            if _normalized_class_level(candidate.class_level) == requested_class
        ]
        if preferred_chapters:
            chapters = preferred_chapters

    chapter_by_id = {candidate.id: candidate for candidate in chapters}
    concepts = (
        db.query(ContentConcept)
        .filter(ContentConcept.chapter_id.in_(chapter_by_id))
        .order_by(ContentConcept.chapter_id, ContentConcept.id)
        .all()
    )
    requested_section = normalize_key(section_id)
    concepts_by_chapter: Dict[int, List[ContentConcept]] = {}
    for concept in concepts:
        concepts_by_chapter.setdefault(concept.chapter_id, []).append(concept)

    # A stable learning-unit ID is more specific than its generated label,
    # which can intentionally reuse the title of a representative member.
    # Resolve the ID before considering legacy concept-title matches.
    id_unit_matches: List[tuple[ContentChapter, Dict[str, Any]]] = []
    for candidate in chapters:
        for unit in _learning_units_for_chapter(candidate, concepts_by_chapter.get(candidate.id, [])):
            if normalize_key(unit["id"]) == requested_section:
                id_unit_matches.append((candidate, unit))
    if len({candidate.id for candidate, _ in id_unit_matches}) == 1:
        matched_chapter, unit = id_unit_matches[0]
        return _resolved_catalog_topic(
            matched_chapter,
            section_id=unit["id"],
            topic=unit["label"],
            concept_ids=[
                concept.concept_id
                for concept in unit["concepts"]
                if concept.concept_id
            ],
            subject=subject,
            chapter_ref=chapter,
            class_level=class_level,
        )

    direct_matches = [
        concept
        for concept in concepts
        if requested_keys.intersection(
            {
                normalize_key(concept.concept_id),
                normalize_key(concept.title),
                *_concept_alias_ids(concept),
            }
        )
    ]
    if direct_matches:
        stable_id_matches = [
            concept
            for concept in direct_matches
            if normalize_key(concept.concept_id) == requested_section
        ]
        alias_matches = [
            concept
            for concept in direct_matches
            if requested_section in _concept_alias_ids(concept)
        ]
        preferred_matches = stable_id_matches or alias_matches or direct_matches
        if len({concept.chapter_id for concept in preferred_matches}) > 1:
            return None
        preferred_matches.sort(key=lambda concept: concept.id)
        concept = preferred_matches[0]
        matched_chapter = chapter_by_id[concept.chapter_id]
        return _resolved_catalog_topic(
            matched_chapter,
            section_id=concept.concept_id,
            topic=concept.title or concept.concept_id,
            concept_ids=[concept.concept_id],
            subject=subject,
            chapter_ref=chapter,
            class_level=class_level,
        )

    unit_matches: List[tuple[ContentChapter, Dict[str, Any]]] = []
    for candidate in chapters:
        for unit in _learning_units_for_chapter(candidate, concepts_by_chapter.get(candidate.id, [])):
            if requested_keys.intersection(
                {normalize_key(unit["id"]), normalize_key(unit["label"])}
            ):
                unit_matches.append((candidate, unit))

    if len({candidate.id for candidate, _ in unit_matches}) != 1:
        return None
    matched_chapter, unit = unit_matches[0]
    current_ids = [
        concept.concept_id
        for concept in unit["concepts"]
        if concept.concept_id
    ]
    return _resolved_catalog_topic(
        matched_chapter,
        section_id=unit["id"],
        topic=unit["label"],
        concept_ids=current_ids,
        subject=subject,
        chapter_ref=chapter,
        class_level=class_level,
    )
