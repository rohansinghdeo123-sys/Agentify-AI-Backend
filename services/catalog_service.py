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
from Logic.planning.curriculum_registry import (
    load_planning_curricula,
    resolve_planning_curriculum,
)
from Logic.planning.published_curriculum import build_published_planning_curriculum
from models import ContentChapter, ContentConcept
from services.topic_grouping import build_learning_units

# Mirrors the catalog the frontend shipped with, so removing the hard-coded
# frontend lists never leaves a student with empty selectors.
BUILTIN_SUBJECT = "Chemistry"
BUILTIN_CLASS_LEVEL = "Class 11"
_PLANNING_CURRICULA = load_planning_curricula()


def _planning_catalog_entries(
    published_chapters: Sequence[ContentChapter] = (),
) -> List[Dict[str, Any]]:
    entries = [
        {
            "supported": True,
            "roadmap_version": "planning_roadmap_v2",
            "class_level": curriculum["class_level"],
            "subject": curriculum["subject"],
            "canonical_slug": curriculum["chapter_slug"],
            "name": curriculum["chapter_title"],
            "chapter_number": curriculum["chapter_number"],
            "aliases": list(curriculum.get("aliases", [])),
        }
        for curriculum in _PLANNING_CURRICULA
    ]
    for chapter in published_chapters:
        _, subject_label, class_label = _catalog_group_identity(
            chapter.subject,
            chapter.class_level,
        )
        identity = (
            _normalized_class_level(chapter.class_level),
            normalize_key(chapter.subject),
            normalize_key(chapter.slug),
        )
        registered_curriculum = resolve_planning_curriculum(
            chapter_ref=str(chapter.slug or chapter.chapter_name or ""),
            subject=str(chapter.subject or "") or None,
            class_level=str(chapter.class_level or "") or None,
        )
        if not all(identity) or registered_curriculum is not None:
            continue
        entries.append(
            {
                "supported": True,
                "roadmap_version": "planning_roadmap_v2",
                # Keep Planning capability metadata in the same display scope
                # as ``subjects``.  Release rows intentionally store Class XI
                # as ``11`` while profiles and the student catalog use
                # ``Class 11``; leaking the raw value here made an exact client
                # filter hide every database-backed Planning chapter.
                "class_level": class_label,
                "subject": subject_label,
                "canonical_slug": chapter.slug,
                "name": chapter.chapter_name or chapter.slug,
                "chapter_number": chapter.chapter_number,
                "aliases": [],
                "source": "published",
            }
        )
    return sorted(
        entries,
        key=lambda entry: (
            _normalized_class_level(entry.get("class_level")),
            normalize_key(entry.get("subject")),
            int(entry.get("chapter_number") or 10_000),
            normalize_key(entry.get("name")),
        ),
    )

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
        "planning_chapters": _planning_catalog_entries(),
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


def _planning_value(concept: Any, field: str, default: Any = None) -> Any:
    if isinstance(concept, dict):
        return concept.get(field, default)
    return getattr(concept, field, default)


def _planning_focus_signals(concepts: Sequence[Any]) -> Dict[str, Any]:
    """Return bounded, source-derived signals for Planning prioritisation.

    The brief generator must never infer syllabus facts from unit names alone.
    These values are taken only from approved concept metadata and deliberately
    exclude teaching prose so a fast Planning request stays small.
    """
    importance_rank = {
        "": 0,
        "none": 0,
        "low": 1,
        "supplementary": 1,
        "medium": 2,
        "moderate": 2,
        "important": 3,
        "high": 3,
        "core": 4,
        "essential": 4,
    }
    weightage_rank = {
        "": 0,
        "none": 0,
        "low": 1,
        "medium": 2,
        "moderate": 2,
        "high": 3,
    }

    def ranked(value: Any, ranks: Dict[str, int]) -> int:
        normalized = str(value or "").strip().lower()
        if normalized in ranks:
            return ranks[normalized]
        for label, score in sorted(ranks.items(), key=lambda pair: -pair[1]):
            if label and label in normalized:
                return score
        return 0

    difficulties: List[int] = []
    for concept in concepts:
        try:
            difficulties.append(
                max(1, min(5, int(_planning_value(concept, "difficulty_level", 1) or 1)))
            )
        except (TypeError, ValueError):
            difficulties.append(1)

    return {
        "concept_count": len(concepts),
        "importance_score": max(
            (
                ranked(_planning_value(concept, "importance_level", ""), importance_rank)
                for concept in concepts
            ),
            default=0,
        ),
        "exam_weightage_score": max(
            (
                ranked(
                    _planning_value(concept, "typical_exam_weightage", ""),
                    weightage_rank,
                )
                for concept in concepts
            ),
            default=0,
        ),
        "average_difficulty": round(sum(difficulties) / len(difficulties), 2)
        if difficulties
        else 1.0,
    }


def _planning_concept_aliases(concept: Any) -> List[str]:
    raw = _planning_value(concept, "raw_json", {})
    raw = raw if isinstance(raw, dict) else {}
    candidates = [
        _planning_value(concept, "concept_id", ""),
        *(raw.get("source_concept_ids") or []),
    ]
    return list(
        dict.fromkeys(
            normalized
            for candidate in candidates
            if (normalized := normalize_key(candidate))
        )
    )


def _planning_subtopic_candidates(
    concepts: Sequence[Any],
    fallback: str,
) -> List[Dict[str, Any]]:
    """Carry every approved title and its source signals into Planning.

    Only four representatives are rendered, but retaining this private,
    grounded candidate set lets learner weakness or strong syllabus metadata
    select a later concept instead of accidentally hiding the reason for a
    Deep-focus label.
    """
    candidates: List[Dict[str, Any]] = []
    by_title: Dict[str, Dict[str, Any]] = {}
    for concept in concepts:
        title = " ".join(str(_planning_value(concept, "title", "") or "").split())
        if not title:
            continue
        signals = _planning_focus_signals([concept])
        existing = by_title.get(title)
        if existing is None:
            existing = {
                "title": title,
                "concept_ids": _planning_concept_aliases(concept),
                "focus_signals": signals,
                "source_order": len(candidates),
            }
            by_title[title] = existing
            candidates.append(existing)
            continue

        existing["concept_ids"] = list(
            dict.fromkeys(
                [*existing["concept_ids"], *_planning_concept_aliases(concept)]
            )
        )
        previous = existing["focus_signals"]
        previous["concept_count"] = int(previous.get("concept_count") or 0) + 1
        for field in ("importance_score", "exam_weightage_score", "average_difficulty"):
            previous[field] = max(
                float(previous.get(field) or 0),
                float(signals.get(field) or 0),
            )

    if candidates:
        return candidates
    return [
        {
            "title": fallback,
            "concept_ids": [],
            "focus_signals": _planning_focus_signals([]),
            "source_order": 0,
        }
    ]


def _planning_candidate_score(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("focus_signals") or {}
    return (
        float(signals.get("importance_score") or 0) * 1.6
        + float(signals.get("exam_weightage_score") or 0) * 1.2
        + float(signals.get("average_difficulty") or 1) * 0.45
    )


def _planning_subtopics(concepts: Sequence[Any], fallback: str) -> List[str]:
    """Expose at most four exact, priority-aware approved titles."""
    candidates = _planning_subtopic_candidates(concepts, fallback)
    selected = sorted(
        sorted(
            candidates,
            key=lambda candidate: (
                -_planning_candidate_score(candidate),
                int(candidate["source_order"]),
            ),
        )[:4],
        key=lambda candidate: int(candidate["source_order"]),
    )
    return [str(candidate["title"]) for candidate in selected]


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
        if requested
        in {
            normalize_key(chapter["slug"]),
            normalize_key(chapter["name"]),
            *(normalize_key(alias) for alias in chapter.get("aliases", [])),
        }
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
            "subtopics": _planning_subtopics(unit["concepts"], unit["label"]),
            "subtopic_candidates": _planning_subtopic_candidates(
                unit["concepts"],
                unit["label"],
            ),
            "focus_signals": _planning_focus_signals(unit["concepts"]),
        }
        for unit in grouped
    ]
    if not units:
        units = [
            {
                "id": chapter["slug"],
                "label": chapter["name"],
                "concept_ids": [],
                "subtopics": [chapter["name"]],
                "subtopic_candidates": _planning_subtopic_candidates(
                    [],
                    chapter["name"],
                ),
                "focus_signals": _planning_focus_signals([]),
            }
        ]
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
                "subtopics": _planning_subtopics(unit["concepts"], unit["label"]),
                "subtopic_candidates": _planning_subtopic_candidates(
                    unit["concepts"],
                    unit["label"],
                ),
                "focus_signals": _planning_focus_signals(unit["concepts"]),
            }
            for unit in _learning_units_for_chapter(
                chapter,
                [concept for concept in concepts if concept.concept_id],
            )
        ]
        chapter_slug = chapter.slug or requested
        chapter_label = chapter.chapter_name or chapter_slug
        if not units:
            units = [
                {
                    "id": chapter_slug,
                    "label": chapter_label,
                    "concept_ids": [],
                    "subtopics": [chapter_label],
                    "subtopic_candidates": _planning_subtopic_candidates(
                        [],
                        chapter_label,
                    ),
                    "focus_signals": _planning_focus_signals([]),
                }
            ]
        return {
            "chapter_slug": chapter_slug,
            "chapter_label": chapter_label,
            "chapter_number": chapter.chapter_number,
            "content_version": chapter.version or "",
            "board": chapter.board or "NCERT",
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


def resolve_published_planning_curriculum(
    db: Session,
    *,
    chapter_ref: str,
    subject: Optional[str] = None,
    class_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve one published chapter and return its stable Planning identity."""
    scope = resolve_catalog_chapter_units(
        db,
        chapter_ref=chapter_ref,
        subject=subject,
        class_level=class_level,
    )
    if not scope or scope.get("source") != "published":
        return None
    return build_published_planning_curriculum(scope)


def find_published_planning_curriculum_by_key(
    db: Session,
    curriculum_key: str,
) -> Optional[Dict[str, Any]]:
    """Validate a stored published-curriculum event against current content."""
    requested = str(curriculum_key or "").strip()
    if not requested.startswith("published_"):
        return None
    chapters = (
        db.query(ContentChapter)
        .filter(ContentChapter.status.in_(APPROVED_STATUSES))
        .order_by(ContentChapter.id)
        .all()
    )
    for chapter in chapters:
        curriculum = resolve_published_planning_curriculum(
            db,
            chapter_ref=str(chapter.slug or ""),
            subject=str(chapter.subject or "") or None,
            class_level=str(chapter.class_level or "") or None,
        )
        if curriculum and curriculum["curriculum_key"] == requested:
            return curriculum
    return None


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
        key, subject_label, class_label = _catalog_group_identity(
            chapter.subject,
            chapter.class_level,
        )
        group = groups.setdefault(
            key,
            {
                "subject": subject_label,
                "class_level": class_label,
                "chapters": [],
            },
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

    return {
        "source": "published",
        "planning_chapters": _planning_catalog_entries(chapters),
        "subjects": list(groups.values()),
    }


def _normalized_class_level(value: Any) -> str:
    normalized = normalize_key(value)
    normalized = normalized.removeprefix("class_").removeprefix("grade_")
    return {"xi": "11", "11th": "11"}.get(normalized, normalized)


def _catalog_group_identity(subject: Any, class_level: Any) -> tuple[tuple[str, str], str, str]:
    subject_text = " ".join(str(subject or BUILTIN_SUBJECT).split())
    subject_key = normalize_key(subject_text)
    class_text = " ".join(str(class_level or "").split())
    class_key = _normalized_class_level(class_text)

    manifest_subject_key = normalize_key(BUILTIN_SUBJECT)
    manifest_class_key = _normalized_class_level(BUILTIN_CLASS_LEVEL)
    subject_label = (
        BUILTIN_SUBJECT
        if subject_key == manifest_subject_key
        else subject_text
    )
    class_label = (
        BUILTIN_CLASS_LEVEL
        if class_key == manifest_class_key
        else class_text
    )
    return (subject_key, class_key), subject_label, class_label


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


def _resolved_planning_manifest_topic(
    section_id: str,
    *,
    subject: Optional[str],
    chapter_ref: Optional[str],
    topic: Optional[str],
    class_level: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Resolve the locked chapter map when published concept rows are absent.

    This keeps Planning-to-Study/Practice handoffs on the same canonical IDs.
    Retrieval may still use published content when present; the manifest is the
    deterministic scope and ordering authority, not a replacement teaching
    corpus.
    """
    curriculum = resolve_planning_curriculum(
        chapter_ref=chapter_ref or "",
        subject=subject,
        class_level=class_level,
    )
    if not curriculum:
        return None
    requested = {
        normalize_key(value)
        for value in (section_id, topic)
        if normalize_key(value)
    }
    matches: List[Dict[str, Any]] = []
    for unit in curriculum["units"]:
        keys = {
            normalize_key(unit["id"]),
            normalize_key(unit["title"]),
            normalize_key(unit["primary_topic_id"]),
            *(normalize_key(value) for value in unit.get("legacy_topic_ids") or []),
            *(
                normalize_key(value)
                for concept in unit["concepts"]
                for value in (concept["id"], concept["title"])
            ),
        }
        if requested.intersection(keys):
            matches.append(unit)
    if len(matches) != 1:
        return None
    unit = matches[0]
    return {
        "section_id": unit["primary_topic_id"],
        "topic": unit["title"],
        "concept_ids": list(
            dict.fromkeys(
                [
                    *(concept["id"] for concept in unit["concepts"]),
                    *(unit.get("legacy_topic_ids") or []),
                ]
            )
        ),
        "subject": curriculum["subject"],
        "chapter": curriculum["chapter_title"],
        "chapter_slug": curriculum["chapter_slug"],
        "class_level": curriculum["class_level"],
        "content_version": curriculum["edition"],
        "catalog_source": "planning_manifest",
        "curriculum_key": curriculum["curriculum_key"],
        "planning_unit_id": unit["id"],
    }


def resolve_catalog_topic(
    db: Session,
    section_id: str,
    *,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    topic: Optional[str] = None,
    class_level: Optional[str] = None,
    catalog_source: Optional[str] = None,
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

    explicit_planning_source = normalize_key(catalog_source) == "planning_manifest"
    planning_curriculum = (
        resolve_planning_curriculum(
            chapter_ref=chapter,
            subject=subject,
            class_level=class_level,
        )
        if chapter and explicit_planning_source
        else None
    )
    # A resolved registered curriculum already proves that the explicit
    # chapter reference is either its canonical slug or one of its validated
    # historical aliases.  Accepting those aliases here keeps saved Planning
    # handoffs working without weakening normal Study/Exam catalog lookups.
    explicit_planning_handoff = bool(explicit_planning_source and planning_curriculum)
    if explicit_planning_handoff:
        # The registered manifest is the complete unit boundary. A partial
        # published micro-concept must never narrow Study or Exam retrieval for
        # an explicit Planning handoff.
        return _resolved_planning_manifest_topic(
            section_id,
            subject=subject,
            chapter_ref=chapter,
            topic=topic,
            class_level=class_level,
        )

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
