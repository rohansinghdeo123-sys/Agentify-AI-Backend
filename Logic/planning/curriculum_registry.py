"""Load and validate deterministic Planning curriculum manifests.

The manifest, rather than an LLM, defines what exists, NCERT order, source
mapping, and dependencies.  Validation is intentionally strict so a malformed
content update fails closed instead of silently giving a student a reordered or
incomplete chapter.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRICULUM_PATH = (
    BACKEND_ROOT
    / "data"
    / "planning"
    / "ncert"
    / "class_11"
    / "chemistry"
    / "some_basic_concepts_of_chemistry.json"
)
PLANNING_CURRICULUM_ROOT = BACKEND_ROOT / "data" / "planning"

IMPORTANCE_VALUES = {"very_high", "high", "moderate", "low"}
DIFFICULTY_VALUES = {"foundation", "steady", "challenging"}
DEPTH_VALUES = {"overview", "working", "mastery"}
STATUS_VALUES = {"not_started", "learning", "practising", "needs_review", "mastered"}


class PlanningCurriculumError(ValueError):
    """Raised when checked-in Planning curriculum data is invalid."""


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _normalized_class_level(value: Any) -> str:
    normalized = _normalized(value).removeprefix("class_").removeprefix("grade_")
    return {"xi": "11", "11th": "11"}.get(normalized, normalized)


def _require_string(payload: Dict[str, Any], key: str, *, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanningCurriculumError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _require_string_list(payload: Dict[str, Any], key: str, *, location: str) -> List[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise PlanningCurriculumError(f"{location}.{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _ensure_unique(values: Iterable[str], *, location: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise PlanningCurriculumError(f"{location} contains duplicate identifiers")


def _validate_source(source: Any) -> Dict[str, Any]:
    if not isinstance(source, dict):
        raise PlanningCurriculumError("source must be an object")
    for key in ("authority", "type", "path"):
        _require_string(source, key, location="source")
    page_count = source.get("page_count")
    if not isinstance(page_count, int) or page_count < 1:
        raise PlanningCurriculumError("source.page_count must be a positive integer")
    sha256 = source.get("sha256")
    if sha256 is not None and (
        not isinstance(sha256, str) or not re.fullmatch(r"[A-Fa-f0-9]{64}", sha256.strip())
    ):
        raise PlanningCurriculumError("source.sha256 must be a 64-character hexadecimal digest")
    return source


def _verify_source_file(source: Dict[str, Any]) -> None:
    """Verify that a manifest still points at the checked-in source artifact."""
    source_path = (BACKEND_ROOT / source["path"]).resolve()
    try:
        source_path.relative_to(BACKEND_ROOT)
    except ValueError as exc:
        raise PlanningCurriculumError("source.path must remain inside the backend repository") from exc
    if not source_path.is_file():
        raise PlanningCurriculumError(f"Planning curriculum source is missing: {source_path}")

    expected_digest = str(source.get("sha256") or "").strip().lower()
    if expected_digest:
        digest = sha256()
        with source_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected_digest:
            raise PlanningCurriculumError(
                "Planning curriculum source digest does not match source.sha256"
            )


def _validate_manifest(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanningCurriculumError("Planning curriculum must be a JSON object")

    for key in (
        "curriculum_key",
        "edition",
        "board",
        "class_level",
        "subject",
        "chapter_slug",
        "chapter_title",
    ):
        _require_string(payload, key, location="curriculum")
    _validate_source(payload.get("source"))

    chapter_number = payload.get("chapter_number")
    if not isinstance(chapter_number, int) or chapter_number < 1:
        raise PlanningCurriculumError("curriculum.chapter_number must be a positive integer")
    if payload.get("content_order_locked") is not True:
        raise PlanningCurriculumError("curriculum.content_order_locked must be true")

    aliases = payload.get("aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise PlanningCurriculumError("curriculum.aliases must be a list of strings")

    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise PlanningCurriculumError("curriculum.units must contain at least one learning unit")

    unit_ids: List[str] = []
    legacy_topic_ids: List[str] = []
    concept_positions: Dict[str, tuple[int, int]] = {}
    section_ids: List[str] = []
    for unit_index, unit in enumerate(units, start=1):
        location = f"units[{unit_index - 1}]"
        if not isinstance(unit, dict):
            raise PlanningCurriculumError(f"{location} must be an object")
        unit_id = _require_string(unit, "id", location=location)
        unit_ids.append(unit_id)
        if unit.get("order") != unit_index:
            raise PlanningCurriculumError(
                f"{location}.order must be {unit_index}; NCERT order cannot contain gaps"
            )
        for key in ("title", "short_description", "why_it_matters", "primary_topic_id"):
            _require_string(unit, key, location=location)

        importance = _require_string(unit, "importance", location=location)
        difficulty = _require_string(unit, "difficulty", location=location)
        depth = _require_string(unit, "depth", location=location)
        if importance not in IMPORTANCE_VALUES:
            raise PlanningCurriculumError(f"{location}.importance is unsupported")
        if difficulty not in DIFFICULTY_VALUES:
            raise PlanningCurriculumError(f"{location}.difficulty is unsupported")
        if depth not in DEPTH_VALUES:
            raise PlanningCurriculumError(f"{location}.depth is unsupported")
        for key in ("exam_relevance", "conceptual_importance"):
            if _require_string(unit, key, location=location) not in IMPORTANCE_VALUES:
                raise PlanningCurriculumError(f"{location}.{key} is unsupported")

        estimate = unit.get("estimated_minutes")
        if not isinstance(estimate, dict):
            raise PlanningCurriculumError(f"{location}.estimated_minutes must be an object")
        minimum, maximum = estimate.get("min"), estimate.get("max")
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 5
            or maximum < minimum
            or maximum > 180
        ):
            raise PlanningCurriculumError(
                f"{location}.estimated_minutes must be a realistic ordered integer range"
            )

        sections = unit.get("ncert_sections")
        if not isinstance(sections, list) or not sections:
            raise PlanningCurriculumError(f"{location}.ncert_sections must not be empty")
        for section_index, section in enumerate(sections):
            section_location = f"{location}.ncert_sections[{section_index}]"
            if not isinstance(section, dict):
                raise PlanningCurriculumError(f"{section_location} must be an object")
            section_ids.append(_require_string(section, "id", location=section_location))
            _require_string(section, "title", location=section_location)
            pages = section.get("pages")
            if not isinstance(pages, dict):
                raise PlanningCurriculumError(f"{section_location}.pages must be an object")
            start, end = pages.get("start"), pages.get("end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
                or end > int(payload["source"]["page_count"])
            ):
                raise PlanningCurriculumError(f"{section_location}.pages is outside the source")

        source_segments = unit.get("source_segments")
        if not isinstance(source_segments, list) or not source_segments:
            raise PlanningCurriculumError(f"{location}.source_segments must not be empty")
        allowed_source_pages = {
            page_number
            for section in sections
            for page_number in range(
                int(section["pages"]["start"]),
                int(section["pages"]["end"]) + 1,
            )
        }
        segment_identities: List[tuple[int, str, str]] = []
        for segment_index, segment in enumerate(source_segments):
            segment_location = f"{location}.source_segments[{segment_index}]"
            if not isinstance(segment, dict):
                raise PlanningCurriculumError(f"{segment_location} must be an object")
            if set(segment).difference({"page", "start_anchor", "end_anchor"}):
                raise PlanningCurriculumError(
                    f"{segment_location} contains unsupported source-slice fields"
                )
            page_number = segment.get("page")
            if not isinstance(page_number, int) or page_number not in allowed_source_pages:
                raise PlanningCurriculumError(
                    f"{segment_location}.page must belong to the unit's NCERT sections"
                )
            anchors: Dict[str, str] = {}
            for anchor_key in ("start_anchor", "end_anchor"):
                anchor = segment.get(anchor_key)
                if anchor is None:
                    anchors[anchor_key] = ""
                    continue
                if (
                    not isinstance(anchor, str)
                    or not anchor.strip()
                    or len(anchor.strip()) > 240
                ):
                    raise PlanningCurriculumError(
                        f"{segment_location}.{anchor_key} must be a bounded non-empty string"
                    )
                anchors[anchor_key] = anchor.strip()
            segment_identities.append(
                (
                    page_number,
                    anchors["start_anchor"],
                    anchors["end_anchor"],
                )
            )
        if len(segment_identities) != len(set(segment_identities)):
            raise PlanningCurriculumError(f"{location}.source_segments contains duplicates")

        concepts = unit.get("concepts")
        if not isinstance(concepts, list) or not concepts:
            raise PlanningCurriculumError(f"{location}.concepts must not be empty")
        concept_ids: List[str] = []
        for concept_index, concept in enumerate(concepts):
            concept_location = f"{location}.concepts[{concept_index}]"
            if not isinstance(concept, dict):
                raise PlanningCurriculumError(f"{concept_location} must be an object")
            concept_id = _require_string(concept, "id", location=concept_location)
            _require_string(concept, "title", location=concept_location)
            _require_string_list(
                concept,
                "prerequisite_concept_ids",
                location=concept_location,
            )
            if concept_id in concept_positions:
                raise PlanningCurriculumError(f"concept id {concept_id!r} is duplicated")
            concept_positions[concept_id] = (unit_index, concept_index)
            concept_ids.append(concept_id)
        if unit["primary_topic_id"] not in concept_ids:
            raise PlanningCurriculumError(
                f"{location}.primary_topic_id must reference a concept in the same unit"
            )
        legacy_ids = unit.get("legacy_topic_ids", [])
        if not isinstance(legacy_ids, list) or any(
            not isinstance(legacy_id, str) or not legacy_id.strip()
            for legacy_id in legacy_ids
        ):
            raise PlanningCurriculumError(
                f"{location}.legacy_topic_ids must be a list of non-empty strings"
            )
        legacy_topic_ids.extend(legacy_id.strip() for legacy_id in legacy_ids)

        _require_string_list(unit, "skills", location=location)
        _require_string_list(unit, "learning_types", location=location)
        learning_route = _require_string_list(unit, "learning_route", location=location)
        if not 2 <= len(learning_route) <= 4:
            raise PlanningCurriculumError(
                f"{location}.learning_route must contain 2 to 4 ordered steps"
            )
        mastery_criteria = _require_string_list(unit, "mastery_criteria", location=location)
        if not mastery_criteria:
            raise PlanningCurriculumError(f"{location}.mastery_criteria must not be empty")
        _require_string_list(unit, "prerequisite_unit_ids", location=location)
        _require_string_list(unit, "dependent_unit_ids", location=location)
        if unit.get("default_status", "not_started") not in STATUS_VALUES:
            raise PlanningCurriculumError(f"{location}.default_status is unsupported")

        practice = unit.get("practice")
        if not isinstance(practice, dict):
            raise PlanningCurriculumError(f"{location}.practice must be an object")
        if not isinstance(practice.get("minimum_items"), int) or practice["minimum_items"] < 0:
            raise PlanningCurriculumError(f"{location}.practice.minimum_items must be non-negative")
        _require_string_list(practice, "modes", location=f"{location}.practice")
        _require_string_list(practice, "source_refs", location=f"{location}.practice")

    _ensure_unique(unit_ids, location="curriculum unit ids")
    _ensure_unique(section_ids, location="NCERT section ids")
    normalized_canonical_ids = {
        _normalized(identifier)
        for identifier in [*unit_ids, *concept_positions]
    }
    normalized_legacy_ids = [_normalized(identifier) for identifier in legacy_topic_ids]
    _ensure_unique(normalized_legacy_ids, location="legacy topic ids")
    if normalized_canonical_ids.intersection(normalized_legacy_ids):
        raise PlanningCurriculumError(
            "legacy topic ids must not collide with curriculum unit or concept ids"
        )
    unit_order = {unit_id: index for index, unit_id in enumerate(unit_ids, start=1)}

    reverse_dependencies: Dict[str, List[str]] = {unit_id: [] for unit_id in unit_ids}
    for unit in units:
        unit_id = unit["id"]
        for prerequisite_id in unit["prerequisite_unit_ids"]:
            if prerequisite_id not in unit_order:
                raise PlanningCurriculumError(
                    f"unit {unit_id!r} references unknown prerequisite {prerequisite_id!r}"
                )
            if unit_order[prerequisite_id] >= unit_order[unit_id]:
                raise PlanningCurriculumError(
                    f"unit {unit_id!r} prerequisite {prerequisite_id!r} must precede it in NCERT order"
                )
            reverse_dependencies[prerequisite_id].append(unit_id)
    for unit in units:
        expected = reverse_dependencies[unit["id"]]
        if unit["dependent_unit_ids"] != expected:
            raise PlanningCurriculumError(
                f"unit {unit['id']!r}.dependent_unit_ids must exactly reverse prerequisites"
            )

    for unit_index, unit in enumerate(units, start=1):
        for concept_index, concept in enumerate(unit["concepts"]):
            for prerequisite_id in concept["prerequisite_concept_ids"]:
                prerequisite_position = concept_positions.get(prerequisite_id)
                if prerequisite_position is None:
                    raise PlanningCurriculumError(
                        f"concept {concept['id']!r} references unknown prerequisite {prerequisite_id!r}"
                    )
                if prerequisite_position >= (unit_index, concept_index):
                    raise PlanningCurriculumError(
                        f"concept {concept['id']!r} prerequisite must occur earlier in NCERT order"
                    )

    unit_by_id = {str(unit["id"]): unit for unit in units}
    ancestor_cache: Dict[str, set[str]] = {}

    def dependency_ancestors(unit_id: str) -> set[str]:
        if unit_id in ancestor_cache:
            return ancestor_cache[unit_id]
        ancestors: set[str] = set()
        for prerequisite_id in unit_by_id[unit_id]["prerequisite_unit_ids"]:
            ancestors.add(prerequisite_id)
            ancestors.update(dependency_ancestors(prerequisite_id))
        ancestor_cache[unit_id] = ancestors
        return ancestors

    for unit_index, unit in enumerate(units, start=1):
        unit_id = str(unit["id"])
        ancestors = dependency_ancestors(unit_id)
        for concept in unit["concepts"]:
            for prerequisite_id in concept["prerequisite_concept_ids"]:
                prerequisite_unit_index = concept_positions[prerequisite_id][0]
                if prerequisite_unit_index == unit_index:
                    continue
                prerequisite_unit_id = unit_ids[prerequisite_unit_index - 1]
                if prerequisite_unit_id not in ancestors:
                    raise PlanningCurriculumError(
                        f"unit {unit_id!r} dependency graph does not cover concept prerequisite "
                        f"{prerequisite_id!r} from unit {prerequisite_unit_id!r}"
                    )

    return payload


@lru_cache(maxsize=8)
def _load_cached(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanningCurriculumError(f"Planning curriculum is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanningCurriculumError(f"Planning curriculum is not valid JSON: {path}") from exc
    curriculum = _validate_manifest(payload)
    _verify_source_file(curriculum["source"])
    return curriculum


def _discover_manifest_paths(root_path: Path | str = PLANNING_CURRICULUM_ROOT) -> List[Path]:
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise PlanningCurriculumError(f"Planning curriculum directory is missing: {root}")
    paths = sorted(
        path.resolve()
        for path in root.rglob("*.json")
        if path.name.lower() not in {"index.json", "registry.json"}
        and not path.name.startswith("_")
    )
    if not paths:
        raise PlanningCurriculumError(f"No Planning curriculum manifests found under: {root}")
    return paths


def _validate_registry(curricula: Sequence[Dict[str, Any]]) -> None:
    curriculum_keys: Dict[str, str] = {}
    scope_aliases: Dict[tuple[str, str, str], str] = {}
    chapter_numbers: Dict[tuple[str, str, int], str] = {}
    for curriculum in curricula:
        curriculum_key = str(curriculum["curriculum_key"])
        normalized_key = _normalized(curriculum_key)
        if normalized_key in curriculum_keys:
            raise PlanningCurriculumError(
                f"Duplicate Planning curriculum key: {curriculum_key!r}"
            )
        curriculum_keys[normalized_key] = curriculum_key

        scope = (
            _normalized_class_level(curriculum["class_level"]),
            _normalized(curriculum["subject"]),
        )
        chapter_number_key = (*scope, int(curriculum["chapter_number"]))
        if chapter_number_key in chapter_numbers:
            raise PlanningCurriculumError(
                "Planning curricula cannot share a chapter number within one class/subject scope"
            )
        chapter_numbers[chapter_number_key] = curriculum_key

        identities = {
            _normalized(curriculum["chapter_slug"]),
            _normalized(curriculum["chapter_title"]),
            *(_normalized(alias) for alias in curriculum.get("aliases", [])),
        }
        for identity in identities:
            alias_key = (*scope, identity)
            previous = scope_aliases.get(alias_key)
            if previous and previous != curriculum_key:
                raise PlanningCurriculumError(
                    f"Planning chapter alias {identity!r} is ambiguous within its syllabus scope"
                )
            scope_aliases[alias_key] = curriculum_key


@lru_cache(maxsize=4)
def _load_registry_cached(path_values: tuple[str, ...]) -> tuple[Dict[str, Any], ...]:
    curricula = tuple(_load_cached(path_value) for path_value in path_values)
    _validate_registry(curricula)
    return curricula


def load_planning_curriculum(path: Path | str = DEFAULT_CURRICULUM_PATH) -> Dict[str, Any]:
    """Return a defensive copy of one validated curriculum manifest."""
    return deepcopy(_load_cached(str(Path(path).resolve())))


def load_planning_curricula(
    root_path: Path | str = PLANNING_CURRICULUM_ROOT,
) -> List[Dict[str, Any]]:
    """Discover every checked-in manifest and return a validated registry."""
    paths = _discover_manifest_paths(root_path)
    curricula = _load_registry_cached(tuple(str(path) for path in paths))
    return deepcopy(list(curricula))


def clear_curriculum_cache() -> None:
    """Test/admin hook for reloading a manifest after an approved update."""
    _load_cached.cache_clear()
    _load_registry_cached.cache_clear()


def resolve_planning_curriculum(
    *,
    chapter_ref: str,
    subject: Optional[str],
    class_level: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Resolve the registered chapter without weakening explicit syllabus scope."""
    requested_class = _normalized_class_level(class_level)
    requested = _normalized(chapter_ref)
    matches: List[Dict[str, Any]] = []
    for curriculum in load_planning_curricula():
        if subject and _normalized(subject) != _normalized(curriculum["subject"]):
            continue
        if (
            requested_class
            and requested_class not in {"other", "general", "unspecified"}
            and requested_class != _normalized_class_level(curriculum["class_level"])
        ):
            continue
        allowed = {
            _normalized(curriculum["chapter_slug"]),
            _normalized(curriculum["chapter_title"]),
            *(_normalized(alias) for alias in curriculum.get("aliases", [])),
        }
        canonical = _normalized(curriculum["chapter_slug"])
        is_scoped_published_slug = requested.endswith(f"_{canonical}")
        if requested in allowed or is_scoped_published_slug:
            matches.append(curriculum)
    return matches[0] if len(matches) == 1 else None
