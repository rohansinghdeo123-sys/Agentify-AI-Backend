"""Build compact, syllabus-ordered learning units from fine-grained concepts.

The content pipeline intentionally keeps every source-grounded detail.  The
student catalog, however, should expose concepts at the size of a meaningful
study session rather than one entry per heading, definition, example, or
formula.  This module provides a deterministic grouping layer that:

* keeps every source concept in exactly one learning unit;
* uses source-page proximity, title/content overlap and declared concept
  relationships to choose boundaries;
* varies the unit budget with the size and complexity of the chapter; and
* creates stable IDs so saved selections survive catalog refreshes.

The functions accept either dictionaries (LLM payloads) or ORM-like objects so
the same rules can be used while ingesting new chapters and while presenting
older, already-published chapters.
"""

from __future__ import annotations

import math
import re
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Sequence, Set


_TITLE_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "with",
    # These describe the shape of a micro-topic rather than its subject matter.
    "application", "applications", "classification", "definition", "example",
    "examples", "explanation", "formula", "formulas", "introduction", "overview",
    "practice", "properties", "property", "summary", "types",
}

_IMPORTANCE_RANK = {
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

# Study and revision retrieval normally request eight source blocks. Keeping a
# learning unit within that boundary means every member remains available to a
# complete-unit workflow without weakening the compact chapter experience.
_MAX_CONCEPTS_PER_UNIT = 8


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _normalized_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _stem(term: str) -> str:
    """Tiny deterministic stemmer sufficient for catalog-title similarity."""
    if len(term) > 5 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("es"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _terms(value: Any) -> Set[str]:
    terms: Set[str] = set()
    for term in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if len(term) <= 2 or term in _TITLE_STOPWORDS:
            continue
        terms.add(_stem(term))
    return terms


def _source_pages(item: Any) -> List[int]:
    pages: Set[int] = set()
    for value in _as_list(_value(item, "source_pages", [])):
        try:
            page = int(value)
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(value or ""))
            page = int(match.group()) if match else 0
        if page > 0:
            pages.add(page)
    return sorted(pages)


def _related_ids(item: Any) -> Set[str]:
    related: Set[str] = set()
    for value in _as_list(_value(item, "related_concepts", [])):
        if isinstance(value, dict):
            candidate = value.get("concept_id") or value.get("id") or value.get("concept")
        else:
            candidate = value
        normalized = _normalized_id(candidate)
        if normalized:
            related.add(normalized)
    return related


def _concept_id(item: Any, index: int = 0) -> str:
    return _normalized_id(_value(item, "concept_id")) or f"concept_{index + 1}"


def _concept_title(item: Any, index: int = 0) -> str:
    title = str(_value(item, "title") or "").strip()
    return title or _concept_id(item, index).replace("_", " ").title()


def _importance(item: Any) -> int:
    value = str(_value(item, "importance_level") or "").strip().lower()
    if value in _IMPORTANCE_RANK:
        return _IMPORTANCE_RANK[value]
    if "high" in value or "core" in value or "essential" in value:
        return 4
    if "medium" in value or "moderate" in value:
        return 2
    return 1


def _difficulty(item: Any) -> int:
    try:
        return max(1, min(5, int(_value(item, "difficulty_level", 1) or 1)))
    except (TypeError, ValueError):
        return 1


def meaningful_topic_budget(concepts: Sequence[Any], *, page_count: int = 0) -> int:
    """Return a chapter-aware target, not one universal topic count.

    Small, already-meaningful chapters are left intact.  Larger chapters use
    their grounded page span, concept volume, formula density and difficulty to
    earn additional units, with a compact upper guardrail that prevents a long
    micro-topic checklist from resurfacing.
    """
    count = len(concepts)
    if count <= 6:
        return count

    referenced_pages = {
        page
        for concept in concepts
        for page in _source_pages(concept)
    }
    teaching_pages = max(int(page_count or 0), len(referenced_pages))

    # Concept volume catches dense source pages; page volume catches long
    # chapters whose LLM extraction was already conservative.
    concept_units = max(3, math.ceil(math.sqrt(count)))
    # A chapter page span is a second signal that prevents the compactor from
    # folding several distinct NCERT sections into one opaque mega-card. Four
    # teaching pages per unit still yields a short roadmap, while long chapters
    # can reach the ten-unit guardrail when their syllabus genuinely needs it.
    page_units = math.ceil(teaching_pages / 4) if teaching_pages else 0
    target = max(concept_units, page_units)

    formula_count = sum(len(_as_list(_value(item, "formulas", []))) for item in concepts)
    average_difficulty = sum(_difficulty(item) for item in concepts) / max(count, 1)
    high_importance = sum(1 for item in concepts if _importance(item) >= 3)
    if count >= 14 and (formula_count >= count // 2 or average_difficulty >= 3.5):
        target += 1
    if count >= 24 and high_importance >= count * 0.6:
        target += 1

    compact_target = max(4, min(10, target))
    retrieval_target = math.ceil(count / _MAX_CONCEPTS_PER_UNIT)
    return min(count, max(compact_target, retrieval_target))


def _page_bounds(group: Dict[str, Any]) -> tuple[int, int]:
    pages = group["pages"]
    if not pages:
        return (0, 0)
    return (min(pages), max(pages))


def _merge_affinity(left: Dict[str, Any], right: Dict[str, Any], ideal_size: float) -> float:
    left_pages = left["pages"]
    right_pages = right["pages"]
    page_union = left_pages | right_pages
    page_overlap = len(left_pages & right_pages) / max(len(page_union), 1)
    left_start, left_end = _page_bounds(left)
    right_start, _ = _page_bounds(right)
    page_gap = max(0, right_start - left_end) if left_end and right_start else 1

    term_union = left["terms"] | right["terms"]
    term_overlap = len(left["terms"] & right["terms"]) / max(len(term_union), 1)
    relationship = bool(
        left["ids"].intersection(right["related"])
        or right["ids"].intersection(left["related"])
    )

    score = page_overlap * 7.0 + term_overlap * 6.0
    score += 4.0 if relationship else 0.0
    if page_gap == 0:
        score += 3.0
    elif page_gap == 1:
        score += 2.0
    elif page_gap == 2:
        score += 0.75
    else:
        score -= min(4.0, page_gap * 0.5)

    combined_size = len(left["members"]) + len(right["members"])
    if combined_size > max(2.0, ideal_size * 1.65):
        score -= (combined_size - ideal_size) * 0.8
    return score


def _merge_groups(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "members": left["members"] + right["members"],
        "indexes": left["indexes"] + right["indexes"],
        "ids": left["ids"] | right["ids"],
        "titles": left["titles"] + right["titles"],
        "terms": left["terms"] | right["terms"],
        "pages": left["pages"] | right["pages"],
        "related": left["related"] | right["related"],
    }


def _balanced_content_groups(
    singleton_groups: Sequence[Dict[str, Any]],
    target: int,
    ideal_size: float,
) -> List[Dict[str, Any]]:
    """Partition ordered concepts into balanced, content-aware units.

    A greedy best-neighbour merge can strand singleton tail units while making
    earlier units too large for normal retrieval limits.  Restricting each
    contiguous segment to ``floor(n/target)`` or ``ceil(n/target)`` members
    guarantees balanced units. Dynamic programming then places the boundaries
    at the weakest adjacent content affinities, preserving chapter-aware
    grouping rather than slicing at arbitrary fixed offsets.
    """
    count = len(singleton_groups)
    if target >= count:
        return list(singleton_groups)

    smaller = count // target
    larger = math.ceil(count / target)
    allowed_sizes = sorted({smaller, larger})
    adjacent_affinity = [
        _merge_affinity(singleton_groups[index], singleton_groups[index + 1], ideal_size)
        for index in range(count - 1)
    ]
    affinity_prefix = [0.0]
    for affinity in adjacent_affinity:
        affinity_prefix.append(affinity_prefix[-1] + affinity)

    # (groups_built, concepts_used) -> (cohesion_score, segment_sizes)
    states: Dict[tuple[int, int], tuple[float, List[int]]] = {(0, 0): (0.0, [])}
    for groups_built in range(target):
        for (built, start), (score, sizes) in list(states.items()):
            if built != groups_built:
                continue
            for size in allowed_sizes:
                end = start + size
                remaining_groups = target - groups_built - 1
                remaining_concepts = count - end
                if end > count:
                    continue
                if not (
                    remaining_groups * smaller
                    <= remaining_concepts
                    <= remaining_groups * larger
                ):
                    continue
                # Internal affinity for [start, end); boundary affinities are
                # deliberately excluded so weak joins become natural cuts.
                cohesion = affinity_prefix[max(start, end - 1)] - affinity_prefix[start]
                candidate = (score + cohesion, [*sizes, size])
                key = (groups_built + 1, end)
                existing = states.get(key)
                if existing is None or candidate[0] > existing[0]:
                    states[key] = candidate

    _, sizes = states[(target, count)]
    result: List[Dict[str, Any]] = []
    start = 0
    for size in sizes:
        merged = singleton_groups[start]
        for group in singleton_groups[start + 1 : start + size]:
            merged = _merge_groups(merged, group)
        result.append(merged)
        start += size
    return result


def _group_label(group: Dict[str, Any]) -> str:
    members = group["members"]
    titles = [
        _concept_title(member, index)
        for index, member in zip(group["indexes"], members)
    ]
    if len(titles) == 1:
        return titles[0]

    # Pick titles that add the most new subject-matter terms.  This naturally
    # collapses variants such as Definition/Properties/Examples while keeping
    # distinct syllabus ideas visible in the unit name.
    ranked_indexes = sorted(
        range(len(members)),
        key=lambda index: (
            -_importance(members[index]),
            -len(_terms(titles[index])),
            group["indexes"][index],
        ),
    )
    selected: List[int] = []
    covered: Set[str] = set()
    for index in ranked_indexes:
        title_terms = _terms(titles[index])
        if selected and title_terms and title_terms <= covered:
            continue
        selected.append(index)
        covered.update(title_terms)
        if len(selected) == 3:
            break
    if not selected:
        selected = [0]
    selected.sort()

    chosen = [titles[index] for index in selected]
    label = " & ".join(chosen)
    if len(label) > 96 and len(chosen) > 1:
        separator = " · "
        per_title = max(20, (96 - len(separator) * (len(chosen) - 1)) // len(chosen))
        compacted: List[str] = []
        for title in chosen:
            if len(title) <= per_title:
                compacted.append(title)
                continue
            prefix = title[: max(1, per_title - 1)].rsplit(" ", 1)[0].rstrip(" ,&-")
            compacted.append(f"{prefix or title[: per_title - 1]}…")
        label = separator.join(compacted)
    if len(label) > 96:
        label = f"{label[:93].rstrip(' ,&-')}…"
    return label


def build_learning_units(
    concepts: Sequence[Any],
    *,
    chapter_key: str = "",
    page_count: int = 0,
) -> List[Dict[str, Any]]:
    """Group concepts into stable, ordered learning units with full coverage."""
    if not concepts:
        return []

    ordered = sorted(
        enumerate(concepts),
        key=lambda pair: (
            min(_source_pages(pair[1]) or [10**9]),
            max(_source_pages(pair[1]) or [10**9]),
            pair[0],
        ),
    )
    groups: List[Dict[str, Any]] = []
    for original_index, concept in ordered:
        concept_id = _concept_id(concept, original_index)
        title = _concept_title(concept, original_index)
        groups.append(
            {
                "members": [concept],
                "indexes": [original_index],
                "ids": {concept_id},
                "titles": [title],
                "terms": _terms(
                    " ".join(
                        [
                            title,
                            str(_value(concept, "definition") or ""),
                            " ".join(map(str, _as_list(_value(concept, "learning_objectives", [])))),
                        ]
                    )
                ),
                "pages": set(_source_pages(concept)),
                "related": _related_ids(concept),
            }
        )

    target = meaningful_topic_budget(concepts, page_count=page_count)
    ideal_size = len(groups) / max(target, 1)
    groups = _balanced_content_groups(groups, target, ideal_size)

    units: List[Dict[str, Any]] = []
    normalized_chapter = _normalized_id(chapter_key) or "chapter"
    for group in groups:
        concept_ids = [
            _concept_id(member, index)
            for index, member in zip(group["indexes"], group["members"])
        ]
        if len(concept_ids) == 1:
            unit_id = concept_ids[0]
        else:
            digest = sha1("|".join(concept_ids).encode("utf-8")).hexdigest()[:10]
            unit_id = f"unit_{normalized_chapter[:120]}_{digest}"
        units.append(
            {
                "id": unit_id,
                "label": _group_label(group),
                "concept_ids": concept_ids,
                "concepts": list(group["members"]),
                "source_pages": sorted(group["pages"]),
            }
        )
    return units


def covered_concept_ids(units: Iterable[Dict[str, Any]]) -> List[str]:
    """Convenience helper used by tests and ingestion validation."""
    return [
        concept_id
        for unit in units
        for concept_id in unit.get("concept_ids", [])
    ]
