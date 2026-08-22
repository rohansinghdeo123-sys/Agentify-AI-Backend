# Logic/tools/knowledge_search.py

import os
import re
import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Logic.knowledge_graph import knowledge_graph  # <-- NEW
from Logic.content_pipeline import extract_pdf_pages, search_approved_content
from Logic.planning.curriculum_registry import resolve_planning_curriculum

logger = logging.getLogger("ai_educator.tools.knowledge_search")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SECTION_FILE_MAP = {
    "alkanes": os.path.join(BASE_DIR, "data", "chemistry", "hydrocarbon", "part1_alkanes.md"),
    "alkenes": os.path.join(BASE_DIR, "data", "chemistry", "hydrocarbon", "part2_alkenes.md"),
    "alkynes": os.path.join(BASE_DIR, "data", "chemistry", "hydrocarbon", "part3_alkynes.md"),
    "aromatics": os.path.join(BASE_DIR, "data", "chemistry", "hydrocarbon", "part4_aromatics.md"),
}

BASICS_PATH = os.path.join(BASE_DIR, "data", "datachemistry_basics.txt")
DEFAULT_GRAPH_PATH = os.path.join(BASE_DIR, "data", "Chapters", "basic_concepts_of_chemistry.json")
DEFAULT_GRAPH_CHAPTER = "basic-concepts-of-chemistry"

STOPWORDS = {
    "what", "is", "the", "of", "define", "explain", "write", "give",
    "state", "why", "how", "are", "in", "for", "with", "and", "from",
    "a", "an", "this", "that", "which", "do", "does", "can", "about",
    "tell", "me", "describe", "discuss", "mention", "list", "name",
}

CHEMISTRY_SYNONYMS = {
    "formula": ["general formula", "molecular formula", "chemical formula", "CₙH"],
    "alkane": ["alkanes", "paraffin", "paraffins", "saturated hydrocarbon", "CₙH₂ₙ₊₂"],
    "alkene": ["alkenes", "olefin", "olefins", "unsaturated hydrocarbon", "CₙH₂ₙ"],
    "alkyne": ["alkynes", "acetylene", "acetylenes", "CₙH₂ₙ₋₂"],
    "aromatic": ["aromatics", "arene", "arenes", "benzene"],
    "isomer": ["isomerism", "isomers", "structural isomer", "chain isomer"],
    "reaction": ["reactions", "reacts", "reactivity", "chemical reaction"],
    "property": ["properties", "physical properties", "chemical properties"],
    "preparation": ["prepare", "prepared", "synthesis", "method of preparation"],
    "nomenclature": ["naming", "IUPAC", "IUPAC nomenclature", "IUPAC name"],
    "boiling point": ["boiling points", "b.p.", "bp"],
    "melting point": ["melting points", "m.p.", "mp"],
}

SECTION_ALIASES = {
    "basic_concepts_of_chemistry": "matter_definition",
    "basic_concept_of_chemistry": "matter_definition",
    "matter": "matter_definition",
    "hydrocarbon": "alkanes",
    "hydrocarbons": "alkanes",
    "aromatic_hydrocarbons": "aromatics",
}


def ensure_default_knowledge_graph_loaded() -> None:
    """Load bundled graph content for entry points that do not import main.py."""
    if knowledge_graph.concepts or not os.path.exists(DEFAULT_GRAPH_PATH):
        return

    try:
        knowledge_graph.load_chapter(DEFAULT_GRAPH_PATH, DEFAULT_GRAPH_CHAPTER)
        logger.info("Loaded bundled knowledge graph: %s", DEFAULT_GRAPH_CHAPTER)
    except Exception as exc:
        logger.warning("Could not load bundled knowledge graph: %s", exc)


def _normalize(text: str) -> str:
    """Normalize Unicode subscripts/superscripts for matching."""
    replacements = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "⁺": "+", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.lower()


def _expand_query(question: str) -> List[str]:
    """Expand query keywords with chemistry-specific synonyms."""
    norm_q = _normalize(question)
    raw_keywords = [
        w for w in re.findall(r"\b\w+\b", norm_q)
        if w not in STOPWORDS and len(w) > 2
    ]

    expanded = set(raw_keywords)
    for kw in raw_keywords:
        for base, synonyms in CHEMISTRY_SYNONYMS.items():
            if kw in _normalize(base) or any(kw in _normalize(s) for s in synonyms):
                expanded.add(_normalize(base))
                for s in synonyms:
                    expanded.add(_normalize(s))

    return list(expanded)


def _score_paragraph(paragraph: str, keywords: List[str]) -> float:
    """Score a paragraph based on keyword density and position."""
    norm_para = _normalize(paragraph)
    para_len = max(len(norm_para.split()), 1)

    # Keyword frequency score
    freq_score = sum(norm_para.count(kw) for kw in keywords)

    # Keyword density (normalize by paragraph length)
    density_score = freq_score / para_len

    # Bonus for paragraphs that contain headings or definitions
    heading_bonus = 0.5 if any(marker in paragraph for marker in ["##", "**", "Definition", "General Formula"]) else 0

    # Bonus for paragraphs with chemical formulas
    formula_bonus = 0.3 if any(c in paragraph for c in ["₂", "₃", "₄", "ₙ", "⁺", "⁻"]) else 0

    return freq_score + (density_score * 10) + heading_bonus + formula_bonus


def _build_concept_context(concept: dict) -> str:
    """Build a text block from a single knowledge graph concept."""
    lines = []
    title = concept.get("title", "")
    if title:
        lines.append(f"# {title}")
    definition = concept.get("definition")
    if definition:
        lines.append(f"\n**Definition:** {definition}")
    core = concept.get("core_explanation")
    if core:
        lines.append(f"\n**Explanation:** {core}")
    key_points = concept.get("key_points", [])
    if key_points:
        lines.append("\n**Key Points:**")
        for point in key_points:
            lines.append(f"- {point}")
    formulas = concept.get("formulas", [])
    if formulas:
        lines.append("\n**Formulas:**")
        for formula in formulas:
            lines.append(f"- {formula}")
    examples = concept.get("examples", [])
    if examples:
        lines.append("\n**Examples:**")
        for example in examples:
            lines.append(f"- {example}")
    common_mistakes = concept.get("common_mistakes", [])
    if common_mistakes:
        lines.append("\n**Common Mistakes:**")
        for mistake in common_mistakes:
            lines.append(f"- {mistake.get('mistake','')} → Correction: {mistake.get('correction','')}")
    return "\n".join(lines)


def _find_exact_graph_concept(section_id: str):
    concept = knowledge_graph.get_concept(section_id)
    if concept:
        return concept

    for candidate in knowledge_graph.concepts.values():
        title_id = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(candidate.get("title") or "").strip().lower(),
        ).strip("_")
        if title_id == section_id:
            return candidate

    return None


@lru_cache(maxsize=4)
def _cached_planning_pdf_pages(
    source_path: str,
    source_sha256: str,
) -> Tuple[Tuple[int, str], ...]:
    """Extract one verified curriculum PDF once per process.

    ``source_sha256`` is part of the cache key so an approved source update can
    never reuse text extracted from an older document.
    """
    del source_sha256  # cache identity; digest validation happens in the manifest loader
    return tuple(
        (int(page["page_number"]), str(page.get("text") or ""))
        for page in extract_pdf_pages(Path(source_path))
    )


def _planning_unit_identities(unit: Dict[str, Any]) -> set[str]:
    values: List[Any] = [
        unit.get("id"),
        unit.get("title"),
        unit.get("primary_topic_id"),
        *(unit.get("legacy_topic_ids") or []),
    ]
    for concept in unit.get("concepts") or []:
        values.extend((concept.get("id"), concept.get("title")))
    return {
        normalized
        for value in values
        if (normalized := re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_"))
    }


def _planning_pdf_chunks(text: str, *, target_chars: int = 900) -> List[str]:
    """Create bounded verbatim word chunks without crossing a source page."""
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0
    for word in words:
        next_length = current_length + len(word) + (1 if current else 0)
        if current and next_length > target_chars:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk]


def _anchor_index(text: str, anchor: str) -> Tuple[int, int]:
    """Locate one authored anchor despite PDF spacing and case artifacts."""
    normalized_text: List[str] = []
    source_positions: List[int] = []
    for source_index, character in enumerate(text):
        for folded_character in unicodedata.normalize("NFKD", character):
            if folded_character.isalnum():
                normalized_text.append(folded_character.lower())
                source_positions.append(source_index)
    normalized_anchor = "".join(
        character.lower()
        for character in unicodedata.normalize("NFKD", anchor)
        if character.isalnum()
    )
    if not normalized_anchor:
        raise ValueError("source anchor is empty after normalization")
    haystack = "".join(normalized_text)
    first = haystack.find(normalized_anchor)
    if first < 0 or haystack.find(normalized_anchor, first + 1) >= 0:
        raise ValueError("source anchor is missing or ambiguous")
    last = first + len(normalized_anchor) - 1
    return source_positions[first], source_positions[last] + 1


def _planning_source_segment(
    page_text: str,
    segment: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Apply one manifest-authored, fail-closed source boundary."""
    start = 0
    end = len(page_text)
    start_anchor = str(segment.get("start_anchor") or "").strip()
    end_anchor = str(segment.get("end_anchor") or "").strip()
    if start_anchor:
        start, _ = _anchor_index(page_text, start_anchor)
    if end_anchor:
        end, _ = _anchor_index(page_text, end_anchor)
    if end <= start:
        raise ValueError("source segment boundaries are reversed or empty")
    body = page_text[start:end].strip()
    if len(body) < 40:
        raise ValueError("source segment is too small to ground an answer")
    return body, {
        "start_anchor": start_anchor or None,
        "end_anchor": end_anchor or None,
        "start_character": start,
        "end_character": end,
    }


def _planning_pdf_error(section_id: str, error: str) -> Dict[str, Any]:
    return {
        "context": "",
        "section_id": section_id,
        "paragraphs_found": 0,
        "keywords_used": [],
        "basics_context": "",
        "source": "ncert_planning_pdf",
        "catalog_source": "planning_manifest",
        "error": error,
    }


def _search_registered_planning_pdf(
    *,
    section_id: str,
    question: str,
    max_paragraphs: int,
    max_chars: int,
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    """Retrieve only the registered unit's exact NCERT source pages."""
    chapter_ref = str(
        scope.get("chapter_slug")
        or scope.get("chapter")
        or ""
    ).strip()
    curriculum = resolve_planning_curriculum(
        chapter_ref=chapter_ref,
        subject=str(scope.get("subject") or "") or None,
        class_level=str(scope.get("class_level") or "") or None,
    )
    if not curriculum:
        return _planning_pdf_error(section_id, "planning_curriculum_scope_invalid")

    requested_values = {
        re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
        for value in (
            section_id,
            scope.get("section_id"),
            scope.get("topic"),
            *(scope.get("concept_ids") or []),
        )
        if str(value or "").strip()
    }
    matching_units = [
        unit
        for unit in curriculum["units"]
        if requested_values.intersection(_planning_unit_identities(unit))
    ]
    if len(matching_units) != 1:
        return _planning_pdf_error(section_id, "planning_learning_unit_not_found")
    unit = matching_units[0]

    page_sections: Dict[int, List[str]] = {}
    for section in unit["ncert_sections"]:
        pages = section.get("pages") or {}
        for page_number in range(int(pages["start"]), int(pages["end"]) + 1):
            page_sections.setdefault(page_number, []).append(str(section["id"]))
    source_segments = list(unit.get("source_segments") or [])
    if not source_segments:
        return _planning_pdf_error(section_id, "planning_source_pages_missing")

    source_reference = curriculum["source"]
    source_path = (Path(BASE_DIR) / str(source_reference["path"])).resolve()
    try:
        extracted_pages = _cached_planning_pdf_pages(
            str(source_path),
            str(source_reference.get("sha256") or ""),
        )
    except Exception as exc:  # noqa: BLE001 - strict retrieval returns a stable failure
        logger.error("Registered Planning PDF extraction failed: %s", exc)
        return _planning_pdf_error(section_id, "planning_source_unavailable")

    query_text = question or str(unit["title"])
    keywords = _expand_query(query_text)
    extracted_by_page = dict(extracted_pages)
    candidates: List[Tuple[float, int, int, int, str, Dict[str, Any]]] = []
    try:
        for segment_index, segment in enumerate(source_segments):
            page_number = int(segment["page"])
            page_text = extracted_by_page.get(page_number, "")
            source_slice, boundary = _planning_source_segment(page_text, segment)
            for chunk_index, chunk in enumerate(_planning_pdf_chunks(source_slice)):
                candidates.append(
                    (
                        _score_paragraph(chunk, keywords),
                        page_number,
                        segment_index,
                        chunk_index,
                        chunk,
                        boundary,
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Registered Planning source slice failed closed: %s", exc)
        return _planning_pdf_error(section_id, "planning_source_slice_invalid")
    if not candidates:
        return _planning_pdf_error(section_id, "planning_source_text_unavailable")

    ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
    selected: List[Dict[str, Any]] = []
    total_chars = 0
    for _score, page_number, segment_index, chunk_index, chunk, boundary in ranked:
        if len(selected) >= max(1, max_paragraphs):
            break
        section_ids = page_sections[page_number]
        prefix = f"[NCERT PDF page {page_number}; sections {', '.join(section_ids)}] "
        remaining = max_chars - total_chars - len(prefix) - (2 if selected else 0)
        if remaining < 80:
            continue
        body = chunk if len(chunk) <= remaining else chunk[:remaining].rsplit(" ", 1)[0]
        if not body:
            continue
        selected.append(
            {
                "page": page_number,
                "segment_index": segment_index,
                "chunk_index": chunk_index,
                "section_ids": section_ids,
                "boundary": boundary,
                "text": f"{prefix}{body}",
            }
        )
        total_chars += len(selected[-1]["text"]) + (2 if len(selected) > 1 else 0)
    if not selected:
        return _planning_pdf_error(section_id, "planning_source_budget_too_small")

    return {
        "context": "\n\n".join(item["text"] for item in selected),
        "section_id": unit["primary_topic_id"],
        "paragraphs_found": len(selected),
        "keywords_used": keywords,
        "basics_context": "",
        "source": "ncert_planning_pdf",
        "catalog_source": "planning_manifest",
        "source_authority": source_reference["authority"],
        "source_document": "NCERT Class XI Chemistry Part I",
        "source_path": source_reference["path"],
        "source_sha256": source_reference.get("sha256", ""),
        "source_edition": curriculum["edition"],
        "source_pages": sorted({int(item["page"]) for item in selected}),
        "matched_sections": list(
            dict.fromkeys(
                section_id
                for item in selected
                for section_id in item["section_ids"]
            )
        ),
        "provenance": [
            {
                "page": int(item["page"]),
                "section_ids": list(item["section_ids"]),
                "segment_index": int(item["segment_index"]),
                "source_slice": dict(item["boundary"]),
            }
            for item in selected
        ],
        "retrieval_mode": "exact_section_slice_lexical",
        "unit_id": unit["id"],
        "content_order_locked": True,
    }


def search_knowledge_base(
    section_id: str,
    question: str,
    max_paragraphs: int = 5,
    max_chars: int = 3000,
    scope: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    TOOL: Search the knowledge base for relevant content.

    If the section_id matches a markdown file, those paragraphs are used.
    Otherwise, the Knowledge Graph (JSON concepts) is queried and the
    concept data is returned as the context.

    Returns:
        dict with keys:
        - "context": str — The retrieved text
        - "section_id": str — Which section was searched
        - "paragraphs_found": int — How many relevant paragraphs (or 1 for a graph concept)
        - "keywords_used": list — What keywords were searched
        - "basics_context": str — Supplementary basics text
    """
    section_id = re.sub(r"[^a-z0-9]+", "_", (section_id or "").strip().lower()).strip("_")
    catalog_source = str((scope or {}).get("catalog_source") or "").strip().lower()
    if catalog_source == "planning_manifest":
        return _search_registered_planning_pdf(
            section_id=section_id,
            question=question,
            max_paragraphs=max_paragraphs,
            max_chars=max_chars,
            scope=dict(scope or {}),
        )
    section_id = SECTION_ALIASES.get(section_id, section_id)
    strict_published_scope = catalog_source == "published"

    try:
        approved_result = search_approved_content(
            section_id=section_id,
            question=question,
            scope=scope,
            max_chars=max_chars,
            limit=max_paragraphs,
        )
        if str(approved_result.get("context") or "").strip():
            return approved_result
        if strict_published_scope:
            return {
                **approved_result,
                "section_id": section_id,
                "source": approved_result.get("source") or "approved_content_pipeline",
                "error": "material_not_found",
            }
    except Exception as exc:
        logger.warning("Approved content pipeline search failed: %s", exc)
        if strict_published_scope:
            # A published catalog selection must fail closed. Falling through
            # to bundled markdown or the global graph can cross the selected
            # chapter, class, or content version and produce an ungrounded
            # revision answer.
            return {
                "context": "",
                "section_id": section_id,
                "paragraphs_found": 0,
                "keywords_used": [],
                "basics_context": "",
                "source": "approved_content_pipeline",
                "error": "approved_content_unavailable",
            }

    # ── Step 1: Try markdown file map ──────────────────────────────────
    if section_id in SECTION_FILE_MAP:
        try:
            with open(SECTION_FILE_MAP[section_id], "r", encoding="utf-8") as f:
                section_text = f.read()
        except FileNotFoundError:
            # Fall through to graph fallback
            section_text = ""

        if section_text:
            # Load basics
            basics_text = ""
            try:
                with open(BASICS_PATH, "r", encoding="utf-8") as f:
                    basics_text = f.read()[:800]
            except FileNotFoundError:
                pass

            # If section is small enough, return all of it
            if len(section_text) <= max_chars:
                return {
                    "context": section_text,
                    "section_id": section_id,
                    "paragraphs_found": len(section_text.split("\n\n")),
                    "keywords_used": [],
                    "basics_context": basics_text,
                    "source": "markdown",
                }

            # Expand query with synonyms
            keywords = _expand_query(question)

            # Split into paragraphs and score
            paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
            scored: List[Tuple[float, str]] = []

            for para in paragraphs:
                score = _score_paragraph(para, keywords)
                scored.append((score, para))

            scored.sort(key=lambda x: x[0], reverse=True)

            selected = []
            total_len = 0

            for score, para in scored[:max_paragraphs * 2]:
                if score <= 0:
                    continue
                if total_len + len(para) > max_chars:
                    continue
                selected.append(para)
                total_len += len(para)

            if not selected:
                fallback = paragraphs[:3]
                return {
                    "context": "\n\n".join(fallback),
                    "section_id": section_id,
                    "paragraphs_found": len(fallback),
                    "keywords_used": keywords,
                    "basics_context": basics_text,
                    "source": "markdown",
                }

            return {
                "context": "\n\n".join(selected),
                "section_id": section_id,
                "paragraphs_found": len(selected),
                "keywords_used": keywords,
                "basics_context": basics_text,
                "source": "markdown",
            }

    # ── Step 2: Knowledge Graph fallback ──────────────────────────────
    ensure_default_knowledge_graph_loaded()
    if knowledge_graph.concepts:
        concept = _find_exact_graph_concept(section_id)

        if concept:
            context = _build_concept_context(concept)
            return {
                "context": context,
                "section_id": section_id,
                "paragraphs_found": 1,
                "keywords_used": [],
                "basics_context": "",
                "source": "knowledge_graph",
            }

    # ── No match at all ──────────────────────────────────────────────
    return {
        "context": "",
        "section_id": section_id,
        "paragraphs_found": 0,
        "keywords_used": [],
        "basics_context": "",
        "error": f"Section '{section_id}' not found in any knowledge source.",
    }
