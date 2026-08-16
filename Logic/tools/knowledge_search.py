# Logic/tools/knowledge_search.py

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from Logic.knowledge_graph import knowledge_graph  # <-- NEW
from Logic.content_pipeline import search_approved_content

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
    section_id = SECTION_ALIASES.get(section_id, section_id)
    strict_published_scope = str((scope or {}).get("catalog_source") or "").strip().lower() == "published"

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
