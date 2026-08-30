"""Production NCERT content ingestion and retrieval pipeline.

PDFs are the source of truth. Concept JSON and chunks are derived layers that
must pass validation and approval before Study Lab retrieval uses them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sqlalchemy.orm import Session

from database import SessionLocal
from Logic import embeddings as embeddings_service
from models import (
    ContentChapter,
    ContentChunk,
    ContentConcept,
    ContentIngestionJob,
    ContentPage,
)
from services.topic_grouping import build_learning_units

try:
    from pypdf import PdfReader
    from pypdf._codecs.symbol import _symbol_encoding as _PYPDF_SYMBOL_ENCODING
    from pypdf.generic import DecodedStreamObject, NameObject
except Exception:  # pragma: no cover - exercised in environments without pypdf
    PdfReader = None  # type: ignore[assignment]
    _PYPDF_SYMBOL_ENCODING = []  # type: ignore[assignment]
    DecodedStreamObject = None  # type: ignore[assignment]
    NameObject = None  # type: ignore[assignment]

try:
    import pdfplumber
except Exception:  # pragma: no cover - publication gate catches unresolved aliases
    pdfplumber = None  # type: ignore[assignment]


logger = logging.getLogger("ai_educator.content_pipeline")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_NCERT_DIR = DATA_DIR / "raw" / "ncert"
APPROVED_STATUSES = {"approved", "published"}
DEFAULT_VERSION = "v1"
CONTENT_GENERATION_PROMPT_VERSION = "ncert-learning-units-v2"
CONTENT_GENERATION_TEMPERATURE = 0.1
CONTENT_GENERATION_MAX_TOKENS = 4096
# Groq's current low-tier reviewer route rejects a single request when prompt
# tokens plus the requested completion exceed 8,000 tokens.  Leave a material
# safety margin because the provider's model tokenizer is more expensive for
# equations and dense numeric tables than the application's generic estimator.
CONTENT_GENERATION_REQUEST_TOKEN_BUDGET = 7200
CONTENT_GENERATION_MIN_OUTPUT_TOKENS = 1800
CONTENT_GENERATION_PREFERRED_OUTPUT_TOKENS = 2600
# Groq's configured reviewer route has an 8k tokens-per-minute ceiling. Keeping
# provider calls one minute apart prevents a valid 7.2k-token request followed
# by another valid request from failing on the cumulative TPM window.
CONTENT_GENERATION_GROQ_BATCH_DELAY_SECONDS = 61.0
CONTENT_GENERATION_SYSTEM_PROMPT = (
    "You convert NCERT textbook pages into strict structured concept JSON. "
    "Use ONLY the supplied page text. Do not add outside facts. "
    "Create a small set of meaningful learning units, not one item per heading or subheading. "
    "Merge a definition with its explanation, properties, formulas, examples, applications, "
    "and special cases whenever they teach the same underlying idea. Do not create standalone "
    "items for tiny definitions, individual examples, single formulas, practice prompts, or summaries. "
    "Preserve every essential syllabus concept by placing it inside the most relevant broader unit. "
    "Simple page batches should usually need 1-2 units; dense batches may need 3 or occasionally 4. "
    "Choose subject- and chapter-specific titles that describe the actual material; never force a "
    "generic template or fixed topic names. "
    "Be concise enough to finish the JSON: keep each definition under 60 words, each core_explanation "
    "under 180 words, key_points to 3-6, examples to at most 3, and other list fields to at most 4 items. "
    "Include formulas only when they are explicitly visible in the supplied text; never reconstruct a "
    "missing equation or add an outside fact. Avoid repeating the same fact across fields. "
    "Return ONLY a JSON array. Each item must include: concept_id, title, "
    "definition, core_explanation, key_points, examples, formulas, properties, "
    "applications, common_mistakes, prerequisites, related_concepts, "
    "learning_objectives, source_pages, difficulty_level, "
    "blooms_taxonomy, typical_exam_weightage, importance_level. "
    "difficulty_level must be an integer from 1 (easiest) to 5 (hardest). "
    "source_pages must be a JSON array of integer page numbers (e.g. [4, 5]), "
    "not page markers. typical_exam_weightage and importance_level must be short strings. "
    "Every concept must cite source_pages from the supplied [PAGE n] markers."
)

# A secondary corruption signal for PDFs outside the structurally repairable
# legacy Bookman family. It never alters text; it only blocks publication when
# the extracted prose itself still looks cipher-like.
_LEGACY_FONT_COMMON_WORDS = {
    "a", "able", "about", "after", "also", "an", "and", "are", "as", "at",
    "be", "been", "between", "by", "can", "carbon", "chemical", "chemistry",
    "compound", "compounds", "for", "from", "has", "have", "in", "into", "is",
    "it", "learn", "may", "of", "on", "or", "organic", "other", "reaction",
    "reactions", "structure", "structures", "that", "the", "their", "these",
    "this", "to", "understand", "unit", "was", "which", "will", "with", "write",
    "you",
}


def _min_coverage_score() -> float:
    # Minimum share of teaching pages a chapter's concepts must cover before it
    # can be approved. Dense chapters carry many exercise/figure pages with no
    # extractable concept, so the default is forgiving; the human review is the
    # real quality gate. Tunable via env without a code change.
    try:
        return float(os.getenv("CONTENT_MIN_COVERAGE_SCORE", "0.60"))
    except ValueError:
        return 0.60

STOPWORDS = {
    "what", "why", "how", "explain", "define", "describe", "tell", "give",
    "with", "from", "about", "than", "more", "less", "into", "this", "that",
    "these", "those", "your", "please", "simple", "simply", "the", "and",
    "are", "was", "were", "for", "does", "can", "chapter", "class", "subject",
    "only", "notes", "study", "material",
}


class ContentConceptPayload(BaseModel):
    concept_id: str = Field(min_length=2, max_length=140)
    title: str = Field(min_length=2, max_length=220)
    definition: str = ""
    core_explanation: str = ""
    key_points: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    formulas: List[Any] = Field(default_factory=list)
    properties: List[str] = Field(default_factory=list)
    applications: List[str] = Field(default_factory=list)
    common_mistakes: List[Dict[str, Any]] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    related_concepts: List[Any] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)
    difficulty_level: int = Field(default=1, ge=1, le=5)
    blooms_taxonomy: str = ""
    typical_exam_weightage: str = ""
    importance_level: str = ""
    # When several old micro-concepts are consolidated, their stable IDs stay
    # here as aliases.  This is persisted inside ContentConcept.raw_json (no DB
    # schema change) so legacy bookmarks and saved plans remain resolvable.
    source_concept_ids: List[str] = Field(default_factory=list)

    @field_validator("concept_id")
    @classmethod
    def normalize_concept_id(cls, value: str) -> str:
        normalized = normalize_key(value)
        if not normalized:
            raise ValueError("concept_id cannot be empty after normalization")
        return normalized[:140]

    @field_validator("source_pages")
    @classmethod
    def normalize_source_pages(cls, value: List[int]) -> List[int]:
        pages = sorted({int(page) for page in value if int(page) > 0})
        return pages

    @field_validator("difficulty_level", mode="before")
    @classmethod
    def coerce_difficulty_level(cls, value: Any) -> int:
        # LLMs routinely return a word ("Medium") or a numeric string instead of
        # the 1-5 integer the schema expects. Map those rather than reject the
        # whole concept, which would otherwise drop every concept as a blocking
        # issue and leave coverage at zero.
        if value is None or isinstance(value, bool) or value == "":
            return 1
        if isinstance(value, (int, float)):
            level = int(value)
        else:
            text = str(value).strip().lower()
            words = {
                "very easy": 1, "trivial": 1,
                "easy": 2, "low": 2, "basic": 2, "beginner": 2, "simple": 2,
                "medium": 3, "moderate": 3, "intermediate": 3, "average": 3, "normal": 3,
                "hard": 4, "high": 4, "difficult": 4, "challenging": 4, "advanced": 4,
                "very hard": 5, "very difficult": 5, "expert": 5,
            }
            if text in words:
                level = words[text]
            else:
                match = re.search(r"\d+", text)
                level = int(match.group()) if match else 1
        return max(1, min(5, level))

    @model_validator(mode="before")
    @classmethod
    def coerce_llm_field_types(cls, data: Any) -> Any:
        # Groq returns loosely-typed JSON: numbers where strings are expected
        # (importance_level: 7), page markers like "[PAGE 28]" instead of the
        # integer 28, and occasionally a bare value where a list is expected.
        # A single mismatch makes Pydantic reject the whole concept, which would
        # drop every concept and leave coverage at zero. Coerce here so one
        # sloppy field never discards otherwise-good teaching content.
        if not isinstance(data, dict):
            return data
        data = dict(data)

        for key in ("definition", "core_explanation", "blooms_taxonomy",
                    "typical_exam_weightage", "importance_level"):
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                data[key] = str(value)

        # concept_id and title are required and length-constrained (>= 2 chars).
        # The model sometimes emits a number or a single character (concept_id:
        # 4), which fails min_length and drops the whole concept. Derive valid
        # values from the content so that can never happen.
        concept_id = normalize_key(data.get("concept_id"))
        title = str(data.get("title") or "").strip()
        if len(concept_id) < 2:
            concept_id = normalize_key(title)
        if len(concept_id) < 2:
            concept_id = normalize_key(str(data.get("definition") or "")[:60])
        data["concept_id"] = (concept_id or "concept")[:140]
        if len(title) < 2:
            title = titleize(concept_id) or "Untitled concept"
        data["title"] = title[:220]

        for key in ("key_points", "examples", "properties", "applications",
                    "prerequisites", "learning_objectives", "source_concept_ids"):
            if data.get(key) is None:
                continue
            items = data[key] if isinstance(data[key], list) else [data[key]]
            data[key] = [item if isinstance(item, str) else str(item)
                         for item in items if item is not None]

        for key in ("formulas", "related_concepts"):
            if data.get(key) is not None and not isinstance(data[key], list):
                data[key] = [data[key]]

        if data.get("common_mistakes") is not None:
            items = data["common_mistakes"]
            if not isinstance(items, list):
                items = [items]
            data["common_mistakes"] = [
                item if isinstance(item, dict) else {"note": str(item)}
                for item in items if item is not None
            ]

        if data.get("source_pages") is not None:
            data["source_pages"] = coerce_page_numbers(data["source_pages"])

        return data


def coerce_page_numbers(value: Any) -> List[int]:
    """Pull integer page numbers out of whatever the model returned: ints,
    numeric strings, or page markers like "[PAGE 28]". Used by both schema
    coercion and the generation batch-page fallback."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    pages: List[int] = []
    for entry in items:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, (int, float)):
            pages.append(int(entry))
        else:
            match = re.search(r"\d+", str(entry))
            if match:
                pages.append(int(match.group()))
    return pages


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def titleize(value: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", value or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.title()


def content_terms(value: str) -> List[str]:
    terms = []
    for term in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if len(term) > 2 and term not in STOPWORDS:
            terms.append(term)
    return sorted(set(terms))


def safe_data_path(path_value: Optional[str], *, default: Path = RAW_NCERT_DIR) -> Path:
    candidate = Path(path_value or default)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    resolved = candidate.resolve()
    data_root = DATA_DIR.resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"Path must stay inside backend/data: {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_metadata_from_pdf_path(pdf_path: Path, root_path: Optional[Path] = None) -> Dict[str, Any]:
    root = root_path.resolve() if root_path else RAW_NCERT_DIR.resolve()
    resolved = pdf_path.resolve()
    try:
        parts = list(resolved.relative_to(root).parts)
    except ValueError:
        parts = list(resolved.parts)

    filename = resolved.stem
    normalized_parts = [normalize_key(part) for part in parts]
    board = "NCERT" if "ncert" in normalized_parts or "raw" in normalized_parts else "NCERT"

    class_level = ""
    for part in normalized_parts:
        match = re.search(r"class_?(\d{1,2})", part)
        if match:
            class_level = match.group(1)
            break

    subject = ""
    if len(parts) >= 2:
        parent = normalize_key(parts[-2])
        if not parent.startswith("class"):
            subject = titleize(parent)

    chapter_number = None
    number_match = re.search(r"(?:chapter|ch|chap)[_\-\s]*(\d{1,3})", filename, re.IGNORECASE)
    if not number_match:
        number_match = re.search(r"\b(\d{1,3})\b", filename)
    if number_match:
        chapter_number = int(number_match.group(1))

    chapter_name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    chapter_name = re.sub(r"(?:chapter|chap|ch)[_\-\s]*\d{1,3}", "", chapter_name, flags=re.IGNORECASE)
    chapter_name = re.sub(r"^\d{1,3}[_\-\s]*", "", chapter_name).strip(" _-")
    chapter_name = titleize(chapter_name or f"Chapter {chapter_number or ''}".strip())

    book_name = ""
    if len(parts) >= 3:
        maybe_book = normalize_key(parts[-3])
        if maybe_book and not maybe_book.startswith("class"):
            book_name = titleize(maybe_book)

    slug_parts = [board, f"class_{class_level}" if class_level else "", subject, f"chapter_{chapter_number or ''}", chapter_name]
    slug = normalize_key("_".join(part for part in slug_parts if part))

    return {
        "board": board,
        "class_level": class_level,
        "subject": subject,
        "book_name": book_name,
        "chapter_number": chapter_number,
        "chapter_name": chapter_name,
        "slug": slug,
    }


_LEGACY_BOOKMAN_FAMILIES = {
    "Bookman-Light",
    "Bookman-Demi",
    "Bookman-LightItalic",
    "Bookman-DemiItalic",
}
_LEGACY_BOOKMAN_EXTRAS = {
    101: 0x00C9,  # É
    112: 0x00E9,  # é
    129: 0x00FC,  # ü
    171: 0x2026,  # …
    178: 0x2014,  # —
    179: 0x201C,  # “
    180: 0x201D,  # ”
    181: 0x2018,  # ‘
    182: 0x2019,  # ’
    259: 0x2013,  # –
    262: 0x2022,  # •
}

# pdfplumber/PDFMiner deliberately preserves Adobe Symbol glyphs without a
# direct Unicode equivalent in the private-use area. NCERT also carries two
# custom mathematical glyphs outside the standard F0xx Symbol range. Convert
# these deterministically before any page reaches retrieval or the LLM.
_PDF_PRIVATE_GLYPH_MAP = {
    0xF103: "α",
    0xF106: "σ",
    0xF8E5: "",
    0xF8E6: "|",  # vertical extender (also used for organic branch bonds)
    0xF8E7: "-",  # horizontal extender
    0xF8E8: "{",
    0xF8E9: "",
    0xF8EA: "",
    0xF8EB: "(",
    0xF8EC: "",
    0xF8ED: "",
    0xF8EE: "[",
    0xF8EF: "",
    0xF8F0: "",
    0xF8F1: "{",
    0xF8F2: "",
    0xF8F3: "",
    0xF8F4: "",
    0xF8F5: "",
    0xF8F6: ")",
    0xF8F7: "",
    0xF8F8: "",
    0xF8F9: "]",
    0xF8FA: "",
    0xF8FB: "",
    0xF8FC: "}",
    0xF8FD: "",
    0xF8FE: "",
}


def _pdf_object(value: Any) -> Any:
    try:
        return value.get_object()
    except (AttributeError, TypeError):
        return value


def _legacy_bookman_cmap():
    """Build the canonical Unicode map used by NCERT's full Bookman fonts."""

    if DecodedStreamObject is None:
        raise RuntimeError("pypdf generic stream support is unavailable.")
    pairs = [*_LEGACY_BOOKMAN_EXTRAS.items(), *((cid, 0x0020) for cid in (239, 257, 264))]
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /AgentifyLegacyBookman-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
        "1 beginbfrange",
        "<0003> <0061> <0020>",
        "endbfrange",
        f"{len(pairs)} beginbfchar",
        *(f"<{cid:04X}> <{codepoint:04X}>" for cid, codepoint in pairs),
        "endbfchar",
        "endcmap",
        "CMapName currentdict /CMap defineresource pop",
        "end",
        "end",
    ]
    stream = DecodedStreamObject()
    stream.set_data(("\n".join(lines) + "\n").encode("ascii"))
    return stream


def _is_full_legacy_bookman_font(font: Any) -> bool:
    """Identify only the full embedded NCERT Type0/Identity-H Bookman fonts.

    Sparse four-kilobyte subsets in earlier chapters and WinAnsi/Type1 fonts
    intentionally fail this structural gate.  The full legacy fonts use a
    stable CID/GID layout; rebuilding their incomplete ToUnicode CMap is
    deterministic and does not rely on guessing whether prose looks English.
    """

    font = _pdf_object(font)
    if not hasattr(font, "get"):
        return False
    base_font = str(font.get("/BaseFont") or "").lstrip("/").split("+")[-1]
    if (
        base_font not in _LEGACY_BOOKMAN_FAMILIES
        or str(font.get("/Subtype") or "") != "/Type0"
        or str(font.get("/Encoding") or "") != "/Identity-H"
    ):
        return False
    descendants = _pdf_object(font.get("/DescendantFonts") or [])
    if not descendants:
        return False
    descendant = _pdf_object(descendants[0])
    if (
        not hasattr(descendant, "get")
        or str(descendant.get("/Subtype") or "") != "/CIDFontType2"
        or str(descendant.get("/CIDToGIDMap") or "") != "/Identity"
    ):
        return False
    descriptor = _pdf_object(descendant.get("/FontDescriptor") or {})
    font_file_ref = descriptor.get("/FontFile2") if hasattr(descriptor, "get") else None
    font_file = _pdf_object(font_file_ref) if font_file_ref is not None else None
    if font_file is None or not hasattr(font_file, "get_data"):
        return False
    try:
        return len(font_file.get_data()) >= 15_000
    except Exception:  # noqa: BLE001 - malformed font must remain untouched
        return False


def _cmap_unicode_mappings(font: Any) -> Dict[int, int]:
    font = _pdf_object(font)
    cmap_ref = font.get("/ToUnicode") if hasattr(font, "get") else None
    cmap = _pdf_object(cmap_ref) if cmap_ref is not None else None
    try:
        text = cmap.get_data().decode("latin1") if cmap is not None else ""
    except (AttributeError, OSError, UnicodeError):
        return {}
    mappings: Dict[int, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        range_match = re.fullmatch(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
            line,
        )
        if range_match:
            start, end, target = (int(value, 16) for value in range_match.groups())
            for offset, cid in enumerate(range(start, end + 1)):
                mappings[cid] = target + offset
            continue
        pair_match = re.fullmatch(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
            line,
        )
        if pair_match:
            cid, target = (int(value, 16) for value in pair_match.groups())
            mappings[cid] = target
    return mappings


def _has_defective_legacy_bookman_cmap(font: Any) -> bool:
    """Recognise the incomplete/misaligned NCERT maps without touching valid maps."""
    if not _is_full_legacy_bookman_font(font):
        return False
    mappings = _cmap_unicode_mappings(font)
    if not mappings:
        return False
    expected = {cid: 0x20 + cid - 3 for cid in range(3, 98)}
    comparable = {cid: target for cid, target in mappings.items() if cid in expected}
    return bool(comparable and any(target != expected[cid] for cid, target in comparable.items()))


def _repair_legacy_ncert_font_maps(reader: Any) -> List[str]:
    """Repair eligible font maps in memory; the source PDF is never modified."""

    if NameObject is None:
        return []
    repaired: set[str] = set()
    for page in reader.pages:
        resources = _pdf_object(page.get("/Resources") or {})
        fonts = _pdf_object(resources.get("/Font") or {}) if hasattr(resources, "get") else {}
        for _resource_name, font_ref in (fonts.items() if hasattr(fonts, "items") else []):
            font = _pdf_object(font_ref)
            if not _has_defective_legacy_bookman_cmap(font):
                continue
            font[NameObject("/ToUnicode")] = _legacy_bookman_cmap()
            repaired.add(str(font.get("/BaseFont") or "<unknown>"))
    return sorted(repaired)


def _legacy_font_text_metrics(value: str) -> Dict[str, float]:
    text = str(value or "")
    tokens = re.findall(r"[A-Za-z]{1,}", text)
    common_words = sum(token.lower() in _LEGACY_FONT_COMMON_WORDS for token in tokens)
    controls = sum(ord(character) < 32 and character not in "\t\n\r" for character in text)
    letters = [character for character in text if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )
    return {
        "tokens": float(len(tokens)),
        "common_words": float(common_words),
        "common_ratio": common_words / max(1, len(tokens)),
        "controls": float(controls),
        "backslashes": float(text.count("\\")),
        "uppercase_ratio": uppercase_ratio,
    }


def _normalize_pdf_formula_glyphs(value: str) -> Tuple[str, int]:
    """Decode NCERT Symbol-font aliases/private glyphs into portable Unicode.

    Some NCERT Symbol fonts reach pypdf as names such as ``/unif0ae`` rather
    than a character.  The suffix is the low byte of Adobe Symbol encoding
    (for example, ``ae`` is a right arrow).  Decode only aliases that resolve
    through that published encoding table; unknown aliases remain visible so
    the extraction quality gate can reject them instead of silently guessing.
    """

    source = str(value or "")
    changed = 0

    def replace_alias(match: re.Match[str]) -> str:
        nonlocal changed
        raw_code = int(match.group(1), 16)
        symbol_index: Optional[int] = None
        if raw_code <= 0xFF:
            symbol_index = raw_code
        elif 0xF000 <= raw_code <= 0xF0FF:
            symbol_index = raw_code - 0xF000
        if symbol_index is None or not _PYPDF_SYMBOL_ENCODING:
            return match.group(0)
        replacement = _PYPDF_SYMBOL_ENCODING[symbol_index]
        replacement = _PDF_PRIVATE_GLYPH_MAP.get(ord(replacement), replacement)
        if replacement and 0xE000 <= ord(replacement[0]) <= 0xF8FF:
            return match.group(0)
        changed += 1
        return replacement

    source = re.sub(
        r"/unif([0-9A-Fa-f]{3,4})",
        replace_alias,
        source,
        flags=re.IGNORECASE,
    )
    normalized: List[str] = []
    for character in source:
        codepoint = ord(character)
        replacement: Optional[str] = None
        if 0xF000 <= codepoint <= 0xF0FF and _PYPDF_SYMBOL_ENCODING:
            replacement = _PYPDF_SYMBOL_ENCODING[codepoint - 0xF000]
            replacement = _PDF_PRIVATE_GLYPH_MAP.get(ord(replacement), replacement)
        elif codepoint in _PDF_PRIVATE_GLYPH_MAP:
            replacement = _PDF_PRIVATE_GLYPH_MAP[codepoint]
        if replacement is None:
            normalized.append(character)
            continue
        normalized.append(replacement)
        changed += 1
    text = "".join(normalized)
    # Multi-piece mathematical delimiters and bond extenders otherwise leave
    # long runs after their top/middle/bottom glyphs are flattened.
    text = re.sub(r"\|{2,}", "|", text)
    text = re.sub(r"-{3,}", "-", text)
    text = re.sub(r"(?:-\s*)*→(?:\s*-)*", " → ", text)
    return text, changed


def _extracted_text_sanity(value: str) -> Dict[str, Any]:
    """Measure encoding sanity separately from text length.

    Length alone made cipher text look perfect.  This signal intentionally
    flags only strong prose corruption; a few control glyphs on equation-heavy
    pages remain reviewable instead of becoming false hard failures.
    """

    text = str(value or "")
    non_space = max(1, sum(not character.isspace() for character in text))
    controls = sum(ord(character) < 32 and character not in "\t\n\r" for character in text)
    slash_ratio = text.count("\\") / non_space
    control_ratio = controls / non_space
    metrics = _legacy_font_text_metrics(text)
    formula_alias_count = len(re.findall(r"/unif[0-9A-Fa-f]+", text))
    cid_placeholder_count = len(re.findall(r"\(cid:\d+\)", text, re.IGNORECASE))
    private_use_count = sum(0xE000 <= ord(character) <= 0xF8FF for character in text)
    cipher_like = bool(
        metrics["tokens"] >= 40
        and metrics["uppercase_ratio"] >= 0.82
        and metrics["common_ratio"] < 0.025
        and (controls >= 5 or text.count("\\") >= 5)
    )
    return {
        "control_ratio": round(control_ratio, 5),
        "backslash_ratio": round(slash_ratio, 5),
        "cipher_like": cipher_like,
        "formula_alias_count": formula_alias_count,
        "cid_placeholder_count": cid_placeholder_count,
        "private_use_count": private_use_count,
        "suspected_formula_glyph_corruption": bool(
            formula_alias_count or cid_placeholder_count or private_use_count
        ),
        # Symbol/formula fonts can legitimately carry controls and backslashes.
        # Keep that visible for operator review, but only cipher-like prose is a
        # blocking encoding failure.
        "needs_symbol_review": bool(
            control_ratio >= 0.03
            or slash_ratio >= 0.025
            or formula_alias_count
            or cid_placeholder_count
            or private_use_count
        ),
        "suspected_encoding_corruption": bool(
            cipher_like or formula_alias_count or cid_placeholder_count or private_use_count
        ),
    }


def _prefer_formula_fallback(primary: str, candidate: str) -> bool:
    primary_aliases = len(re.findall(r"/unif[0-9A-Fa-f]+", str(primary or "")))
    candidate_aliases = len(re.findall(r"/unif[0-9A-Fa-f]+", str(candidate or "")))
    candidate_cids = len(re.findall(r"\(cid:\d+\)", str(candidate or ""), re.IGNORECASE))
    candidate_length = len(str(candidate or "").strip())
    minimum_length = max(80, int(len(str(primary or "").strip()) * 0.35))
    return bool(
        primary_aliases
        and not candidate_cids
        and candidate_aliases < primary_aliases
        and candidate_length >= minimum_length
    )


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Install pypdf to extract NCERT PDFs.")
    reader = PdfReader(str(pdf_path))
    repaired_fonts = _repair_legacy_ncert_font_maps(reader)
    if repaired_fonts:
        logger.warning(
            "Repaired legacy NCERT font mapping | pdf=%s fonts=%s",
            pdf_path.name,
            ", ".join(repaired_fonts),
        )

    pages: List[Dict[str, Any]] = []
    plumber_document = None
    try:
        for index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            extraction_engine = "pypdf"
            recovered_aliases = 0
            if "/unif" in raw_text and pdfplumber is not None:
                if plumber_document is None:
                    plumber_document = pdfplumber.open(str(pdf_path))
                fallback_text = plumber_document.pages[index - 1].extract_text() or ""
                if _prefer_formula_fallback(raw_text, fallback_text):
                    recovered_aliases = len(
                        re.findall(r"/unif[0-9A-Fa-f]+", raw_text)
                    ) - len(re.findall(r"/unif[0-9A-Fa-f]+", fallback_text))
                    raw_text = fallback_text
                    extraction_engine = "pdfplumber_formula_fallback"
            raw_text, normalized_formula_glyphs = _normalize_pdf_formula_glyphs(raw_text)
            # NCERT's bullet glyph is exposed as a C1 control by non-Bookman Symbol
            # resources. Other residual control glyphs are layout artifacts, not
            # instructional formula characters, and are normalised to spaces.
            raw_text = raw_text.replace("\x9a", "•")
            raw_text = "".join(
                character
                if character in "\t\n\r" or not (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
                else " "
                for character in raw_text
            )
            raw_text = re.sub(r"(?<=\s)\ufffd(?=\s)", "–", raw_text)
            text = re.sub(r"[ \t]+", " ", raw_text).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            char_count = len(text)
            sanity = _extracted_text_sanity(text)
            length_quality = 0.0 if char_count == 0 else min(1.0, char_count / 900)
            sanity_multiplier = 0.15 if sanity["suspected_encoding_corruption"] else 1.0
            quality = length_quality * sanity_multiplier
            pages.append(
                {
                    "page_number": index,
                    "text": text,
                    "char_count": char_count,
                    "extraction_quality": round(quality, 3),
                    "text_sanity": sanity,
                    "decoded_fonts": repaired_fonts,
                    "extraction_engine": extraction_engine,
                    "formula_aliases_recovered": recovered_aliases,
                    "formula_glyphs_normalized": normalized_formula_glyphs,
                }
            )
        typical_page_length = median(
            [page["char_count"] for page in pages if page["char_count"] > 0]
        ) if pages else 0
        for extracted_page in pages:
            abnormal_length = bool(
                typical_page_length
                and extracted_page["char_count"] >= 8_000
                and extracted_page["char_count"] > typical_page_length * 4
            )
            extracted_page["text_sanity"]["abnormal_length"] = abnormal_length
            if abnormal_length:
                extracted_page["text_sanity"]["suspected_encoding_corruption"] = True
                extracted_page["extraction_quality"] = round(
                    float(extracted_page["extraction_quality"]) * 0.15,
                    3,
                )
    finally:
        if plumber_document is not None:
            plumber_document.close()
    return pages


def chunk_pages(
    pages: Sequence[Dict[str, Any]],
    *,
    max_chars: int = 1400,
    min_chars: int = 220,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("text") or "").strip()
        if not text:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z0-9])", text) if part.strip()]
        buffer: List[str] = []
        buffer_len = 0
        chunk_index = 0

        def flush() -> None:
            nonlocal buffer, buffer_len, chunk_index
            combined = " ".join(buffer).strip()
            if len(combined) < min_chars and chunks:
                chunks[-1]["text"] = f'{chunks[-1]["text"]}\n\n{combined}'.strip()
                chunks[-1]["token_estimate"] = estimate_tokens(chunks[-1]["text"])
                chunks[-1]["lexical_terms"] = content_terms(chunks[-1]["text"])
            elif combined:
                chunk_index += 1
                chunks.append(
                    {
                        "chunk_id": "",
                        "text": combined,
                        "page_start": page_number,
                        "page_end": page_number,
                        "section_title": infer_section_title(combined),
                        "token_estimate": estimate_tokens(combined),
                        "lexical_terms": content_terms(combined),
                    }
                )
            buffer = []
            buffer_len = 0

        for paragraph in paragraphs:
            if buffer and buffer_len + len(paragraph) > max_chars:
                flush()
            buffer.append(paragraph)
            buffer_len += len(paragraph)
        flush()

    return chunks


def infer_section_title(text: str) -> str:
    first_line = str(text or "").strip().splitlines()[0] if text else ""
    candidate = first_line[:90].strip()
    if len(candidate.split()) <= 9 and not candidate.endswith("."):
        return candidate
    return ""


def estimate_tokens(text: str) -> int:
    return max(1, round(len(str(text or "")) / 4))


def validate_concept_payloads(
    payload: Any,
    *,
    available_pages: Iterable[int],
) -> Tuple[List[ContentConceptPayload], List[Dict[str, Any]]]:
    raw_items = payload if isinstance(payload, list) else payload.get("concepts", []) if isinstance(payload, dict) else []
    issues: List[Dict[str, Any]] = []
    validated: List[ContentConceptPayload] = []
    seen_ids: set[str] = set()
    page_set = {int(page) for page in available_pages}

    if not isinstance(raw_items, list):
        return [], [{"severity": "error", "message": "Concept payload must be a JSON array or an object with concepts[]."}]

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            issues.append({"severity": "error", "index": index, "message": "Concept item must be an object."})
            continue
        try:
            concept = ContentConceptPayload.model_validate(item)
        except ValidationError as exc:
            issues.append({"severity": "error", "index": index, "message": "Concept schema validation failed.", "details": exc.errors()})
            continue
        if concept.concept_id in seen_ids:
            # Overlapping batches re-emit the same concept; keep the first
            # occurrence and drop the rest instead of flagging duplicates.
            continue
        seen_ids.add(concept.concept_id)
        concept_issues: List[str] = []
        if not concept.definition and not concept.core_explanation and not concept.key_points:
            concept_issues.append("missing_teaching_content")
        if not concept.source_pages:
            concept_issues.append("missing_source_pages")
        elif any(page not in page_set for page in concept.source_pages):
            concept_issues.append("source_page_out_of_range")
        if concept_issues:
            issues.append(
                {
                    "severity": "error" if "missing_source_pages" in concept_issues else "warning",
                    "concept_id": concept.concept_id,
                    "message": "Concept has validation issues.",
                    "issues": concept_issues,
                }
            )
        validated.append(concept)

    return validated, issues


def build_coverage_report(
    pages: Sequence[Dict[str, Any]],
    concepts: Sequence[ContentConceptPayload] | Sequence[ContentConcept],
    chunks: Sequence[Dict[str, Any]] | Sequence[ContentChunk],
    issues: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    page_numbers = {int(page["page_number"] if isinstance(page, dict) else page.page_number) for page in pages}
    extracted_pages = {
        int(page["page_number"] if isinstance(page, dict) else page.page_number)
        for page in pages
        if int(page["char_count"] if isinstance(page, dict) else page.char_count or 0) > 0
    }
    concept_pages: set[int] = set()
    for concept in concepts:
        source_pages = concept.source_pages if not isinstance(concept, dict) else concept.get("source_pages", [])
        concept_pages.update(int(page) for page in source_pages or [] if int(page) > 0)

    covered_pages = extracted_pages & concept_pages if concepts else set()
    extraction_quality = 0.0
    if pages:
        extraction_quality = sum(float(page["extraction_quality"] if isinstance(page, dict) else page.extraction_quality or 0.0) for page in pages) / len(pages)
    coverage_score = (len(covered_pages) / len(extracted_pages)) if extracted_pages and concepts else 0.0
    blocking_issues = [issue for issue in issues if issue.get("severity") == "error"]
    missing_pages = sorted(extracted_pages - concept_pages) if concepts else sorted(extracted_pages)

    return {
        "page_count": len(page_numbers),
        "extracted_page_count": len(extracted_pages),
        "chunk_count": len(chunks),
        "concept_count": len(concepts),
        "pages_referenced_by_concepts": sorted(concept_pages),
        "missing_source_pages": missing_pages,
        "coverage_score": round(coverage_score, 3),
        "extraction_quality": round(extraction_quality, 3),
        "issues": list(issues),
        "blocking_issue_count": len(blocking_issues),
        "ready_for_approval": bool(concepts and not blocking_issues and coverage_score >= _min_coverage_score()),
    }


def create_job(db: Session, *, job_type: str, source_path: str) -> ContentIngestionJob:
    job = ContentIngestionJob(
        job_id=f"content_job_{uuid.uuid4().hex[:14]}",
        job_type=job_type,
        status="running",
        source_path=source_path,
        summary={},
    )
    db.add(job)
    db.flush()
    return job


def ingest_pdf_file(
    db: Session,
    pdf_path: Path,
    *,
    root_path: Optional[Path] = None,
    replace: bool = True,
) -> ContentChapter:
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file not found: {pdf_path}")

    metadata = infer_metadata_from_pdf_path(pdf_path, root_path)
    source_hash = file_sha256(pdf_path)
    chapter = db.query(ContentChapter).filter(ContentChapter.slug == metadata["slug"]).one_or_none()
    if chapter is None and metadata.get("chapter_number") is not None:
        identity_matches = (
            db.query(ContentChapter)
            .filter(
                ContentChapter.board == metadata["board"],
                ContentChapter.class_level == metadata["class_level"],
                ContentChapter.subject == metadata["subject"],
                ContentChapter.chapter_number == metadata["chapter_number"],
            )
            .order_by(ContentChapter.id)
            .all()
        )
        hash_matches = [row for row in identity_matches if row.source_hash == source_hash]
        if len(hash_matches) == 1:
            chapter = hash_matches[0]
        elif len(identity_matches) == 1:
            # A new edition of the same curriculum chapter replaces that
            # chapter and is versioned on approval; it must not create a second
            # row merely because the descriptive filename (and slug) changed.
            chapter = identity_matches[0]
        elif len(identity_matches) > 1:
            raise ValueError(
                "Ambiguous curriculum identity: multiple chapter rows exist for "
                f"{metadata['board']} Class {metadata['class_level']} "
                f"{metadata['subject']} chapter {metadata['chapter_number']}."
            )
    if chapter is None:
        chapter = ContentChapter(slug=metadata["slug"])
        db.add(chapter)

    for key, value in metadata.items():
        setattr(chapter, key, value)
    chapter.pdf_path = str(pdf_path)
    chapter.source_hash = source_hash
    chapter.status = "uploaded"
    # Keep the version across re-ingests; approval bumps it when the source
    # hash differs from what students last saw.
    chapter.version = chapter.version or DEFAULT_VERSION
    chapter.updated_at = datetime.utcnow()
    db.flush()

    if replace:
        db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).delete(synchronize_session=False)
        db.query(ContentChunk).filter(ContentChunk.chapter_id == chapter.id).delete(synchronize_session=False)

    pages = extract_pdf_pages(pdf_path)
    for page in pages:
        page_metadata = {
            "source": "pdf_extraction",
            "text_sanity": page.get("text_sanity") or {},
            "extraction_engine": page.get("extraction_engine") or "pypdf",
            "formula_aliases_recovered": int(page.get("formula_aliases_recovered") or 0),
            "formula_glyphs_normalized": int(page.get("formula_glyphs_normalized") or 0),
        }
        if page.get("decoded_fonts"):
            page_metadata["decoded_fonts"] = page["decoded_fonts"]
        db.add(
            ContentPage(
                chapter_id=chapter.id,
                page_number=page["page_number"],
                text=page["text"],
                char_count=page["char_count"],
                extraction_quality=page["extraction_quality"],
                metadata_json=page_metadata,
            )
        )
    db.flush()

    chunks = chunk_pages(pages)
    chunk_vectors: List[Optional[List[float]]] = [None] * len(chunks)
    if chunks and embeddings_service.embeddings_enabled():
        try:
            chunk_vectors = embeddings_service.embed_texts([chunk["text"] for chunk in chunks])
        except Exception:
            logger.exception(
                "Chunk embedding failed during ingest; storing chunks without embeddings | chapter=%s",
                chapter.slug,
            )
            chunk_vectors = [None] * len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = f"{chapter.slug}_chunk_{index:04d}"
        chunk["chunk_id"] = chunk_id
        vector = chunk_vectors[index - 1]
        metadata = {
            "board": chapter.board,
            "class": chapter.class_level,
            "subject": chapter.subject,
            "chapter": chapter.chapter_name,
            "source": "pdf",
        }
        if vector:
            metadata["embedding_model"] = embeddings_service.embedding_model()
        db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id=chunk_id,
                text=chunk["text"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                section_title=chunk["section_title"],
                token_estimate=chunk["token_estimate"],
                lexical_terms=chunk["lexical_terms"],
                embedding=vector,
                metadata_json=metadata,
            )
        )

    extraction_issues = [
        {
            "severity": "error",
            "code": "suspected_encoding_corruption",
            "page": page["page_number"],
            "message": "Extracted PDF text appears font-encoded or cipher-like.",
        }
        for page in pages
        if (page.get("text_sanity") or {}).get("suspected_encoding_corruption")
    ]
    report = build_coverage_report(pages, [], chunks, extraction_issues)
    chapter.page_count = report["page_count"]
    chapter.extracted_page_count = report["extracted_page_count"]
    chapter.chunk_count = report["chunk_count"]
    chapter.extraction_quality = report["extraction_quality"]
    chapter.coverage_score = report["coverage_score"]
    chapter.validation_report = report
    chapter.status = "indexed" if chunks else "failed"
    db.flush()
    return chapter


def run_ingest_folder_job(
    db: Session,
    job: ContentIngestionJob,
    *,
    root_path: Optional[str] = None,
    replace: bool = True,
) -> Dict[str, Any]:
    """Execute folder ingestion against an existing job row (sync or worker)."""
    root = safe_data_path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    chapters: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    try:
        pdfs = sorted(root.rglob("*.pdf"))
        for pdf_path in pdfs:
            try:
                chapter = ingest_pdf_file(db, pdf_path, root_path=root, replace=replace)
                chapters.append(serialize_chapter(chapter))
            except Exception as exc:
                logger.exception("Content ingestion failed for %s", pdf_path)
                errors.append({"path": str(pdf_path), "error": str(exc)})
        job.status = "completed" if not errors else "needs_review"
        job.summary = {
            **(job.summary or {}),
            "root": str(root),
            "pdf_count": len(pdfs),
            "chapters": len(chapters),
            "errors": errors,
        }
        db.commit()
        return {"job": serialize_job(job), "chapters": chapters, "errors": errors}
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        db.commit()
        raise


def ingest_pdf_folder(db: Session, root_path: Optional[str] = None, *, replace: bool = True) -> Dict[str, Any]:
    root = safe_data_path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    job = create_job(db, job_type="ingest_folder", source_path=str(root))
    return run_ingest_folder_job(db, job, root_path=root_path, replace=replace)


def import_concepts_for_chapter(
    db: Session,
    chapter_id: int,
    payload: Any,
    *,
    replace: bool = True,
) -> ContentChapter:
    chapter = db.query(ContentChapter).filter(ContentChapter.id == chapter_id).one_or_none()
    if chapter is None:
        raise ValueError(f"Chapter not found: {chapter_id}")
    pages = db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).order_by(ContentPage.page_number).all()
    available_pages = [page.page_number for page in pages]
    concepts, issues = validate_concept_payloads(payload, available_pages=available_pages)
    extraction_issues = [
        {
            "severity": "error",
            "code": "suspected_encoding_corruption",
            "page": page.page_number,
            "message": "Extracted PDF text appears font-encoded or cipher-like.",
        }
        for page in pages
        if ((page.metadata_json or {}).get("text_sanity") or {}).get(
            "suspected_encoding_corruption"
        )
    ]
    issues = [*extraction_issues, *issues]

    if replace:
        db.query(ContentConcept).filter(ContentConcept.chapter_id == chapter.id).delete(synchronize_session=False)
    for concept in concepts:
        raw = concept.model_dump()
        concept_issues = [issue for issue in issues if issue.get("concept_id") == concept.concept_id]
        db.add(
            ContentConcept(
                chapter_id=chapter.id,
                concept_id=concept.concept_id,
                title=concept.title,
                definition=concept.definition,
                core_explanation=concept.core_explanation,
                key_points=concept.key_points,
                examples=concept.examples,
                formulas=concept.formulas,
                properties=concept.properties,
                applications=concept.applications,
                common_mistakes=concept.common_mistakes,
                prerequisites=concept.prerequisites,
                related_concepts=concept.related_concepts,
                learning_objectives=concept.learning_objectives,
                source_pages=concept.source_pages,
                difficulty_level=concept.difficulty_level,
                blooms_taxonomy=concept.blooms_taxonomy,
                typical_exam_weightage=concept.typical_exam_weightage,
                importance_level=concept.importance_level,
                raw_json=raw,
                validation_issues=concept_issues,
            )
        )
    db.flush()
    chunks = db.query(ContentChunk).filter(ContentChunk.chapter_id == chapter.id).all()
    report = build_coverage_report(
        [{"page_number": page.page_number, "char_count": page.char_count, "extraction_quality": page.extraction_quality} for page in pages],
        concepts,
        chunks,
        issues,
    )
    chapter.concept_count = len(concepts)
    chapter.coverage_score = report["coverage_score"]
    chapter.extraction_quality = report["extraction_quality"]
    chapter.validation_report = report
    chapter.status = "validated" if report["ready_for_approval"] else "needs_review"
    chapter.updated_at = datetime.utcnow()
    db.flush()
    return chapter


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Parse one complete model JSON array without salvaging partial output.

    A valid prefix is not a valid batch: accepting it would silently drop NCERT
    material while later coverage accounting credits the batch as complete.
    """
    cleaned = str(text or "").strip()
    fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("model response must be one complete JSON array")
    if any(not isinstance(item, dict) for item in data):
        raise ValueError("every generated concept must be a JSON object")
    return data


def _page_batches(pages: Sequence[ContentPage], max_chars: int) -> List[List[ContentPage]]:
    batches: List[List[ContentPage]] = []
    current: List[ContentPage] = []
    current_len = 0
    for page in pages:
        page_text = (page.text or "").strip()
        if not page_text:
            continue
        if current and current_len + len(page_text) > max_chars:
            batches.append(current)
            current = []
            current_len = 0
        current.append(page)
        current_len += len(page_text)
    if current:
        batches.append(current)
    return batches


@dataclass(frozen=True)
class _GenerationPageSlice:
    """A source-preserving fragment used only when one model request is too large."""

    page_number: int
    text: str


def _generation_messages(
    chapter: ContentChapter,
    batch: Sequence[ContentPage | _GenerationPageSlice],
    *,
    batch_label: str,
) -> List[Dict[str, str]]:
    page_text = "\n\n".join(
        f"[PAGE {page.page_number}]\n{(page.text or '').strip()}"
        for page in batch
    )
    return [
        {"role": "system", "content": CONTENT_GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Board: {chapter.board}\n"
                f"Class: {chapter.class_level}\n"
                f"Subject: {chapter.subject}\n"
                f"Chapter: {chapter.chapter_name}\n"
                f"Batch: {batch_label}\n\n"
                f"{page_text}"
            ),
        },
    ]


def _estimate_generation_input_tokens(messages: Sequence[Dict[str, str]]) -> int:
    """Conservatively estimate input tokens for formula- and table-heavy NCERT text.

    The shared cost estimator is intentionally lightweight (roughly four
    characters per token).  That materially under-counts logarithm tables,
    equations, and other short-token-dense textbook content.  The additional
    signals below deliberately over-estimate those inputs, while the fixed
    buffer covers message framing and tokenizer differences.
    """

    text = "\n".join(str(message.get("content") or "") for message in messages)
    # Keep this estimator local: importing Logic.coach.costing at module load
    # would execute Logic.coach.__init__, whose retriever imports this module.
    generic_estimate = max(1, (len(text) + 3) // 4)
    non_space_runs = len(re.findall(r"\S+", text))
    byte_count = len(text.encode("utf-8"))
    return max(
        generic_estimate * 2,
        (non_space_runs * 5 + 1) // 2,
        (byte_count + 2) // 3,
    ) + 128


def _generation_output_limit(messages: Sequence[Dict[str, str]]) -> int:
    return min(
        CONTENT_GENERATION_MAX_TOKENS,
        CONTENT_GENERATION_REQUEST_TOKEN_BUDGET
        - _estimate_generation_input_tokens(messages),
    )


def _split_generation_text(text: str) -> Tuple[str, str]:
    """Split an oversized page near a readable boundary without dropping text."""

    value = str(text or "").strip()
    if len(value) < 2:
        return value, ""
    midpoint = len(value) // 2
    lower_bound = max(1, midpoint // 2)
    upper_bound = min(len(value) - 1, midpoint + midpoint // 2)
    candidates = [
        value.rfind("\n", lower_bound, upper_bound),
        value.rfind(". ", lower_bound, upper_bound),
        value.rfind(" ", lower_bound, upper_bound),
    ]
    split_at = max(candidates)
    if split_at < lower_bound:
        split_at = midpoint
    elif value[split_at : split_at + 2] == ". ":
        split_at += 1
    return value[:split_at].strip(), value[split_at:].strip()


def _generation_request_batches(
    chapter: ContentChapter,
    batch: Sequence[ContentPage | _GenerationPageSlice],
    *,
    batch_label: str,
    minimum_output_tokens: int = CONTENT_GENERATION_MIN_OUTPUT_TOKENS,
) -> List[List[_GenerationPageSlice]]:
    """Fit one legacy page batch into safe, lossless model requests.

    Page batches remain the stable outer checkpoint boundary.  Only an
    uncached oversized batch is divided, so completed legacy checkpoints keep
    their exact signatures and retries resume at the smallest successful
    request fragment.
    """

    pending = [
        _GenerationPageSlice(
            page_number=int(page.page_number),
            text=(page.text or "").strip(),
        )
        for page in batch
        if (page.text or "").strip()
    ]
    fitted: List[_GenerationPageSlice] = []
    while pending:
        page_slice = pending.pop(0)
        messages = _generation_messages(
            chapter,
            [page_slice],
            batch_label=batch_label,
        )
        if _generation_output_limit(messages) >= minimum_output_tokens:
            fitted.append(page_slice)
            continue
        left, right = _split_generation_text(page_slice.text)
        if not left or not right:
            raise ValueError(
                f"NCERT page {page_slice.page_number} cannot fit within the "
                "content-generation request token budget."
            )
        pending[0:0] = [
            _GenerationPageSlice(page_slice.page_number, left),
            _GenerationPageSlice(page_slice.page_number, right),
        ]

    requests: List[List[_GenerationPageSlice]] = []
    current: List[_GenerationPageSlice] = []
    for page_slice in fitted:
        candidate = [*current, page_slice]
        messages = _generation_messages(
            chapter,
            candidate,
            batch_label=batch_label,
        )
        if (
            current
            and _generation_output_limit(messages)
            < minimum_output_tokens
        ):
            requests.append(current)
            current = [page_slice]
        else:
            current = candidate
    if current:
        requests.append(current)
    return requests


def _generation_batch_is_reference_only(
    batch: Sequence[ContentPage | _GenerationPageSlice],
) -> bool:
    """Identify the numeric log-table appendix that carries no teaching prose.

    NCERT Chemistry includes multi-page logarithm/antilogarithm tables followed
    by blank Notes pages.  Asking the model to invent a concept for those pages
    is both wasteful and educationally wrong.  Keep this intentionally narrow:
    at least one page must explicitly be a log/antilog table, and every
    companion page must be another numeric table or a short Notes page.
    """

    texts = [str(page.text or "").strip() for page in batch if str(page.text or "").strip()]
    if not texts:
        return False

    def first_line(value: str) -> str:
        return next((line.strip().lower() for line in value.splitlines() if line.strip()), "")

    def is_log_table(value: str) -> bool:
        heading = first_line(value)
        return bool(re.match(r"^(?:anti\s*)?logarithms\b", heading))

    def is_table_companion(value: str) -> bool:
        heading = first_line(value)
        if heading == "notes" and len(value) < 500:
            return True
        if not re.match(r"^table\s+(?:i|ii|1|2)\b", heading, re.IGNORECASE):
            return False
        non_space = max(1, sum(not character.isspace() for character in value))
        numeric = sum(character.isdigit() for character in value)
        return numeric / non_space >= 0.45

    return any(is_log_table(value) for value in texts) and all(
        is_log_table(value) or is_table_companion(value)
        for value in texts
    )


def _generation_cache_root() -> Path:
    configured = str(os.getenv("CONTENT_GENERATION_CACHE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (DATA_DIR / "processed" / "content_generation_cache").resolve()


def _content_generation_route_policy(gateway: Any, selected_model: str) -> Dict[str, Any]:
    """Fingerprint the routes that may legitimately produce a cached batch."""
    router = getattr(gateway, "router", None)
    routes: List[Dict[str, str]] = []
    if router is not None and hasattr(router, "_candidate_routes"):
        try:
            candidates = router._candidate_routes(  # noqa: SLF001 - same internal routing subsystem
                role="reviewer",
                complexity="balanced",
                input_tokens=1,
                output_tokens=CONTENT_GENERATION_MAX_TOKENS,
            )
        except Exception:  # noqa: BLE001 - policy still invalidates on env changes
            candidates = []
        for route in candidates:
            provider = str(getattr(route, "provider", "") or "").strip().lower()
            model = str(getattr(route, "model", "") or "").strip()
            if provider and model and {"provider": provider, "model": model} not in routes:
                routes.append({"provider": provider, "model": model})
    return {
        "selected_model": str(selected_model or "unknown"),
        "routes": routes,
        "provider_order": str(
            os.getenv("COACH_PROVIDER_ORDER")
            or os.getenv("COACH_LLM_PROVIDER")
            or "groq"
        ),
        "route_preference": str(os.getenv("COACH_ROUTE_PREFERENCE") or "balanced"),
        "max_attempts": str(os.getenv("COACH_LLM_MAX_ATTEMPTS") or ""),
    }


def _default_generation_batch_delay(route_policy: Dict[str, Any]) -> float:
    """Return conservative provider pacing when no operator override exists."""

    providers = {
        str(route.get("provider") or "").strip().lower()
        for route in (route_policy.get("routes") or [])
        if isinstance(route, dict)
    }
    providers.update(
        item.strip().lower()
        for item in str(route_policy.get("provider_order") or "").split(",")
        if item.strip()
    )
    return CONTENT_GENERATION_GROQ_BATCH_DELAY_SECONDS if "groq" in providers else 0.0


def _generation_batch_cache_path(
    chapter: ContentChapter,
    batch: Sequence[ContentPage | _GenerationPageSlice],
    *,
    batch_index: int,
    batch_count: int,
    max_batch_chars: int,
    route_policy: Dict[str, Any],
    max_tokens: int = CONTENT_GENERATION_MAX_TOKENS,
) -> Tuple[Path, str]:
    """Return an exact-input checkpoint path and signature for one model batch.

    A checkpoint is reusable only when the source PDF, extracted page text,
    batching, prompt, and selected model all match. This prevents a retry from
    silently mixing content produced from different source or prompt versions.
    """
    page_payload = [
        {
            "page_number": int(page.page_number),
            "text_sha256": sha256((page.text or "").encode("utf-8")).hexdigest(),
        }
        for page in batch
    ]
    identity = {
        "schema_version": 2,
        "chapter_id": int(chapter.id),
        "chapter_slug": str(chapter.slug or ""),
        "board": str(chapter.board or ""),
        "class_level": str(chapter.class_level or ""),
        "subject": str(chapter.subject or ""),
        "chapter_name": str(chapter.chapter_name or ""),
        "source_hash": str(chapter.source_hash or ""),
        "prompt_version": CONTENT_GENERATION_PROMPT_VERSION,
        "prompt_sha256": sha256(
            CONTENT_GENERATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "route_policy": route_policy,
        "temperature": CONTENT_GENERATION_TEMPERATURE,
        "max_tokens": int(max_tokens),
        "max_batch_chars": int(max_batch_chars),
        "batch_index": int(batch_index),
        "batch_count": int(batch_count),
        "pages": page_payload,
    }
    signature = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    chapter_key = normalize_key(chapter.slug or chapter.chapter_name or chapter.id)
    return _generation_cache_root() / chapter_key / f"{signature}.json", signature


def _load_generation_batch_cache(
    path: Path,
    signature: str,
    route_policy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return []
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or payload.get("signature") != signature
    ):
        return []
    model_record = payload.get("model")
    if not isinstance(model_record, dict):
        return []
    actual_pair = {
        "provider": str(model_record.get("provider") or "").strip().lower(),
        "model": str(model_record.get("model") or "").strip(),
    }
    allowed_pairs = [
        {
            "provider": str(item.get("provider") or "").strip().lower(),
            "model": str(item.get("model") or "").strip(),
        }
        for item in route_policy.get("routes", [])
        if isinstance(item, dict)
    ]
    if not actual_pair["provider"] or not actual_pair["model"] or actual_pair not in allowed_pairs:
        return []
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return []
    return [item for item in items if isinstance(item, dict)]


def _store_generation_batch_cache(
    path: Path,
    signature: str,
    items: Sequence[Dict[str, Any]],
    *,
    model_record: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically persist a fully parsed batch; incomplete responses are never cached."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "signature": signature,
        "prompt_version": CONTENT_GENERATION_PROMPT_VERSION,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": {
            key: model_record.get(key)
            for key in ("provider", "model", "fallback")
            if model_record and model_record.get(key) is not None
        },
        "items": list(items),
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _normalize_generated_batch_items(
    items: Sequence[Dict[str, Any]],
    batch_pages: Sequence[int],
) -> List[Dict[str, Any]]:
    """Validate a complete model batch and enforce page-local source grounding."""
    if not items:
        raise ValueError("model returned an empty concept array")
    allowed_pages = {int(page) for page in batch_pages}
    normalized: List[Dict[str, Any]] = []
    for raw_item in items:
        item = dict(raw_item)
        cited_pages = coerce_page_numbers(item.get("source_pages"))
        if not cited_pages:
            raise ValueError("every generated concept must cite at least one supplied source page")
        if any(page not in allowed_pages for page in cited_pages):
            raise ValueError(
                "concept cites a source page outside its supplied generation batch"
            )
        item["source_pages"] = cited_pages
        normalized.append(ContentConceptPayload.model_validate(item).model_dump())
    return normalized


def _unique_values(values: Iterable[Any]) -> List[Any]:
    """Stable de-duplication for both strings and structured formula/mistake data."""
    result: List[Any] = []
    seen: set[str] = set()
    for value in values:
        try:
            marker = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            marker = str(value)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _ranked_text(values: Iterable[Any], ranks: Dict[str, int], default: str = "") -> str:
    cleaned = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not cleaned:
        return default
    return max(
        cleaned,
        key=lambda value: (
            ranks.get(value.lower(), 0),
            len(value),
        ),
    )


def consolidate_concept_payloads(
    payload: Sequence[Dict[str, Any]] | Sequence[ContentConceptPayload],
    *,
    chapter_key: str,
    page_count: int = 0,
) -> List[Dict[str, Any]]:
    """Merge micro-concepts into complete learning units without dropping content.

    Group selection is content-aware and syllabus ordered (see
    services.topic_grouping).  This function performs the actual material
    merge: every source page, explanation, formula, example, objective and
    mistake from the original concepts is retained in the resulting units.
    """
    concepts: List[ContentConceptPayload] = []
    rejected_payloads: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_payloads: set[str] = set()
    for item in payload:
        try:
            concept = item if isinstance(item, ContentConceptPayload) else ContentConceptPayload.model_validate(item)
        except ValidationError:
            # Keep malformed objects in the import payload so validation can
            # report and block them; do not silently erase source material just
            # because other concepts in the chapter were valid.
            if isinstance(item, dict):
                rejected_payloads.append(dict(item))
            continue

        serialized = json.dumps(concept.model_dump(), sort_keys=True, ensure_ascii=False)
        if serialized in seen_payloads:
            # An exact replay contains no additional source material.
            continue
        seen_payloads.add(serialized)

        if concept.concept_id in seen_ids:
            # Concept generation runs in independent page batches, so two
            # different syllabus ideas can legitimately receive the same
            # model-generated ID. Preserve both and keep the original ID as a
            # legacy alias instead of silently discarding the later batch.
            original_id = concept.concept_id
            digest = sha256(serialized.encode("utf-8")).hexdigest()[:10]
            suffix = f"_{digest}"
            disambiguated_id = f"{original_id[:140 - len(suffix)]}{suffix}"
            collision_index = 2
            while disambiguated_id in seen_ids:
                numbered_suffix = f"_{digest}_{collision_index}"
                disambiguated_id = (
                    f"{original_id[:140 - len(numbered_suffix)]}{numbered_suffix}"
                )
                collision_index += 1
            concept = concept.model_copy(
                update={
                    "concept_id": disambiguated_id,
                    "source_concept_ids": _unique_values(
                        [original_id, *concept.source_concept_ids]
                    ),
                }
            )
        seen_ids.add(concept.concept_id)
        concepts.append(concept)

    if not concepts:
        return [dict(item) for item in payload if isinstance(item, dict)]

    units = build_learning_units(
        concepts,
        chapter_key=chapter_key,
        page_count=page_count,
    )
    merged: List[Dict[str, Any]] = []
    bloom_rank = {
        "remember": 1,
        "understand": 2,
        "apply": 3,
        "analyze": 4,
        "analyse": 4,
        "evaluate": 5,
        "create": 6,
    }
    weight_rank = {"none": 0, "low": 1, "medium": 2, "moderate": 2, "high": 3}
    importance_rank = {
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

    for unit in units:
        members: List[ContentConceptPayload] = list(unit["concepts"])
        if len(members) == 1:
            raw = members[0].model_dump()
            raw["source_concept_ids"] = _unique_values(
                [members[0].concept_id, *members[0].source_concept_ids]
            )
            merged.append(raw)
            continue

        original_ids = _unique_values(
            source_id
            for member in members
            for source_id in [member.concept_id, *member.source_concept_ids]
            if normalize_key(source_id)
        )
        explanations: List[str] = []
        for member in members:
            detail = "\n\n".join(
                part.strip()
                for part in (member.definition, member.core_explanation)
                if part and part.strip()
            )
            if detail:
                explanations.append(f"{member.title}: {detail}")

        merged.append(
            {
                "concept_id": unit["id"],
                "title": unit["label"],
                "definition": next(
                    (member.definition.strip() for member in members if member.definition.strip()),
                    "",
                ),
                "core_explanation": "\n\n".join(_unique_values(explanations)),
                "key_points": _unique_values(
                    point for member in members for point in member.key_points
                ),
                "examples": _unique_values(
                    example for member in members for example in member.examples
                ),
                "formulas": _unique_values(
                    formula for member in members for formula in member.formulas
                ),
                "properties": _unique_values(
                    prop for member in members for prop in member.properties
                ),
                "applications": _unique_values(
                    application for member in members for application in member.applications
                ),
                "common_mistakes": _unique_values(
                    mistake for member in members for mistake in member.common_mistakes
                ),
                "prerequisites": _unique_values(
                    prerequisite for member in members for prerequisite in member.prerequisites
                ),
                "related_concepts": _unique_values(
                    related for member in members for related in member.related_concepts
                ),
                "learning_objectives": _unique_values(
                    objective for member in members for objective in member.learning_objectives
                ),
                "source_pages": sorted(
                    {page for member in members for page in member.source_pages}
                ),
                "difficulty_level": max(member.difficulty_level for member in members),
                "blooms_taxonomy": _ranked_text(
                    (member.blooms_taxonomy for member in members), bloom_rank
                ),
                "typical_exam_weightage": _ranked_text(
                    (member.typical_exam_weightage for member in members), weight_rank
                ),
                "importance_level": _ranked_text(
                    (member.importance_level for member in members), importance_rank
                ),
                "source_concept_ids": original_ids,
            }
        )
    return [*merged, *rejected_payloads]


def generate_concepts_for_chapter(
    db: Session,
    chapter_id: int,
    *,
    replace: bool = True,
    max_batch_chars: int = 9000,
) -> ContentChapter:
    """Generate draft concept JSON from extracted pages using configured model routing."""
    chapter = db.query(ContentChapter).filter(ContentChapter.id == chapter_id).one_or_none()
    if chapter is None:
        raise ValueError(f"Chapter not found: {chapter_id}")
    pages = db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).order_by(ContentPage.page_number).all()
    if not pages:
        raise ValueError("Chapter has no extracted pages. Run ingestion first.")

    from Logic.coach.model_gateway import model_gateway

    generated: List[Dict[str, Any]] = []
    failed_batches = 0
    batches = _page_batches(pages, max_chars=max_batch_chars)
    model_name = model_gateway.model_for("reviewer", complexity="balanced")
    route_policy = _content_generation_route_policy(model_gateway, model_name)
    default_batch_delay = _default_generation_batch_delay(route_policy)
    try:
        configured_delay = os.getenv("CONTENT_GENERATION_BATCH_DELAY_SECONDS")
        batch_delay_seconds = max(
            0.0,
            float(configured_delay) if configured_delay not in (None, "") else default_batch_delay,
        )
    except ValueError:
        logger.warning(
            "Invalid CONTENT_GENERATION_BATCH_DELAY_SECONDS; using provider-safe default %.1fs",
            default_batch_delay,
        )
        batch_delay_seconds = default_batch_delay
    provider_calls_made = 0
    for batch_index, batch in enumerate(batches, start=1):
        batch_pages = [page.page_number for page in batch]
        if _generation_batch_is_reference_only(batch):
            logger.info(
                "Skipping non-instructional logarithm appendix batch %s/%s for %s",
                batch_index,
                len(batches),
                chapter.slug,
            )
            continue
        # Check the original page-batch signature first.  This preserves every
        # successful checkpoint produced before request-budget enforcement.
        legacy_cache_path, legacy_cache_signature = _generation_batch_cache_path(
            chapter,
            batch,
            batch_index=batch_index,
            batch_count=len(batches),
            max_batch_chars=max_batch_chars,
            route_policy=route_policy,
        )
        cached_items = _load_generation_batch_cache(
            legacy_cache_path,
            legacy_cache_signature,
            route_policy,
        )
        if cached_items:
            try:
                cached_items = _normalize_generated_batch_items(
                    cached_items,
                    batch_pages,
                )
            except (TypeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "Ignoring invalid concept-generation checkpoint %s/%s for %s: %s",
                    batch_index,
                    len(batches),
                    chapter.slug,
                    exc,
                )
            else:
                logger.info(
                    "Using concept-generation checkpoint %s/%s for %s",
                    batch_index,
                    len(batches),
                    chapter.slug,
                )
                generated.extend(cached_items)
                continue
        request_batches = _generation_request_batches(
            chapter,
            batch,
            batch_label=f"{batch_index}/{len(batches)}",
        )
        for request_index, request_batch in enumerate(request_batches, start=1):
            request_pages = [page.page_number for page in request_batch]
            request_label = f"{batch_index}/{len(batches)}"
            if len(request_batches) > 1:
                request_label += f" (part {request_index}/{len(request_batches)})"
            messages = _generation_messages(
                chapter,
                request_batch,
                batch_label=request_label,
            )
            request_max_tokens = _generation_output_limit(messages)
            if request_max_tokens < CONTENT_GENERATION_MIN_OUTPUT_TOKENS:
                raise ValueError(
                    f"Concept-generation request {request_label} exceeds the safe "
                    "token budget after source-preserving splitting."
                )
            cache_batch_index = (
                batch_index
                if len(request_batches) == 1
                else batch_index * 1000 + request_index
            )
            cache_path, cache_signature = _generation_batch_cache_path(
                chapter,
                request_batch,
                batch_index=cache_batch_index,
                batch_count=len(batches),
                max_batch_chars=max_batch_chars,
                route_policy=route_policy,
                max_tokens=request_max_tokens,
            )
            cached_items = _load_generation_batch_cache(
                cache_path,
                cache_signature,
                route_policy,
            )
            if cached_items:
                try:
                    cached_items = _normalize_generated_batch_items(
                        cached_items,
                        request_pages,
                    )
                except (TypeError, ValueError, ValidationError) as exc:
                    logger.warning(
                        "Ignoring invalid concept-generation checkpoint %s for %s: %s",
                        request_label,
                        chapter.slug,
                        exc,
                    )
                else:
                    logger.info(
                        "Using concept-generation checkpoint %s for %s",
                        request_label,
                        chapter.slug,
                    )
                    generated.extend(cached_items)
                    continue

            delivery_batches = [request_batch]
            if request_max_tokens < CONTENT_GENERATION_PREFERRED_OUTPUT_TOKENS:
                delivery_batches = _generation_request_batches(
                    chapter,
                    request_batch,
                    batch_label=request_label,
                    minimum_output_tokens=CONTENT_GENERATION_PREFERRED_OUTPUT_TOKENS,
                )
                logger.info(
                    "Split uncached concept-generation batch %s into %s output-safe part(s)",
                    request_label,
                    len(delivery_batches),
                )

            for delivery_index, delivery_batch in enumerate(delivery_batches, start=1):
                delivery_pages = [page.page_number for page in delivery_batch]
                delivery_label = request_label
                delivery_cache_path = cache_path
                delivery_cache_signature = cache_signature
                delivery_messages = messages
                delivery_max_tokens = request_max_tokens
                delivery_cached_items: List[Dict[str, Any]] = []
                if len(delivery_batches) > 1:
                    delivery_label += (
                        f" (output part {delivery_index}/{len(delivery_batches)})"
                    )
                    delivery_messages = _generation_messages(
                        chapter,
                        delivery_batch,
                        batch_label=delivery_label,
                    )
                    delivery_max_tokens = _generation_output_limit(delivery_messages)
                    delivery_cache_path, delivery_cache_signature = (
                        _generation_batch_cache_path(
                            chapter,
                            delivery_batch,
                            batch_index=(
                                batch_index * 1_000_000
                                + request_index * 1_000
                                + delivery_index
                            ),
                            batch_count=len(batches),
                            max_batch_chars=max_batch_chars,
                            route_policy=route_policy,
                            max_tokens=delivery_max_tokens,
                        )
                    )
                    delivery_cached_items = _load_generation_batch_cache(
                        delivery_cache_path,
                        delivery_cache_signature,
                        route_policy,
                    )
                if delivery_cached_items:
                    try:
                        delivery_cached_items = _normalize_generated_batch_items(
                            delivery_cached_items,
                            delivery_pages,
                        )
                    except (TypeError, ValueError, ValidationError) as exc:
                        logger.warning(
                            "Ignoring invalid concept-generation checkpoint %s for %s: %s",
                            delivery_label,
                            chapter.slug,
                            exc,
                        )
                    else:
                        logger.info(
                            "Using concept-generation checkpoint %s for %s",
                            delivery_label,
                            chapter.slug,
                        )
                        generated.extend(delivery_cached_items)
                        continue
                if provider_calls_made and batch_delay_seconds:
                    time.sleep(batch_delay_seconds)
                response = model_gateway.complete(
                    role="reviewer",
                    complexity="balanced",
                    agent_name="content_ingestion_agent",
                    task=(
                        f"generate_content_concepts:{chapter.slug}:batch_{batch_index}"
                        f":part_{request_index}:output_part_{delivery_index}"
                    ),
                    student_visible=False,
                    safety_tier="strict_source_grounding",
                    messages=delivery_messages,
                    temperature=CONTENT_GENERATION_TEMPERATURE,
                    max_tokens=delivery_max_tokens,
                )
                provider_calls_made += 1
                records = model_gateway.records()
                model_record = records[-1] if records else {}
                try:
                    if model_record.get("truncated"):
                        raise ValueError(
                            "model response was truncated at the output-token limit"
                        )
                    items = _extract_json_array(response)
                    items = _normalize_generated_batch_items(items, delivery_pages)
                    _store_generation_batch_cache(
                        delivery_cache_path,
                        delivery_cache_signature,
                        items,
                        model_record=model_record,
                    )
                    logger.info(
                        "Checkpointed concept-generation batch %s for %s (%s units)",
                        delivery_label,
                        chapter.slug,
                        len(items),
                    )
                    generated.extend(items)
                except Exception as exc:  # noqa: BLE001 - one bad batch must not fail the chapter
                    failed_batches += 1
                    logger.warning(
                        "Concept generation batch %s for %s yielded no parseable JSON: %s",
                        delivery_label,
                        chapter.slug,
                        exc,
                    )

    if failed_batches:
        raise ValueError(
            "Concept generation is incomplete: "
            f"{failed_batches} request batch(es) returned no usable JSON. "
            "No partial chapter was imported; retry the chapter."
        )
    compacted = consolidate_concept_payloads(
        generated,
        chapter_key=chapter.slug or chapter.chapter_name or str(chapter.id),
        page_count=len([page for page in pages if (page.text or "").strip()]),
    )
    return import_concepts_for_chapter(db, chapter.id, compacted, replace=replace)


def _next_version(value: str) -> str:
    match = re.fullmatch(r"v(\d+)", str(value or "").strip().lower())
    if match:
        return f"v{int(match.group(1)) + 1}"
    return "v2"


def _build_chapter_report(db: Session, chapter: ContentChapter) -> Dict[str, Any]:
    """Recompute the coverage/validation report for a chapter from its current
    stored pages and concepts, so threshold/config changes are reflected without
    re-running generation."""
    pages = db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).order_by(ContentPage.page_number).all()
    concepts = db.query(ContentConcept).filter(ContentConcept.chapter_id == chapter.id).order_by(ContentConcept.concept_id).all()
    chunks = db.query(ContentChunk).filter(ContentChunk.chapter_id == chapter.id).all()
    return build_coverage_report(
        [{"page_number": page.page_number, "char_count": page.char_count, "extraction_quality": page.extraction_quality} for page in pages],
        concepts,
        chunks,
        chapter.validation_report.get("issues", []) if chapter.validation_report else [],
    )


def approve_chapter(db: Session, chapter_id: int, *, approved_by: str = "") -> ContentChapter:
    chapter = db.query(ContentChapter).filter(ContentChapter.id == chapter_id).one_or_none()
    if chapter is None:
        raise ValueError(f"Chapter not found: {chapter_id}")
    if not chapter.concept_count:
        raise ValueError("Chapter has no validated concepts. Import or generate concept JSON first.")
    # Re-evaluate against current data and the active coverage threshold (rather
    # than the snapshot stored at generation time) and persist it, so config
    # changes take effect without re-running generation.
    report = _build_chapter_report(db, chapter)
    chapter.validation_report = report
    chapter.coverage_score = report["coverage_score"]
    if not report.get("ready_for_approval"):
        raise ValueError("Chapter is not ready for approval. Review validation_report first.")
    # Approval is the moment content goes live for students. If the PDF was
    # re-ingested since the last approval, bump the version so reports and
    # traces can say which source revision answered a question.
    if chapter.published_source_hash and chapter.published_source_hash != chapter.source_hash:
        chapter.version = _next_version(chapter.version)
    chapter.published_source_hash = chapter.source_hash
    chapter.status = "approved"
    chapter.approved_by = approved_by or "admin"
    chapter.approved_at = datetime.utcnow()
    chapter.updated_at = datetime.utcnow()
    db.flush()
    return chapter


def publish_chapter(db: Session, chapter_id: int, *, published_by: str = "") -> ContentChapter:
    chapter = approve_chapter(db, chapter_id, approved_by=published_by)
    chapter.status = "published"
    chapter.published_at = datetime.utcnow()
    db.flush()
    return chapter


def embed_missing_chunks(db: Session, *, chapter_id: Optional[int] = None) -> Dict[str, Any]:
    """Backfill embeddings for chunks ingested before embeddings were configured."""
    query = db.query(ContentChunk)
    if chapter_id is not None:
        query = query.filter(ContentChunk.chapter_id == chapter_id)
    # SQLAlchemy JSON stores Python ``None`` as JSON text ``null`` on SQLite by
    # default, so ``IS NULL`` misses legacy rows. Filter the small chapter batch
    # after deserialisation to cover SQL NULL, JSON null, and empty vectors on
    # both SQLite staging and PostgreSQL production.
    rows = [row for row in query.order_by(ContentChunk.id).all() if not row.embedding]

    if not embeddings_service.embeddings_enabled():
        return {
            "enabled": False,
            "embedded": 0,
            "missing": len(rows),
            "message": "Embeddings are not configured. Set EMBEDDINGS_API_KEY (or OPENAI_API_KEY).",
        }
    if not rows:
        return {"enabled": True, "embedded": 0, "missing": 0, "model": embeddings_service.embedding_model()}

    vectors = embeddings_service.embed_texts([row.text or " " for row in rows])
    model_name = embeddings_service.embedding_model()
    for row, vector in zip(rows, vectors):
        row.embedding = vector
        metadata = dict(row.metadata_json or {})
        metadata["embedding_model"] = model_name
        row.metadata_json = metadata
    db.flush()
    return {"enabled": True, "embedded": len(rows), "missing": 0, "model": model_name}


def serialize_chapter(chapter: ContentChapter) -> Dict[str, Any]:
    return {
        "id": chapter.id,
        "board": chapter.board,
        "class_level": chapter.class_level,
        "subject": chapter.subject,
        "book_name": chapter.book_name,
        "chapter_number": chapter.chapter_number,
        "chapter_name": chapter.chapter_name,
        "slug": chapter.slug,
        "pdf_path": chapter.pdf_path,
        "status": chapter.status,
        "version": chapter.version,
        "published_source_hash": chapter.published_source_hash or "",
        "page_count": chapter.page_count,
        "extracted_page_count": chapter.extracted_page_count,
        "chunk_count": chapter.chunk_count,
        "concept_count": chapter.concept_count,
        "coverage_score": chapter.coverage_score,
        "extraction_quality": chapter.extraction_quality,
        "validation_report": chapter.validation_report or {},
        "approved_by": chapter.approved_by,
        "approved_at": chapter.approved_at.isoformat() if chapter.approved_at else None,
        "published_at": chapter.published_at.isoformat() if chapter.published_at else None,
        "updated_at": chapter.updated_at.isoformat() if chapter.updated_at else None,
    }


def serialize_job(job: ContentIngestionJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "source_path": job.source_path,
        "summary": job.summary or {},
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def list_chapters(db: Session, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(ContentChapter).order_by(
        ContentChapter.class_level,
        ContentChapter.subject,
        ContentChapter.chapter_number,
        ContentChapter.chapter_name,
    )
    if status:
        query = query.filter(ContentChapter.status == status)
    return [serialize_chapter(chapter) for chapter in query.all()]


def chapter_report(db: Session, chapter_id: int) -> Dict[str, Any]:
    chapter = db.query(ContentChapter).filter(ContentChapter.id == chapter_id).one_or_none()
    if chapter is None:
        raise ValueError(f"Chapter not found: {chapter_id}")
    pages = db.query(ContentPage).filter(ContentPage.chapter_id == chapter.id).order_by(ContentPage.page_number).all()
    concepts = db.query(ContentConcept).filter(ContentConcept.chapter_id == chapter.id).order_by(ContentConcept.concept_id).all()
    report = _build_chapter_report(db, chapter)
    return {
        "chapter": serialize_chapter(chapter),
        "report": report,
        "page_preview": [
            {"page_number": page.page_number, "char_count": page.char_count, "quality": page.extraction_quality}
            for page in pages[:20]
        ],
        "concept_preview": [
            {"concept_id": concept.concept_id, "title": concept.title, "source_pages": concept.source_pages}
            for concept in concepts[:30]
        ],
    }


def _chapter_scope_matches(chapter: ContentChapter, scope: Optional[Dict[str, Any]], section_id: str) -> bool:
    """Hard scope filters: subject and chapter must match when supplied."""
    if not scope:
        return True
    subject = normalize_key(scope.get("subject"))
    chapter_value = normalize_key(scope.get("chapter"))
    chapter_slug = normalize_key(scope.get("chapter_slug"))
    class_level = normalize_key(scope.get("class_level")).removeprefix("class_")
    content_version = str(scope.get("content_version") or "").strip()
    if subject and subject != normalize_key(chapter.subject):
        return False
    chapter_haystack = normalize_key(
        f"{chapter.chapter_name} {chapter.slug} chapter_{chapter.chapter_number or ''}"
    )
    if chapter_value and chapter_value not in chapter_haystack:
        return False
    if chapter_slug and chapter_slug != normalize_key(chapter.slug):
        return False
    if class_level and class_level != normalize_key(chapter.class_level).removeprefix("class_"):
        return False
    if content_version and content_version != str(chapter.version or "").strip():
        return False
    return True


def _chapter_matches_topic(chapter: ContentChapter, scope: Optional[Dict[str, Any]], section_id: str) -> bool:
    """Soft topic filter: True when the scope topic names this chapter.

    Topics are often finer-grained than chapter names (e.g. topic "alkanes"
    inside chapter "Hydrocarbons"), so callers must treat a miss as
    "no preference", not exclusion — see search_approved_content.
    """
    topic = normalize_key((scope or {}).get("topic") or (scope or {}).get("section_id") or section_id)
    if not topic or topic in {"general", "open", "any", "all"}:
        return False
    haystack = normalize_key(f"{chapter.chapter_name} {chapter.slug} {chapter.subject}")
    return topic in haystack


def _score_text(text: str, terms: Sequence[str]) -> int:
    normalized = str(text or "").lower()
    return sum(normalized.count(term) for term in terms)


def _min_semantic_similarity() -> float:
    try:
        return float(os.getenv("EMBEDDINGS_MIN_SIMILARITY", "0.25"))
    except ValueError:
        return 0.25


_RRF_K = 60.0


def search_approved_content(
    section_id: str,
    question: str,
    *,
    scope: Optional[Dict[str, Any]] = None,
    max_chars: int = 5000,
    limit: int = 6,
) -> Dict[str, Any]:
    """Hybrid retrieval over approved content.

    Candidates are ranked lexically (term frequency, as before) and — when an
    embedding provider is configured and chunks carry embeddings — semantically
    by cosine similarity to the question. The two rankings are fused with
    reciprocal rank fusion, so semantically-phrased questions match material
    they share no keywords with, while exact-term matches keep their edge.
    Without embeddings the behavior is identical to the old lexical search.
    """
    terms = content_terms(
        f"{section_id} {(scope or {}).get('topic') or ''} {question}"
    )
    if not terms:
        terms = content_terms(section_id)
    if not terms:
        return {"context": "", "source": "content_pipeline", "paragraphs_found": 0}

    # Embed the query before opening a pooled DB connection so the external
    # embedding HTTP round trip never pins a connection for its full duration.
    query_vector = embeddings_service.embed_query(question or section_id)
    min_similarity = _min_semantic_similarity()

    db = SessionLocal()
    try:
        chapters = db.query(ContentChapter).filter(ContentChapter.status.in_(APPROVED_STATUSES)).all()
        chapters = [chapter for chapter in chapters if _chapter_scope_matches(chapter, scope, section_id)]
        # Topic narrows to matching chapters when it names one; otherwise the
        # topic is finer-grained than chapter names and ranking handles it.
        topic_matched = [chapter for chapter in chapters if _chapter_matches_topic(chapter, scope, section_id)]
        if topic_matched:
            chapters = topic_matched
        if not chapters:
            return {"context": "", "source": "content_pipeline", "paragraphs_found": 0}
        chapter_ids = [chapter.id for chapter in chapters]
        concept_rows = db.query(ContentConcept).filter(ContentConcept.chapter_id.in_(chapter_ids)).all()
        chunk_rows = db.query(ContentChunk).filter(ContentChunk.chapter_id.in_(chapter_ids)).all()
        chapter_by_id = {chapter.id: chapter for chapter in chapters}

        candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}
        raw_concept_ids = (scope or {}).get("concept_ids") or []
        if not isinstance(raw_concept_ids, (list, tuple, set)):
            raw_concept_ids = [raw_concept_ids]
        requested_concept_ids = [
            normalize_key(value)
            for value in raw_concept_ids
            if normalize_key(value)
        ]
        member_rank = {
            concept_id: index
            for index, concept_id in enumerate(requested_concept_ids)
        }
        effective_limit = max(int(limit or 0), len(member_rank), 1)
        allowed_source_pages = {
            page
            for concept in concept_rows
            if normalize_key(concept.concept_id) in member_rank
            for page in coerce_page_numbers(concept.source_pages)
        }
        requested_topic_keys = {
            normalize_key(value)
            for value in (
                section_id,
                (scope or {}).get("section_id"),
                (scope or {}).get("topic"),
            )
            if normalize_key(value)
        }
        for concept in concept_rows:
            if member_rank and normalize_key(concept.concept_id) not in member_rank:
                continue
            text = "\n".join(
                [
                    concept.title or "",
                    concept.definition or "",
                    concept.core_explanation or "",
                    " ".join(concept.key_points or []),
                    " ".join(map(str, concept.examples or [])),
                    " ".join(map(str, concept.formulas or [])),
                ]
            )
            lexical = _score_text(text, terms)
            normalized_concept_id = normalize_key(concept.concept_id)
            exact_topic_match = bool(
                normalized_concept_id in member_rank
                or requested_topic_keys.intersection(
                    {normalize_key(concept.concept_id), normalize_key(concept.title)}
                )
            )
            if lexical or exact_topic_match:
                candidates[("concept", concept.id)] = {
                    # A catalog ID/title match is authoritative and must rank
                    # ahead of merely similar text from the same chapter. All
                    # scoped members receive the same exact-match boost so the
                    # student's question, not member order, decides their rank.
                    "lexical": lexical + 3 + (1000 if exact_topic_match else 0),
                    "semantic": 0.0,
                    "exact_topic_match": exact_topic_match,
                    "type": "concept",
                    "payload": {
                        "chapter": chapter_by_id.get(concept.chapter_id),
                        "title": concept.title,
                        "text": text,
                        "pages": concept.source_pages or [],
                        "section_id": concept.concept_id,
                    },
                }
        warned_embedding_contract = False
        compatible_vector_count = 0
        for chunk in chunk_rows:
            if member_rank and allowed_source_pages:
                try:
                    page_start = int(chunk.page_start or chunk.page_end or 0)
                    page_end = int(chunk.page_end or chunk.page_start or 0)
                except (TypeError, ValueError):
                    page_start = page_end = 0
                if page_start > page_end:
                    page_start, page_end = page_end, page_start
                if not page_start or not any(
                    page_start <= page <= page_end
                    for page in allowed_source_pages
                ):
                    continue
            chunk_terms = set(chunk.lexical_terms or [])
            lexical = len(chunk_terms.intersection(terms)) * 2 + _score_text(chunk.text or "", terms)
            semantic = 0.0
            if query_vector is not None and chunk.embedding:
                embedding_metadata = dict(chunk.metadata_json or {})
                stored_model = str(embedding_metadata.get("embedding_model") or "")
                stored_endpoint_host = str(
                    embedding_metadata.get("embedding_endpoint_host") or ""
                )
                if embeddings_service.compatible_with_stored(
                    query_vector,
                    chunk.embedding,
                    stored_model=stored_model,
                    stored_endpoint_host=stored_endpoint_host,
                ):
                    compatible_vector_count += 1
                    semantic = embeddings_service.similarity(query_vector, chunk.embedding)
                    if semantic < min_similarity:
                        semantic = 0.0
                elif not warned_embedding_contract:
                    logger.warning(
                        "Semantic retrieval skipped incompatible stored vectors "
                        "(configured_model=%s configured_endpoint=%s stored_model=%s "
                        "stored_endpoint=%s query_dim=%d stored_dim=%d).",
                        embeddings_service.embedding_model(),
                        embeddings_service.embedding_endpoint_host() or "unknown",
                        stored_model or "unknown",
                        stored_endpoint_host or "unknown",
                        len(query_vector),
                        len(chunk.embedding),
                    )
                    warned_embedding_contract = True
            if lexical or semantic:
                candidates[("chunk", chunk.id)] = {
                    "lexical": lexical,
                    "semantic": semantic,
                    "exact_topic_match": False,
                    "type": "chunk",
                    "payload": {
                        "chapter": chapter_by_id.get(chunk.chapter_id),
                        "title": chunk.section_title or f"Pages {chunk.page_start}-{chunk.page_end}",
                        "text": chunk.text,
                        "pages": [page for page in (chunk.page_start, chunk.page_end) if page],
                        "section_id": chunk.chunk_id,
                    },
                }

        lexical_ranking = [
            key
            for key, candidate in sorted(
                candidates.items(), key=lambda item: (-item[1]["lexical"], item[0])
            )
            if candidate["lexical"] > 0
        ]
        semantic_ranking = [
            key
            for key, candidate in sorted(
                candidates.items(), key=lambda item: (-item[1]["semantic"], item[0])
            )
            if candidate["semantic"] > 0
        ]

        fused: Dict[Tuple[str, int], float] = {}
        for ranking in (lexical_ranking, semantic_ranking):
            for rank, key in enumerate(ranking, start=1):
                fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        ordered_keys = sorted(
            fused,
            key=lambda key: (
                not bool(candidates[key].get("exact_topic_match")),
                -fused[key],
                key,
            ),
        )

        blocks: List[str] = []
        used_pages: List[int] = []
        used_sections: List[str] = []
        total_chars = 0
        for key in ordered_keys[: effective_limit * 2]:
            candidate = candidates[key]
            payload = candidate["payload"]
            chapter = payload["chapter"]
            if chapter is None:
                continue
            page_label = ", ".join(str(page) for page in sorted(set(payload["pages"]))) or "unknown"
            header = (
                f"## {payload['title']}\n"
                f"Source: {chapter.board} Class {chapter.class_level} {chapter.subject}, "
                f"{chapter.chapter_name}, page(s): {page_label}, type: {candidate['type']}\n"
            )
            block = f"{header}{payload['text']}".strip()
            if total_chars + len(block) > max_chars:
                # A detailed concept can legitimately exceed the revision
                # budget. Preserve a bounded prefix of the best first match
                # instead of dropping the correct concept and reporting that
                # published material does not exist.
                if blocks:
                    continue
                block = block[:max_chars].rstrip()
                if not block:
                    continue
            blocks.append(block)
            used_pages.extend(int(page) for page in payload["pages"] if page)
            used_sections.append(str(payload["section_id"]))
            total_chars += len(block)
            if len(blocks) >= effective_limit:
                break

        if not blocks:
            return {"context": "", "source": "content_pipeline", "paragraphs_found": 0}
        return {
            "context": "\n\n".join(blocks),
            "section_id": section_id,
            "paragraphs_found": len(blocks),
            "keywords_used": terms,
            "basics_context": "",
            "source": "approved_content_pipeline",
            "source_pages": sorted(set(used_pages)),
            "matched_sections": used_sections,
            "retrieval_mode": (
                "hybrid" if query_vector is not None and compatible_vector_count else "lexical"
            ),
            "semantic_matches": len(semantic_ranking),
        }
    finally:
        db.close()
