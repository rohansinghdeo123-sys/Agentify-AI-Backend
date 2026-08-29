"""End-to-end content automation: download NCERT PDFs and run the full pipeline.

For each (class, subject) in scope this:
  1. politely downloads NCERT chapter PDFs (rate-limited, resumable, validated),
  2. ingests each PDF (pages + chunks),
  3. generates structured concepts via the content agents,
  4. embeds chunks for semantic retrieval,
  5. auto-publishes ONLY chapters that clear the existing quality gate
     (``publish_chapter`` raises if coverage/validation gates fail), otherwise
     leaves them as ``needs_review`` for a human to inspect in the admin report.

Designed to run as a CLI (``scripts/automate_content.py``) against the live DB.
Every chapter is isolated: one failure never stops the whole run.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from database import SessionLocal
from Logic.content_pipeline import (
    RAW_NCERT_DIR,
    embed_missing_chunks,
    file_sha256,
    generate_concepts_for_chapter,
    infer_metadata_from_pdf_path,
    ingest_pdf_file,
    normalize_key,
    publish_chapter,
    serialize_chapter,
    titleize,
)
from models import ContentChapter, ContentChunk, ContentPage

logger = logging.getLogger("ai_educator.content_automation")

NCERT_PDF_BASE = "https://ncert.nic.in/textbook/pdf"
USER_AGENT = (
    "AgentifyAI-EducationalContentBot/1.0 "
    "(NCERT study-material ingestion for an education app; +https://agentifyai.in)"
)

# NCERT book codes per (class_level, subject). Each subject may span parts; the
# orchestrator numbers chapters continuously across parts. Config-driven so the
# scope can be extended without code changes.
NCERT_BOOKS: Dict[Tuple[str, str], List[str]] = {
    ("11", "Physics"): ["keph1", "keph2"],
    ("11", "Chemistry"): ["kech1", "kech2"],
    ("11", "Maths"): ["kemh1"],
    ("12", "Physics"): ["leph1", "leph2"],
    ("12", "Chemistry"): ["lech1", "lech2"],
    ("12", "Maths"): ["lemh1", "lemh2"],
}

DEFAULT_CLASSES = ["11", "12"]
DEFAULT_SUBJECTS = ["Physics", "Chemistry", "Maths"]
MIN_PDF_BYTES = 10_000


@dataclass(frozen=True)
class NCERTChapterSource:
    """Stable mapping from one curriculum chapter to its NCERT source PDF.

    NCERT book-part chapter numbers restart at one, while AgentifyAI chapter
    numbers continue across parts. Keeping both numbers and the canonical
    filename together prevents a rediscovery run from renumbering content or
    producing opaque duplicate files.
    """

    chapter_number: int
    title: str
    filename_slug: str
    book_code: str
    book_chapter_number: int

    @property
    def filename(self) -> str:
        return f"chapter_{self.chapter_number:02d}_{self.filename_slug}.pdf"

    @property
    def source_url(self) -> str:
        return f"{NCERT_PDF_BASE}/{self.book_code}{self.book_chapter_number:02d}.pdf"


# Current NCERT Class XI Chemistry curriculum. This is deliberately structured
# data rather than downloader conditionals: future curricula can be added by
# supplying the same chapter-to-book mapping without changing the algorithm.
NCERT_CURRICULUM: Dict[Tuple[str, str], Tuple[NCERTChapterSource, ...]] = {
    ("11", "Chemistry"): (
        NCERTChapterSource(1, "Some Basic Concepts of Chemistry", "some_basic_concepts_of_chemistry", "kech1", 1),
        NCERTChapterSource(2, "Structure of Atom", "structure_of_atom", "kech1", 2),
        NCERTChapterSource(
            3,
            "Classification of Elements and Periodicity in Properties",
            "classification_of_elements_and_periodicity_in_properties",
            "kech1",
            3,
        ),
        NCERTChapterSource(
            4,
            "Chemical Bonding and Molecular Structure",
            "chemical_bonding_and_molecular_structure",
            "kech1",
            4,
        ),
        NCERTChapterSource(5, "Thermodynamics", "thermodynamics", "kech1", 5),
        NCERTChapterSource(6, "Equilibrium", "equilibrium", "kech1", 6),
        NCERTChapterSource(7, "Redox Reactions", "redox_reactions", "kech2", 1),
        NCERTChapterSource(
            8,
            "Organic Chemistry - Some Basic Principles and Techniques",
            "organic_chemistry_some_basic_principles_and_techniques",
            "kech2",
            2,
        ),
        NCERTChapterSource(9, "Hydrocarbons", "hydrocarbons", "kech2", 3),
    ),
}


# ---------------------------------------------------------------------------
# Polite downloading
# ---------------------------------------------------------------------------
def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def robots_allows(session: requests.Session) -> bool:
    """Respect robots.txt for the textbook PDF path. Empty/missing robots = allow."""
    try:
        resp = session.get("https://ncert.nic.in/robots.txt", timeout=20)
        if resp.status_code != 200 or not resp.text.strip():
            return True
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser.can_fetch(USER_AGENT, f"{NCERT_PDF_BASE}/test.pdf")
    except Exception:
        return True


def _is_valid_pdf(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_PDF_BYTES:
            return False
        with path.open("rb") as handle:
            return handle.read(5).startswith(b"%PDF")
    except Exception:
        return False


def _download_pdf(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single PDF to ``dest``; returns True only on a valid PDF."""
    try:
        resp = session.get(url, timeout=90, stream=True)
        if resp.status_code != 200:
            return False
        tmp = dest.with_suffix(".part")
        size = 0
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)
                    size += len(chunk)
        if not _is_valid_pdf(tmp):
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        logger.info("downloaded %s (%.1f MB)", dest.name, size / (1024 * 1024))
        return True
    except Exception as exc:  # noqa: BLE001 - one bad download must not stop the run
        logger.warning("download failed %s: %s", url, exc)
        return False


def _remote_pdf_exists(session: requests.Session, code: str, nn: int) -> bool:
    """HEAD-probe a chapter URL (no download) to discover if it exists."""
    try:
        resp = session.head(f"{NCERT_PDF_BASE}/{code}{nn:02d}.pdf", timeout=30, allow_redirects=True)
        return resp.status_code == 200 and "pdf" in resp.headers.get("content-type", "").lower()
    except Exception:
        return False


def _existing_chapter_pdf(
    subject_dir: Path,
    chapter_number: int,
    *,
    preferred_filename: Optional[str] = None,
    expected_title: str = "",
) -> Optional[Path]:
    """Return a valid existing ``chapter_NN*.pdf`` without creating a copy.

    Earlier manual runs saved semantic names such as
    ``chapter_01_some_basic_concepts_of_chemistry.pdf``. Looking only for
    ``chapter_01.pdf`` caused those chapters to be downloaded a second time.
    Prefer the canonical semantic name, then any descriptive file, and finally
    the old generic name so every supported historical layout remains resumable.
    """
    generic_name = f"chapter_{chapter_number:02d}.pdf"
    candidates = list(subject_dir.glob(f"chapter_{chapter_number:02d}*.pdf"))
    candidates.sort(
        key=lambda path: (
            0 if preferred_filename and path.name == preferred_filename else
            2 if path.name == generic_name else
            1,
            path.name.lower(),
        )
    )
    for path in candidates:
        if not _is_valid_pdf(path):
            continue
        if expected_title and not _pdf_matches_curriculum_title(path, expected_title):
            logger.warning(
                "existing PDF title does not match curriculum; ignoring %s (expected %s)",
                path.name,
                expected_title,
            )
            continue
        return path
    return None


def _pdf_matches_curriculum_title(path: Path, expected_title: str) -> bool:
    """Confirm a reusable PDF identifies the expected curriculum chapter.

    A numeric filename plus a PDF header is not enough: an accidentally renamed
    chapter would otherwise be ingested and then labelled with the canonical
    title. NCERT chapter titles occur on the opening page; reading at most two
    pages keeps resume checks cheap while tolerating cover-page variations.
    """
    expected = normalize_key(expected_title)
    if not expected:
        return True
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        opening_text = "\n".join(
            (page.extract_text() or "")
            for page in list(reader.pages[:2])
        )
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF is never safe to reuse
        logger.warning("could not validate existing PDF title %s: %s", path.name, exc)
        return False
    return expected in normalize_key(opening_text)


def _download_known_curriculum(
    session: requests.Session,
    subject_dir: Path,
    curriculum: Sequence[NCERTChapterSource],
    *,
    book_codes: Sequence[str],
    delay_seconds: float,
    max_chapters: int,
    skip_existing: bool,
) -> List[Path]:
    """Download a structured curriculum using canonical semantic filenames."""
    allowed_codes = set(book_codes)
    required_codes = {chapter.book_code for chapter in curriculum}
    if not required_codes.issubset(allowed_codes):
        missing_codes = ", ".join(sorted(required_codes - allowed_codes))
        raise RuntimeError(
            f"Incomplete NCERT source mapping: missing book code(s) {missing_codes}."
        )
    chapters = [
        chapter
        for chapter in curriculum
        if chapter.book_code in allowed_codes and chapter.chapter_number <= max_chapters
    ]
    if len(chapters) != len(curriculum):
        raise RuntimeError(
            f"Configured max_chapters={max_chapters} excludes part of the "
            f"{len(curriculum)}-chapter curriculum."
        )
    paths: List[Path] = []
    for chapter in chapters:
        if skip_existing:
            existing = _existing_chapter_pdf(
                subject_dir,
                chapter.chapter_number,
                preferred_filename=chapter.filename,
                expected_title=chapter.title,
            )
            if existing is not None:
                logger.info("skip (exists) %s", existing.name)
                paths.append(existing)
                continue

        dest = subject_dir / chapter.filename
        if _download_pdf(session, chapter.source_url, dest):
            if _pdf_matches_curriculum_title(dest, chapter.title):
                paths.append(dest)
            else:
                # A PDF header and plausible size do not prove that NCERT
                # returned the requested chapter. Never let a changed or
                # misrouted upstream URL enter the canonical curriculum slot.
                logger.error(
                    "downloaded PDF title does not match curriculum; rejecting %s "
                    "(expected %s)",
                    dest.name,
                    chapter.title,
                )
                dest.unlink(missing_ok=True)
        time.sleep(delay_seconds)
    if len(paths) != len(chapters):
        completed = {int(path.name.split("_")[1]) for path in paths}
        missing = [str(chapter.chapter_number) for chapter in chapters if chapter.chapter_number not in completed]
        raise RuntimeError(
            "NCERT curriculum download incomplete; missing chapter(s): "
            + ", ".join(missing)
            + ". No ingestion was started. Re-run to resume safely."
        )
    return paths


def download_subject(
    session: requests.Session,
    class_level: str,
    subject: str,
    book_codes: Sequence[str],
    *,
    dest_root: Path = RAW_NCERT_DIR,
    delay_seconds: float = 4.0,
    max_chapters: int = 30,
    skip_existing: bool = True,
) -> List[Path]:
    """Download all chapters for one subject, numbered continuously across parts.

    Subjects with structured curriculum metadata use canonical semantic
    filenames. Other configured subjects retain discovery-based downloading as
    a backwards-compatible fallback. Both paths reuse any valid existing
    ``chapter_NN*.pdf`` file, including older descriptive filenames."""
    subject_dir = dest_root / f"class_{class_level}" / subject.lower()
    subject_dir.mkdir(parents=True, exist_ok=True)

    curriculum = NCERT_CURRICULUM.get((class_level, subject))
    if curriculum:
        return _download_known_curriculum(
            session,
            subject_dir,
            curriculum,
            book_codes=book_codes,
            delay_seconds=delay_seconds,
            max_chapters=max_chapters,
            skip_existing=skip_existing,
        )

    # Phase 1: discover every real chapter across all book parts.
    discovered: List[Tuple[str, int]] = []
    for code in book_codes:
        misses = 0
        for nn in range(1, max_chapters + 1):
            if _remote_pdf_exists(session, code, nn):
                discovered.append((code, nn))
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    break
            time.sleep(min(delay_seconds, 1.5))

    # Phase 2: download to stable sequential filenames.
    paths: List[Path] = []
    for index, (code, nn) in enumerate(discovered, start=1):
        dest = subject_dir / f"chapter_{index:02d}.pdf"
        if skip_existing:
            existing = _existing_chapter_pdf(subject_dir, index)
            if existing is not None:
                logger.info("skip (exists) %s", existing.name)
                paths.append(existing)
                continue
        if _download_pdf(session, f"{NCERT_PDF_BASE}/{code}{nn:02d}.pdf", dest):
            paths.append(dest)
        time.sleep(delay_seconds)
    return paths


# ---------------------------------------------------------------------------
# Per-chapter pipeline
# ---------------------------------------------------------------------------
# Lines that begin a sentence/epigraph, not a title — NCERT chapters often open
# with a quotation, so a title must not look like prose.
_TITLE_SKIP_STARTERS = (
    "the ", "it ", "a ", "an ", "in ", "this ", "these ", "those ", "when ",
    "as ", "after ", "every", "chemical", "chemistry deals", "scientists",
)


def _infer_title(db, chapter: ContentChapter) -> str:
    """Best-effort chapter title from the first page.

    Prefer the line immediately after an NCERT ``UNIT N`` marker. This accepts
    legitimate long titles and NCERT's occasionally irregular extracted casing
    while retaining a conservative title-like fallback for other PDFs.
    """
    page = (
        db.query(ContentPage)
        .filter(ContentPage.chapter_id == chapter.id)
        .order_by(ContentPage.page_number)
        .first()
    )
    if not page or not page.text:
        return ""
    lines = [line.strip() for line in page.text.splitlines() if line.strip()][:20]
    for index, raw in enumerate(lines[:-1]):
        if re.fullmatch(r"(?i)unit\s+\d{1,3}", raw):
            candidate = lines[index + 1]
            words = candidate.split()
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in candidate) / max(len(candidate), 1)
            if (
                1 <= len(words) <= 14
                and len(candidate) <= 140
                and alpha_ratio > 0.75
                and "�" not in candidate
                and not candidate.rstrip().endswith((".", ",", ";", ":"))
            ):
                return titleize(candidate)

    for raw in lines[:12]:
        low = raw.lower()
        words = raw.split()
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in raw) / max(len(raw), 1)
        if (
            1 <= len(words) <= 12          # long chapter titles are legitimate
            and len(raw) <= 100
            and raw[0].isupper()           # titles are capitalized; rejects "able to"
            and "." not in raw             # rejects author lines like "Glenn T. Seaborg"
            and "�" not in raw        # rejects garbled-encoding lines
            and not raw.rstrip().endswith((",", ";", ":"))
            and not low.startswith(("chapter", "unit", "page", "ncert"))
            and not low.startswith(_TITLE_SKIP_STARTERS)
            and alpha_ratio > 0.9
        ):
            return titleize(raw)
    return ""


def _curriculum_title(chapter: ContentChapter) -> str:
    curriculum = NCERT_CURRICULUM.get(
        (
            str(getattr(chapter, "class_level", "") or ""),
            str(getattr(chapter, "subject", "") or ""),
        )
    )
    if not curriculum:
        return ""
    return next(
        (
            source.title
            for source in curriculum
            if source.chapter_number == getattr(chapter, "chapter_number", None)
        ),
        "",
    )


def _chapter_for_pdf(db, pdf_path: Path) -> Tuple[Optional[ContentChapter], Dict[str, Any], str]:
    """Resolve a PDF to one curriculum row across legacy/canonical filenames."""

    metadata = infer_metadata_from_pdf_path(pdf_path, RAW_NCERT_DIR)
    source_hash = file_sha256(pdf_path)
    chapter = (
        db.query(ContentChapter)
        .filter(ContentChapter.slug == metadata["slug"])
        .one_or_none()
    )
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
            chapter = identity_matches[0]
        elif len(identity_matches) > 1:
            raise ValueError(
                "Ambiguous curriculum identity: multiple stored chapter rows match "
                f"Class {metadata['class_level']} {metadata['subject']} "
                f"chapter {metadata['chapter_number']}."
            )
    return chapter, metadata, source_hash


def _existing_ingested_chapter(db, pdf_path: Path) -> Optional[ContentChapter]:
    """Return an already-ingested chapter for this exact PDF (matching source
    hash, chunks present) so a resume can skip re-ingest + re-embed and spend
    quota only on concept generation. None if not ingested or the PDF changed."""
    try:
        chapter, metadata, source_hash = _chapter_for_pdf(db, pdf_path)
        if not chapter or chapter.source_hash != source_hash:
            return None
        has_chunks = (
            db.query(ContentChunk.id)
            .filter(ContentChunk.chapter_id == chapter.id)
            .first()
            is not None
        )
        if not has_chunks:
            return None
        # Canonicalise unpublished/resumed content without re-extracting or
        # re-embedding an unchanged PDF.  Published rows are intentionally left
        # stable so historical links keep resolving.
        if chapter.status not in ("approved", "published"):
            for key, value in metadata.items():
                setattr(chapter, key, value)
            chapter.pdf_path = str(pdf_path)
            chapter.source_hash = source_hash
            db.flush()
        return chapter
    except Exception:  # noqa: BLE001 - fall back to a normal ingest
        return None


def process_chapter(
    db,
    pdf_path: Path,
    *,
    auto_publish: bool = True,
    max_batch_chars: int = 9000,
    reuse_ingest: bool = True,
) -> Dict[str, Any]:
    """Ingest -> generate concepts -> embed -> (gated) publish one chapter.

    When ``reuse_ingest`` and the PDF is already ingested unchanged, the pages/
    chunks/embeddings are reused as-is (no re-extract, no re-embed) so a
    quota-interrupted resume only re-runs the concept-generation step.

    Replacements of already-live content are one database transaction. Readers
    keep seeing the last published version until extraction, generation,
    embeddings, and the publication gate have all succeeded. A failure rolls
    back the entire replacement while source-bound filesystem checkpoints still
    make the next attempt resumable.
    """
    existing_before, _metadata, _source_hash = _chapter_for_pdf(db, pdf_path)
    preserve_live_version = bool(
        existing_before and existing_before.status in ("approved", "published")
    )
    if preserve_live_version and not auto_publish:
        raise ValueError(
            "A live chapter cannot be re-ingested with --no-publish because the "
            "current schema has one live row per chapter. Use auto-publish for an "
            "atomic gated replacement; the previous version is preserved on failure."
        )

    try:
        chapter = _existing_ingested_chapter(db, pdf_path) if reuse_ingest else None
        if chapter is None:
            chapter = ingest_pdf_file(db, pdf_path, root_path=RAW_NCERT_DIR)
            if not preserve_live_version:
                db.commit()
                db.refresh(chapter)

        title = _curriculum_title(chapter) or _infer_title(db, chapter)
        if title and title.lower() not in chapter.chapter_name.lower():
            chapter.chapter_name = title
        # New/unpublished chapters commit safe milestones for resumability. A
        # live replacement deliberately defers every commit until publication.
        if not preserve_live_version:
            db.commit()
            db.refresh(chapter)

        generate_concepts_for_chapter(db, chapter.id, max_batch_chars=max_batch_chars)
        if not preserve_live_version:
            db.commit()
            db.refresh(chapter)

        embed_result = embed_missing_chunks(db, chapter_id=chapter.id)
        if not preserve_live_version:
            db.commit()
            db.refresh(chapter)

        final_status = chapter.status
        publish_error = ""
        if auto_publish:
            try:
                publish_chapter(db, chapter.id, published_by="automation")
                db.commit()
                db.refresh(chapter)
                final_status = "published"
            except ValueError as exc:
                if preserve_live_version:
                    db.rollback()
                    raise RuntimeError(
                        "Replacement did not pass the publication gate; the "
                        f"previous live chapter was preserved: {exc}"
                    ) from exc
                # A first-time chapter has no prior live version to protect.
                # Persist its recomputed report for review in the admin page.
                db.commit()
                db.refresh(chapter)
                final_status = chapter.status or "needs_review"
                publish_error = str(exc)
                logger.info("held for review: %s (%s)", chapter.slug, exc)

        return {
            "chapter": serialize_chapter(chapter),
            "status": final_status,
            "embeddings": embed_result,
            "publish_error": publish_error,
        }
    except Exception:
        if preserve_live_version:
            db.rollback()
        raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _already_completed(db, pdf_path: Path) -> bool:
    """True if this PDF is already fully published with matching content, so a
    resume can skip it without re-spending LLM/embedding quota. Only fully
    published/approved chapters are skipped; ingested-but-unprocessed chapters
    (no concepts yet) are re-processed so a quota-interrupted run can finish."""
    try:
        chapter, _metadata, source_hash = _chapter_for_pdf(db, pdf_path)
        if not chapter or chapter.status not in ("approved", "published"):
            return False
        if not (chapter.concept_count or 0):
            return False
        return chapter.published_source_hash == source_hash
    except Exception:  # noqa: BLE001 - never let a skip-check abort the run
        return False


def run_automation(
    *,
    classes: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[str]] = None,
    chapter_numbers: Optional[Sequence[int]] = None,
    delay_seconds: float = 4.0,
    max_chapters: int = 30,
    auto_publish: bool = True,
    download_only: bool = False,
    skip_existing: bool = True,
    skip_completed: bool = True,
    reuse_ingest: bool = True,
    dest_root: Path = RAW_NCERT_DIR,
    db_factory=SessionLocal,
) -> Dict[str, Any]:
    """Run the full automation for the configured scope. Returns a run summary."""
    classes = list(classes or DEFAULT_CLASSES)
    subjects = list(subjects or DEFAULT_SUBJECTS)
    selected_chapters = {
        int(chapter_number)
        for chapter_number in (chapter_numbers or [])
        if int(chapter_number) > 0
    }
    session = _make_session()
    if not robots_allows(session):
        raise RuntimeError("NCERT robots.txt disallows fetching textbook PDFs.")

    summary: Dict[str, Any] = {
        "classes": classes,
        "subjects": subjects,
        "selected_chapters": sorted(selected_chapters),
        "sources_ready": 0,
        "downloaded": 0,
        "reused_sources": 0,
        "published": 0,
        "needs_review": 0,
        "skipped": 0,
        "failed": [],
        "chapters": [],
    }

    for class_level in classes:
        for subject in subjects:
            book_codes = NCERT_BOOKS.get((class_level, subject))
            if not book_codes:
                logger.warning("no NCERT books configured for Class %s %s", class_level, subject)
                continue
            logger.info("=== Class %s %s (%s) ===", class_level, subject, ", ".join(book_codes))
            subject_dir = dest_root / f"class_{class_level}" / subject.lower()
            existing_sources = {
                path.resolve()
                for path in subject_dir.glob("chapter_*.pdf")
                if _is_valid_pdf(path)
            }
            pdf_paths = download_subject(
                session, class_level, subject, book_codes,
                dest_root=dest_root, delay_seconds=delay_seconds,
                max_chapters=max_chapters, skip_existing=skip_existing,
            )
            ready_count = len(pdf_paths)
            downloaded_count = (
                sum(path.resolve() not in existing_sources for path in pdf_paths)
                if skip_existing
                else ready_count
            )
            summary["sources_ready"] += ready_count
            summary["downloaded"] += downloaded_count
            summary["reused_sources"] += ready_count - downloaded_count
            if download_only:
                continue

            for pdf_path in pdf_paths:
                metadata = infer_metadata_from_pdf_path(pdf_path, dest_root)
                if (
                    selected_chapters
                    and int(metadata.get("chapter_number") or 0) not in selected_chapters
                ):
                    continue
                db = db_factory()
                try:
                    # A forced re-ingest must never be short-circuited merely
                    # because the previous source revision is already live.
                    if skip_completed and reuse_ingest and _already_completed(db, pdf_path):
                        summary["skipped"] += 1
                        logger.info("[SKIP] already published: Class %s %s %s",
                                    class_level, subject, pdf_path.name)
                        continue
                    result = process_chapter(
                        db,
                        pdf_path,
                        auto_publish=auto_publish,
                        reuse_ingest=reuse_ingest,
                    )
                    status = result["status"]
                    if status == "published":
                        summary["published"] += 1
                    else:
                        summary["needs_review"] += 1
                    summary["chapters"].append({
                        "class": class_level, "subject": subject,
                        "file": pdf_path.name, "status": status,
                        "slug": result["chapter"].get("slug"),
                        "coverage": result["chapter"].get("coverage_score"),
                        "concepts": result["chapter"].get("concept_count"),
                        "publish_error": result.get("publish_error", ""),
                    })
                    logger.info("[%s] Class %s %s %s -> %s",
                                status.upper(), class_level, subject, pdf_path.name, result["chapter"].get("slug"))
                except Exception as exc:  # noqa: BLE001 - isolate per-chapter failures
                    db.rollback()
                    logger.exception("FAILED Class %s %s %s", class_level, subject, pdf_path.name)
                    summary["failed"].append({"class": class_level, "subject": subject, "file": pdf_path.name, "error": str(exc)})
                finally:
                    db.close()

    logger.info(
        "Automation done: sources_ready=%d downloaded=%d reused=%d "
        "published=%d needs_review=%d skipped=%d failed=%d",
        summary["sources_ready"], summary["downloaded"], summary["reused_sources"],
        summary["published"], summary["needs_review"], summary["skipped"],
        len(summary["failed"]),
    )
    return summary
