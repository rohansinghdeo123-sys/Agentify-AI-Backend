"""Validated, time-aware NCERT teaching lessons; no model call on page navigation.

Authored pages are shared content, while proficiency/style/session adaptations
are computed per learner. Core science is never removed for a shorter session.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "ncert_teacher_v1"
SOURCE = "NCERT · teacher explanation"
LEVELS = {"essential", "foundation", "extension"}
KINDS = {"explanation", "definition", "example", "formula", "worked_example", "misconception", "connection", "practice", "takeaway", "clarification"}


@lru_cache(maxsize=1)
def teacher_curricula():
    """Source-bound snapshots retain the published Planning IDs and estimates."""
    from Logic.planning.curriculum_registry import resolve_planning_curriculum
    first = resolve_planning_curriculum(chapter_ref="some_basic_concepts_of_chemistry", subject="Chemistry", class_level="Class 11")
    others = json.loads((ROOT / "data/notes/teacher_chemistry/curricula.json").read_text(encoding="utf-8"))
    return (first, *others)


def resolve_teacher_curriculum(chapter_ref):
    import re
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    requested = normalize(chapter_ref)
    if not requested:
        return None
    matches = [c for c in teacher_curricula() if requested in {
        normalize(c["chapter_slug"]), normalize(c["chapter_title"]),
        *(normalize(a) for a in c.get("aliases", [])),
    }]
    if len(matches) == 1:
        return matches[0]
    # Registered manifests also accept the archived full publication slug.
    from Logic.planning.curriculum_registry import resolve_planning_curriculum
    return resolve_planning_curriculum(chapter_ref=chapter_ref, subject="Chemistry", class_level="Class 11") if not matches else None


def resolve_teacher_unit(chapter, topic, preferred_unit=None):
    curriculum = resolve_teacher_curriculum(chapter)
    if not curriculum:
        return None
    if curriculum["chapter_number"] == 1:
        # Keep the audited archived Chapter 1 URLs without applying its alias
        # map to any other chapter.
        from services.master_notes_service import resolve_unit
        return resolve_unit(chapter, topic, preferred_unit)
    units = curriculum["units"]
    if not topic:
        return curriculum, next((u for u in units if u["id"] == preferred_unit), units[0])
    matches = [u for u in units if topic in {
        u["id"], u["primary_topic_id"], *u.get("legacy_topic_ids", []),
        *(s["id"] for s in u["ncert_subtopics"]), *(c["id"] for c in u["concepts"]),
    }]
    return (curriculum, matches[0]) if len(matches) == 1 else None


def uses_teacher_notes(scope):
    """Published Revision retains its contract; notes Ask AI opts in explicitly."""
    return (str(scope.get("subject") or "").lower() in {"", "chemistry"}
            and str(scope.get("class_level") or "").lower() in {"", "11", "class 11", "xi", "class xi"}
            and (scope.get("catalog_source") == "planning_manifest"
                 or (scope.get("catalog_source") == "published" and scope.get("notes_content_version") == VERSION)))


@lru_cache(maxsize=256)
def _load_lesson(unit_id: str):
    for curriculum in teacher_curricula():
        for index, unit in enumerate(curriculum["units"]):
            if unit["id"] != unit_id:
                continue
            directory = curriculum.get("lesson_directory", "data/notes/teacher_chapter1")
            lesson = json.loads((ROOT / directory / f"unit_{index}.json").read_text(encoding="utf-8"))
            validate_lesson(lesson, unit)
            return lesson
    raise ValueError("No verified teaching lesson for this Planning unit")


def validate_lesson(lesson, unit):
    if lesson.get("unit_id") != unit["id"]:
        raise ValueError("Teaching lesson does not match its Planning unit")
    if "\ufffd" in lesson.get("title", ""):
        raise ValueError("Teaching topic title has damaged source encoding")
    pages = lesson.get("pages", [])
    if [p.get("page_number") for p in pages] != list(range(1, 11)):
        raise ValueError("Teaching lessons must contain exactly ten ordered pages")
    allowed = {s["id"] for s in unit["ncert_subtopics"]}
    covered, identities = set(), set()
    for page in pages:
        sections = set(page.get("source_section_ids", []))
        if not sections or not sections <= allowed:
            raise ValueError("Teaching page crosses the assigned NCERT topic")
        covered.update(sections)
        if not page.get("title", "").strip() or "\ufffd" in page["title"] or not page.get("id") or page["id"] in identities:
            raise ValueError("Teaching pages need unique identities and meaningful headings")
        identities.add(page["id"])
        blocks = page.get("blocks", [])
        if sum(b.get("level") == "essential" for b in blocks) < 2:
            raise ValueError("Each page must remain meaningful for all proficiency levels")
        for block in blocks:
            if block.get("kind") not in KINDS or block.get("level") not in LEVELS or len(block.get("text", "").strip()) < 15 or "\ufffd" in block["text"]:
                raise ValueError("Invalid teaching block")
        if page["page_number"] < 10 and not page.get("transition", "").strip():
            raise ValueError("Teaching pages need a connected next step")
    if covered != allowed:
        raise ValueError("An essential mapped NCERT subtopic is missing")
    questions = lesson.get("questions", [])
    if not 5 <= len(questions) <= 10:
        raise ValueError("Final revision needs five to ten understanding questions")
    if any(not q.get("question") or not q.get("answer") or not q.get("source_section_ids") or not set(q.get("source_section_ids", [])) <= allowed for q in questions):
        raise ValueError("Practice questions need scoped answer explanations")
    if any("\ufffd" in q["question"] + q["answer"] for q in questions):
        raise ValueError("Practice questions have damaged source encoding")


def _source_pages(unit, section_ids):
    # Subtopics may reference a parent source section (Chapter 2), rather than
    # sharing its ID (Chapter 1 and published concept snapshots).
    parent_ids = {s["section_id"] for s in unit["ncert_subtopics"] if s["id"] in section_ids}
    pages = {page for s in unit["ncert_sections"] if s["id"] in parent_ids
             for page in range(s["pages"]["start"], s["pages"]["end"] + 1)}
    if unit.get("source_segments"):
        pages.intersection_update(segment["page"] for segment in unit["source_segments"])
    return sorted(pages)


@lru_cache(maxsize=16)
def _verify_source(path, digest):
    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest().lower() != digest.lower():
        raise ValueError("The NCERT source changed; review the authored teaching lessons")
    return True


def compile_lesson(curriculum, unit, *, proficiency="new_to_it", style="balanced"):
    _verify_source(curriculum["source"]["path"], curriculum["source"]["sha256"])
    lesson = _load_lesson(unit["id"])
    # Configured time remains authoritative. Difficulty and depth are authored
    # into each topic; these gates select support, never a target word count.
    levels = {"essential"}
    if proficiency in {"new_to_it", "know_a_little"} or style == "detailed":
        levels.add("foundation")
    if style != "concise" and (unit["estimated_minutes"]["max"] >= 60 or
                               (proficiency == "mostly_confident" and unit["estimated_minutes"]["max"] >= 40) or
                               (style == "detailed" and unit["estimated_minutes"]["max"] >= 45)):
        levels.add("extension")
    pages = []
    for authored in lesson["pages"]:
        page = deepcopy(authored)
        page["blocks"] = [b for b in page["blocks"] if b["level"] in levels]
        page["source"] = SOURCE
        page["source_pages"] = _source_pages(unit, page["source_section_ids"])
        page["subtopic_ids"] = page["source_section_ids"]
        page["teaching_format"] = "guided_lesson"
        if page["page_number"] == 10:
            page["questions"] = deepcopy(lesson["questions"])
        # The same natural teaching text grounds contextual Ask AI.
        page["text"] = "\n\n".join(b["text"] for b in page["blocks"])
        if page.get("transition"):
            page["text"] += "\n\n" + page["transition"]
        if page.get("questions"):
            page["text"] += "\n\nCheck your understanding\n" + "\n".join(
                f"{i+1}. {q['question']}" for i, q in enumerate(page["questions"]))
        pages.append(page)
    return pages


def session_portion(pages, unit, *, start_page=0, session_minutes=None):
    """Divide the configured topic workload; never re-estimate its duration."""
    if session_minutes is None:
        return None
    if session_minutes <= 0:
        return None
    start_page = max(0, min(9, int(start_page)))
    # Worked solutions and final practice carry more of the configured workload
    # than an introductory observation, independent of paragraph word count.
    weights = [1 + sum(b["kind"] == "worked_example" for b in p["blocks"]) +
               (len(p.get("questions", [])) / 3) for p in pages]
    workload = unit["estimated_minutes"]["min"]
    remaining = session_minutes
    end_page = start_page
    for i in range(start_page, 10):
        share = workload * weights[i] / sum(weights)
        if i > start_page and share > remaining + 1e-9:
            break
        end_page = i
        remaining -= share
        if remaining <= 0:
            break
    return {"start_page": start_page, "end_page": end_page,
            "allocated_minutes": session_minutes, "additional_practice_needed": end_page < 9 or remaining < -1e-9}


def selected_teacher_chapter(state, curriculum):
    """Only the learner's matching syllabus selection may supply adaptation."""
    if not state or str(state.subject).lower() != "chemistry" or str(state.class_level).lower() not in {"11", "class 11", "xi", "class xi"}:
        return None
    return next((s for s in state.selections_json or [] if (resolve_teacher_curriculum(
        s.get("chapter_ref")) or {}).get("curriculum_key") == curriculum["curriculum_key"]), None)


def learner_settings(db, user_id, state, curriculum, unit):
    from models import AICoachProfile
    proficiency = "new_to_it"
    allocation = None
    style = "balanced"
    matching = selected_teacher_chapter(state, curriculum)
    if matching:
        proficiency = matching.get("chapter_proficiency", proficiency)
        # Reuse the existing analytics-based proficiency; reading alone
        # must not create new mastery evidence or advance the itinerary.
        from services.study_route_service import _build_portfolio_from_state
        portfolio = _build_portfolio_from_state(db, state)
        chapter = next((c for c in portfolio["chapters"] if c["curriculum_key"] == curriculum["curriculum_key"]), {})
        proficiency = chapter.get("effective_proficiency", {}).get("effective", proficiency)
        route = (state.runtime_json or {}).get("today_route", {})
        allocation = sum(i["minutes"] for i in route.get("items", [])
                         if i.get("unit_id") == unit["id"] and i.get("role") == "main_focus") or None
    profile = db.query(AICoachProfile).filter_by(user_id=user_id).first()
    if profile:
        prefs = profile.study_preferences or {}
        preferred = str(prefs.get("preferred_style") or prefs.get("answer_style") or prefs.get("feedback_depth") or "").lower()
        import re
        preferred = re.sub(r"[^a-z0-9]+", "_", preferred)
        if any(word in preferred for word in ["concise", "brief", "short"]):
            style = "concise"
        elif any(word in preferred for word in ["detail", "step_by_step", "deep"]):
            style = "detailed"
    return proficiency, style, allocation


def retrieve_teacher_context(chapter_ref, topic_id, *, question="", subtopic_ids=None, page_number=None, proficiency="new_to_it", style="balanced"):
    resolved = resolve_teacher_unit(chapter_ref, topic_id)
    if not resolved:
        return None
    curriculum, unit = resolved
    pages = compile_lesson(curriculum, unit, proficiency=proficiency, style=style)
    allowed = {s["id"] for s in unit["ncert_subtopics"]}
    requested = set(subtopic_ids or allowed)
    if not requested <= allowed:
        raise ValueError("Requested subtopics are outside the assigned NCERT topic")
    if page_number is not None:
        if isinstance(page_number, bool) or not isinstance(page_number, int) or not 1 <= page_number <= 10:
            raise ValueError("Invalid notes page number")
        pages = [p for p in pages if p["page_number"] == page_number]
    pages = [p for p in pages if set(p["source_section_ids"]) <= requested]
    # Keep complete page reasoning/formulas. Questions rank relevant teaching
    # steps; they do not introduce another topic or trigger an external source.
    import re
    terms = set(re.findall(r"[a-z]{3,}", question.lower()))
    ranked = sorted(pages, key=lambda p: -len(terms.intersection(re.findall(r"[a-z]{3,}", p["text"].lower()))))
    selected = ranked[:4] if question else pages
    selected.sort(key=lambda p: p["page_number"])
    return {"context": "\n\n".join(f"[NCERT sections {', '.join(p['source_section_ids'])}; PDF pages {p['source_pages']}; {p['title']}]\n{p['text']}" for p in selected),
            "source": "ncert_teacher_notes", "retrieval_mode": "scoped_ncert_teaching_pages",
            "unit_id": unit["id"], "section_id": unit["primary_topic_id"],
            "paragraphs_found": len(selected), "keywords_used": sorted(terms),
            "source_pages": sorted({n for p in selected for n in p["source_pages"]}),
            "matched_sections": [p["id"] for p in selected], "content_order_locked": True,
            "provenance": [{"source":"NCERT", "section_id":p["id"], "subtopic_ids":p["source_section_ids"], "pages":p["source_pages"]} for p in selected]}
