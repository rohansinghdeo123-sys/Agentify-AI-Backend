"""Exam-intelligence agents.

Each LLM-backed agent returns a sanitized, stable dict and NEVER raises on bad
model output: if the model call fails or returns unusable JSON, a deterministic
fallback keeps the feature working (degraded but safe). Observed facts (the
question list) drive the aggregates, so stored analysis is always self-consistent.

Agents:
- ``parse_paper``                -> PaperParserAgent (deterministic)
- ``analyze_paper``              -> ExamPatternAnalyzerAgent (LLM)
- ``aggregate_analyses``         -> cross-paper pattern intelligence (deterministic)
- ``generate_probable_questions``-> ProbableQuestionAgent (LLM + fallback)
- ``generate_written_question``  -> written question setter (LLM + fallback)
- ``evaluate_written_answer``    -> WrittenAnswerEvaluatorAgent + TeacherFeedbackAgent (LLM + fallback)
- ``derive_weaknesses``          -> StudentWeaknessTrackerAgent (deterministic)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from Logic.coach.model_gateway import model_gateway

from . import json_utils as J
from . import prompts as P
from .parsers import ParsedPaper, parse_upload

logger = logging.getLogger("ai_educator.exam.agents")

# Keep analyzer input within a safe token budget; real papers are far shorter.
MAX_ANALYZER_CHARS = 24000
_SHORT_TYPES = {"very_short_answer", "short_answer", "mcq", "true_false", "fill_in_blank"}
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "what", "why", "how",
    "define", "explain", "state", "write", "give", "describe", "discuss", "name",
    "general", "open", "any", "all", "none", "subject", "chapter", "topic",
}


# ---------------------------------------------------------------------------
# PaperParserAgent
# ---------------------------------------------------------------------------
def parse_paper(data: bytes, filename: str = "", content_type: str = "") -> ParsedPaper:
    """Deterministically turn uploaded bytes into text + confidence + warnings."""
    return parse_upload(data, filename=filename, content_type=content_type)


# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------
def _complete_json(
    *,
    role: str,
    agent_name: str,
    task: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[str, Optional[str]]:
    try:
        raw = model_gateway.complete(
            role,
            messages,
            complexity="balanced",
            agent_name=agent_name,
            task=task,
            student_visible=False,
            safety_tier="strict_source_grounding",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(raw or ""), None
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the route
        logger.warning("Exam LLM call failed (%s): %s", task, exc)
        return "", str(exc)


def build_reference_context(
    section_id: str,
    query: str,
    *,
    subject: str = "",
    chapter: str = "",
    class_level: str = "",
    catalog_source: str = "",
    max_chars: int = 2200,
) -> str:
    """Best-effort grounding from approved syllabus content. Never raises."""
    try:
        from Logic.planning.curriculum_registry import resolve_planning_curriculum
        from Logic.content_pipeline import search_approved_content
        from Logic.tools.knowledge_search import search_knowledge_base

        scope: Dict[str, Any] = {}
        if subject:
            scope["subject"] = subject
        if chapter:
            scope["chapter"] = chapter
        if class_level:
            scope["class_level"] = class_level
        explicit_planning_scope = catalog_source.strip().lower() == "planning_manifest"
        curriculum = resolve_planning_curriculum(
            chapter_ref=chapter,
            subject=subject or None,
            class_level=class_level or None,
        ) if chapter and explicit_planning_scope else None
        if curriculum and section_id:
            scope.update(
                {
                    "catalog_source": "planning_manifest",
                    "chapter_slug": curriculum["chapter_slug"],
                    "class_level": curriculum["class_level"],
                }
            )
            result = search_knowledge_base(
                section_id=section_id or chapter,
                question=query or section_id,
                scope=scope,
                max_chars=max_chars,
                max_paragraphs=4,
            )
        else:
            result = search_approved_content(
                section_id=section_id or (chapter or subject or "general"),
                question=query or section_id,
                scope=scope or None,
                max_chars=max_chars,
                limit=4,
            )
        return str(result.get("context") or "")[:max_chars]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reference augmentation skipped: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# ExamPatternAnalyzerAgent
# ---------------------------------------------------------------------------
def _empty_analysis_block() -> Dict[str, Any]:
    return {
        "total_questions": 0,
        "total_marks": None,
        "section_breakdown": {},
        "marks_distribution": {},
        "question_type_distribution": {},
        "difficulty_distribution": {},
        "topic_frequency": {},
        "repeated_concepts": [],
        "high_frequency_concepts": [],
        "chapter_weightage": {},
        "short_vs_long": {},
        "pattern_style": "",
        "pattern_summary": "",
        "warnings": [],
    }


def _sanitize_question(item: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    text = J.clean_text(item.get("question_text") or item.get("question"), max_len=2000)
    if not text:
        return None
    return {
        "question_number": J.clean_text(item.get("question_number") or f"Q{index + 1}", max_len=20),
        "section_name": J.clean_text(item.get("section_name"), max_len=40),
        "question_text": text,
        "marks": J.coerce_optional_marks(item.get("marks")),
        "question_type": J.clean_text(item.get("question_type"), max_len=40).lower(),
        "intent": J.clean_text(item.get("intent"), max_len=40).lower(),
        "difficulty": J.clean_text(item.get("difficulty"), max_len=20).lower() or "medium",
        "topic": J.clean_text(item.get("topic"), max_len=120),
        "concept_tags": J.as_str_list(item.get("concept_tags"), max_items=12, max_len=120),
        "expected_answer_style": J.clean_text(item.get("expected_answer_style"), max_len=160),
        "confidence": J.clamp_float(item.get("confidence"), 0.0, 1.0, 0.5),
    }


def _recompute_aggregates(questions: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute observed aggregates from the structured questions so the stored
    analysis is always consistent with questions[]. Inference fields (pattern
    summary, repeated/high-frequency concepts, chapter weightage, style) are
    kept from the model but sanitized."""
    marks_known = [q["marks"] for q in questions if q["marks"] is not None]
    all_marks_known = bool(questions) and len(marks_known) == len(questions)

    marks_dist: Counter = Counter()
    for value in marks_known:
        key = str(int(value)) if float(value).is_integer() else str(value)
        marks_dist[key] += 1
    type_dist = Counter(q["question_type"] for q in questions if q["question_type"])
    diff_dist = Counter(q["difficulty"] for q in questions if q["difficulty"])
    topic_freq = Counter(q["topic"].lower() for q in questions if q["topic"])

    short_count = sum(1 for q in questions if q["question_type"] in _SHORT_TYPES)
    long_count = len(questions) - short_count

    warnings = J.as_str_list(analysis.get("warnings"), max_items=12, max_len=240)
    if questions and not all_marks_known:
        warnings.append("Marks were not detectable for every question; total_marks is partial or unknown.")

    repeated = J.as_str_list(analysis.get("repeated_concepts"), max_items=20, max_len=120)
    if not repeated:
        repeated = [topic for topic, n in topic_freq.most_common() if n > 1][:20]
    high_freq = J.as_str_list(analysis.get("high_frequency_concepts"), max_items=12, max_len=120)
    if not high_freq:
        high_freq = [topic for topic, _ in topic_freq.most_common(6)]

    chapter_weightage = analysis.get("chapter_weightage")
    if not isinstance(chapter_weightage, dict):
        chapter_weightage = {}

    return {
        "total_questions": len(questions),
        "total_marks": (round(sum(marks_known), 2) if all_marks_known else None),
        "section_breakdown": analysis.get("section_breakdown") if isinstance(analysis.get("section_breakdown"), dict) else {},
        "marks_distribution": dict(marks_dist),
        "question_type_distribution": dict(type_dist),
        "difficulty_distribution": dict(diff_dist),
        "topic_frequency": dict(topic_freq),
        "repeated_concepts": repeated,
        "high_frequency_concepts": high_freq,
        "chapter_weightage": chapter_weightage,
        "short_vs_long": {"short_answer": short_count, "long_answer": long_count},
        "pattern_style": J.clean_text(analysis.get("pattern_style"), max_len=40).lower(),
        "pattern_summary": J.clean_text(analysis.get("pattern_summary"), max_len=1500),
        "warnings": warnings[:12],
    }


def analyze_paper(
    *,
    paper_text: str,
    class_level: str = "",
    subject: str = "",
    chapter_name: str = "",
    exam_type: str = "",
) -> Dict[str, Any]:
    """ExamPatternAnalyzerAgent: structure one paper into questions + analysis."""
    text = (paper_text or "").strip()
    base_warnings: List[str] = []
    if not text:
        block = _empty_analysis_block()
        block["warnings"] = ["No extractable text was available to analyze."]
        return {
            "paper_title": "",
            "exam_type": (exam_type or "unknown"),
            "questions": [],
            "analysis": block,
            "confidence": 0.0,
            "warnings": block["warnings"],
        }
    if len(text) > MAX_ANALYZER_CHARS:
        text = text[:MAX_ANALYZER_CHARS]
        base_warnings.append("Paper text was long and was truncated for analysis.")

    raw, err = _complete_json(
        role="reviewer",
        agent_name="exam_pattern_analyzer_agent",
        task="analyze_exam_paper",
        messages=P.analyzer_messages(
            text, class_level=class_level, subject=subject, chapter_name=chapter_name, exam_type=exam_type
        ),
        temperature=0.1,
        max_tokens=4096,
    )
    payload = J.extract_json_object(raw) or {}

    questions: List[Dict[str, Any]] = []
    for index, item in enumerate(J.extract_json_array(payload.get("questions"))):
        sanitized = _sanitize_question(item, index)
        if sanitized:
            questions.append(sanitized)

    analysis_in = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    analysis = _recompute_aggregates(questions, analysis_in)
    analysis["warnings"] = (base_warnings + analysis["warnings"])[:12]

    if not questions:
        analysis["warnings"].append(
            "The analyzer could not extract structured questions from this paper."
            + (f" ({err})" if err else "")
        )

    confidence = J.clamp_float(payload.get("confidence"), 0.0, 1.0, 0.0)
    if questions and confidence == 0.0:
        confidence = round(sum(q["confidence"] for q in questions) / len(questions), 3)

    return {
        "paper_title": J.clean_text(payload.get("paper_title"), max_len=200),
        "exam_type": J.clean_text(payload.get("exam_type") or exam_type or "unknown", max_len=40).lower(),
        "questions": questions,
        "analysis": analysis,
        "confidence": confidence,
        "warnings": analysis["warnings"],
    }


# ---------------------------------------------------------------------------
# Cross-paper pattern intelligence (deterministic)
# ---------------------------------------------------------------------------
def aggregate_analyses(analyses: List[Dict[str, Any]], *, papers_meta: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Merge several per-paper analysis payloads into one pattern report.

    The per-paper intelligence already came from the model; aggregation is pure
    arithmetic over the observed question lists, so it needs no LLM call.
    """
    papers_meta = papers_meta or []
    marks_dist: Counter = Counter()
    type_dist: Counter = Counter()
    diff_dist: Counter = Counter()
    topic_freq: Counter = Counter()
    chapter_weightage: Counter = Counter()
    repeated: Counter = Counter()
    total_questions = 0
    total_marks_known: List[float] = []
    all_marks_known = True
    confidences: List[float] = []
    styles: Counter = Counter()

    for idx, analysis in enumerate(analyses):
        block = analysis.get("analysis") if isinstance(analysis, dict) else {}
        block = block if isinstance(block, dict) else {}
        questions = J.extract_json_array(analysis.get("questions")) if isinstance(analysis, dict) else []
        total_questions += int(block.get("total_questions") or len(questions) or 0)
        confidences.append(J.clamp_float(analysis.get("confidence"), 0.0, 1.0, 0.0))
        if block.get("pattern_style"):
            styles[str(block["pattern_style"]).lower()] += 1

        for key, value in (block.get("marks_distribution") or {}).items():
            marks_dist[str(key)] += J.coerce_int(value)
        for key, value in (block.get("question_type_distribution") or {}).items():
            type_dist[str(key)] += J.coerce_int(value)
        for key, value in (block.get("difficulty_distribution") or {}).items():
            diff_dist[str(key)] += J.coerce_int(value)
        for key, value in (block.get("topic_frequency") or {}).items():
            topic_freq[str(key).lower()] += J.coerce_int(value)
        for concept in block.get("repeated_concepts") or []:
            repeated[str(concept).lower()] += 1

        meta = papers_meta[idx] if idx < len(papers_meta) else {}
        chapter_label = str(meta.get("chapter_name") or block.get("pattern_style") or "").strip() or "general"
        chapter_weightage[chapter_label] += int(block.get("total_questions") or len(questions) or 0)

        if block.get("total_marks") is None:
            all_marks_known = False
        else:
            total_marks_known.append(J.clamp_float(block.get("total_marks"), 0.0, 100000.0, 0.0))

    # Repeated concepts across the whole set: anything appearing more than once.
    cross_repeated = [c for c, n in topic_freq.most_common() if n > 1][:25] or [c for c, _ in repeated.most_common(15)]
    high_freq = [c for c, _ in topic_freq.most_common(8)]
    pattern_style = styles.most_common(1)[0][0] if styles else "mixed"

    summary = _deterministic_pattern_summary(
        total_questions=total_questions,
        marks_dist=marks_dist,
        type_dist=type_dist,
        high_freq=high_freq,
        pattern_style=pattern_style,
        paper_count=len(analyses),
    )

    return {
        "total_questions": total_questions,
        "total_marks": (round(sum(total_marks_known), 2) if all_marks_known and total_marks_known else None),
        "marks_distribution": dict(marks_dist),
        "question_type_distribution": dict(type_dist),
        "difficulty_distribution": dict(diff_dist),
        "topic_frequency": dict(topic_freq),
        "repeated_concepts": cross_repeated,
        "high_frequency_concepts": high_freq,
        "chapter_weightage": dict(chapter_weightage),
        "pattern_style": pattern_style,
        "pattern_summary": summary,
        "confidence_score": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
    }


def _deterministic_pattern_summary(*, total_questions, marks_dist, type_dist, high_freq, pattern_style, paper_count) -> str:
    parts = [f"Across {paper_count} uploaded paper(s) with {total_questions} question(s)."]
    if marks_dist:
        top_marks = ", ".join(f"{m}-mark x{n}" for m, n in marks_dist.most_common(4))
        parts.append(f"Most common marks: {top_marks}.")
    if type_dist:
        top_types = ", ".join(f"{t} x{n}" for t, n in type_dist.most_common(3))
        parts.append(f"Question types: {top_types}.")
    if high_freq:
        parts.append("High-frequency concepts: " + ", ".join(high_freq[:5]) + ".")
    if pattern_style:
        parts.append(f"Overall pattern leans {pattern_style.replace('_', ' ')}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# ProbableQuestionAgent
# ---------------------------------------------------------------------------
def _sanitize_probable(item: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    text = J.clean_text(item.get("question") or item.get("question_text"), max_len=1200)
    if not text:
        return None
    priority = J.clean_text(item.get("priority"), max_len=12).lower()
    if priority not in {"high", "medium", "low"}:
        priority = "medium"
    return {
        "id": J.clean_text(item.get("id") or f"P{index + 1}", max_len=12) or f"P{index + 1}",
        "question": text,
        "marks": J.coerce_optional_marks(item.get("marks")),
        "question_type": J.clean_text(item.get("question_type"), max_len=40).lower(),
        "intent": J.clean_text(item.get("intent"), max_len=40).lower(),
        "topic": J.clean_text(item.get("topic"), max_len=120),
        "priority": priority,
        "based_on": J.clean_text(item.get("based_on"), max_len=240),
        "source": "uploaded_papers_pattern",
    }


def _fallback_probable_questions(analysis_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    topics = list((analysis_payload.get("topic_frequency") or {}).keys())
    topics += [t for t in (analysis_payload.get("high_frequency_concepts") or []) if t not in topics]
    topics = [t for t in topics if t][:6] or ["this chapter"]
    templates = [
        ("Define and explain: {topic}.", 3, "explanation"),
        ("State the key points of {topic}.", 2, "short_note"),
        ("Explain {topic} in detail with examples.", 5, "explanation"),
        ("Differentiate the important aspects of {topic}.", 3, "difference"),
    ]
    out: List[Dict[str, Any]] = []
    for topic in topics:
        tmpl, marks, intent = templates[len(out) % len(templates)]
        out.append({
            "id": f"P{len(out) + 1}",
            "question": tmpl.format(topic=topic),
            "marks": marks,
            "question_type": "long_answer" if marks >= 5 else "short_answer",
            "intent": intent,
            "topic": topic,
            "priority": "high" if len(out) < 2 else "medium",
            "based_on": "repeated/high-frequency concept in your uploaded papers",
            "source": "uploaded_papers_pattern",
        })
        if len(out) >= 8:
            break
    return out


def generate_probable_questions(
    *,
    analysis_payload: Dict[str, Any],
    class_level: str = "",
    subject: str = "",
    chapter_name: str = "",
    generation_mode: str = "mixed",
    count: int = 8,
    augment_context: str = "",
) -> Dict[str, Any]:
    """ProbableQuestionAgent: most-probable practice questions (never guaranteed)."""
    raw, err = _complete_json(
        role="tutor",
        agent_name="probable_question_agent",
        task="generate_probable_questions",
        messages=P.probable_messages(
            analysis_payload=analysis_payload,
            class_level=class_level,
            subject=subject,
            chapter_name=chapter_name,
            generation_mode=generation_mode,
            count=count,
            augment_context=augment_context,
        ),
        temperature=0.4,
        max_tokens=3000,
    )
    payload = J.extract_json_object(raw) or {}

    questions: List[Dict[str, Any]] = []
    for index, item in enumerate(J.extract_json_array(payload.get("probable_questions"))):
        sanitized = _sanitize_probable(item, index)
        if sanitized:
            questions.append(sanitized)
        if len(questions) >= count:
            break
    fallback_used = not questions
    if fallback_used:
        questions = _fallback_probable_questions(analysis_payload)[:count]

    priority_topics: List[Dict[str, Any]] = []
    for item in J.extract_json_array(payload.get("priority_topics"), list_keys=("priority_topics", "items")):
        topic = J.clean_text(item.get("topic"), max_len=120)
        if not topic:
            continue
        weight = J.clean_text(item.get("weight"), max_len=12).lower()
        priority_topics.append({
            "topic": topic,
            "reason": J.clean_text(item.get("reason"), max_len=240),
            "weight": weight if weight in {"high", "medium", "low"} else "medium",
        })
    if not priority_topics:
        priority_topics = [
            {"topic": t, "reason": "frequently asked in your uploaded papers", "weight": "high"}
            for t in (analysis_payload.get("high_frequency_concepts") or [])[:5]
        ]

    strategy = J.clean_text(payload.get("strategy_summary"), max_len=1500)
    if not strategy:
        strategy = (
            "Prioritize the high-frequency concepts above, practice the most common "
            "marks formats first, and revise repeated concepts last for retention."
        )

    confidence = J.clamp_float(payload.get("confidence"), 0.0, 1.0, 0.0)
    if confidence == 0.0 and not fallback_used:
        confidence = 0.6

    return {
        "probable_questions": questions,
        "priority_topics": priority_topics,
        "strategy_summary": strategy,
        "disclaimer": P.PROBABLE_DISCLAIMER,
        "confidence": confidence,
        "fallback_used": fallback_used,
    }


# ---------------------------------------------------------------------------
# Written question setter
# ---------------------------------------------------------------------------
def generate_written_question(
    *,
    class_level: str = "",
    subject: str = "",
    chapter_name: str = "",
    topic: str = "",
    marks_focus: str = "",
    question_type: str = "",
    augment_context: str = "",
) -> Dict[str, Any]:
    raw, err = _complete_json(
        role="tutor",
        agent_name="written_question_agent",
        task="generate_written_question",
        messages=P.written_question_messages(
            class_level=class_level,
            subject=subject,
            chapter_name=chapter_name,
            topic=topic,
            marks_focus=marks_focus,
            question_type=question_type,
            augment_context=augment_context,
        ),
        temperature=0.5,
        max_tokens=1200,
    )
    payload = J.extract_json_object(raw) or {}

    default_marks = J.coerce_int(marks_focus, 0) or 3
    marks_total = J.coerce_optional_marks(payload.get("marks_total")) or float(default_marks)
    focus = topic or chapter_name or subject or "this chapter"
    question_text = J.clean_text(payload.get("question_text"), max_len=1500)
    if not question_text:
        question_text = f"Explain {focus} in detail with suitable examples."
        question_type = question_type or "long_answer"
    expected_points = J.as_str_list(payload.get("expected_points"), max_items=15, max_len=400)

    return {
        "question_text": question_text,
        "question_type": J.clean_text(payload.get("question_type") or question_type, max_len=40).lower() or "long_answer",
        "marks_total": marks_total,
        "topic": J.clean_text(payload.get("topic") or topic, max_len=120) or focus,
        "command_word": J.clean_text(payload.get("command_word"), max_len=40).lower(),
        "expected_points": expected_points,
        "fallback_used": not bool(payload),
    }


# ---------------------------------------------------------------------------
# WrittenAnswerEvaluatorAgent + TeacherFeedbackAgent
# ---------------------------------------------------------------------------
_RUBRIC_KEYS = (
    "concept_accuracy", "key_points_covered", "completeness",
    "formula_keyword_usage", "step_logic", "explanation_clarity", "exam_presentation",
)


def _keywords(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(w) > 3 and w not in _STOPWORDS
    }


def _fallback_evaluation(
    *, question_text: str, marks_total: float, student_answer: str, expected_points: List[str]
) -> Dict[str, Any]:
    """Lexical-overlap grading used only when the LLM is unavailable, so a student
    still gets a usable (rough) score and the missing points instead of an error."""
    answer_kw = _keywords(student_answer)
    covered, missing = [], []
    if expected_points:
        for point in expected_points:
            point_kw = _keywords(point)
            overlap = len(point_kw & answer_kw) / max(1, len(point_kw))
            (covered if overlap >= 0.34 else missing).append(point)
        ratio = len(covered) / len(expected_points)
    else:
        # No marking scheme available: score on answer substance only.
        ratio = min(1.0, len(answer_kw) / 25) if student_answer.strip() else 0.0
    marks_awarded = round(ratio * float(marks_total or 0), 2)
    return {
        "marks_awarded": marks_awarded,
        "marks_total": float(marks_total or 0),
        "covered_points": covered,
        "missing_points": missing,
        "incorrect_points": [],
        "weak_explanation_areas": ([] if ratio >= 0.6 else ["Answer lacked depth on the expected points."]),
        "presentation_feedback": "Automatic check: structure your answer in clear points or steps.",
        "teacher_feedback": (
            "This is an automatic preliminary score (the AI evaluator was unavailable). "
            "Compare your answer with the expected points and revise the gaps."
        ),
        "model_answer": "",
        "improve_to_full_marks": "Cover the missing points listed above and add relevant examples/keywords.",
        "rubric_scores": {key: round(ratio, 2) for key in _RUBRIC_KEYS},
        "next_question_suggestion": f"Re-attempt: {question_text}" if question_text else "",
        "weakness_tags": ([] if ratio >= 0.6 else [{"topic": "", "weakness_type": "missing_key_points", "note": "Several expected points were missing."}]),
        "fallback_used": True,
    }


def evaluate_written_answer(
    *,
    question_text: str,
    question_type: str,
    marks_total: float,
    student_answer: str,
    expected_points: List[str],
    class_level: str = "",
    subject: str = "",
    chapter_name: str = "",
    topic: str = "",
    augment_context: str = "",
) -> Dict[str, Any]:
    """Grade a written answer like a teacher: rubric, marks, feedback, model answer."""
    marks_total = float(marks_total or 0)
    if not str(student_answer or "").strip():
        return {
            "marks_awarded": 0.0,
            "marks_total": marks_total,
            "covered_points": [],
            "missing_points": list(expected_points or []),
            "incorrect_points": [],
            "weak_explanation_areas": [],
            "presentation_feedback": "No answer was submitted.",
            "teacher_feedback": "No answer was submitted, so no marks could be awarded. Attempt the question to get feedback.",
            "model_answer": "",
            "improve_to_full_marks": "Write an answer covering the expected key points.",
            "rubric_scores": {key: 0.0 for key in _RUBRIC_KEYS},
            "next_question_suggestion": "",
            "weakness_tags": [],
            "fallback_used": False,
        }

    raw, err = _complete_json(
        role="reviewer",
        agent_name="written_answer_evaluator_agent",
        task="evaluate_written_answer",
        messages=P.written_eval_messages(
            question_text=question_text,
            question_type=question_type,
            marks_total=marks_total,
            student_answer=student_answer,
            expected_points=expected_points,
            class_level=class_level,
            subject=subject,
            chapter_name=chapter_name,
            augment_context=augment_context,
        ),
        temperature=0.2,
        max_tokens=2200,
    )
    payload = J.extract_json_object(raw)
    if not payload:
        return _fallback_evaluation(
            question_text=question_text,
            marks_total=marks_total,
            student_answer=student_answer,
            expected_points=expected_points,
        )

    marks_awarded = J.clamp_float(payload.get("marks_awarded"), 0.0, marks_total or 0.0, 0.0)
    rubric_in = payload.get("rubric_scores") if isinstance(payload.get("rubric_scores"), dict) else {}
    rubric = {key: J.clamp_float(rubric_in.get(key), 0.0, 1.0, 0.0) for key in _RUBRIC_KEYS}

    weakness_tags: List[Dict[str, Any]] = []
    for tag in J.extract_json_array(payload.get("weakness_tags"), list_keys=("weakness_tags", "items")):
        wtype = J.clean_text(tag.get("weakness_type"), max_len=40).lower()
        weakness_tags.append({
            "topic": J.clean_text(tag.get("topic") or topic, max_len=120),
            "weakness_type": wtype or "missing_key_points",
            "note": J.clean_text(tag.get("note"), max_len=240),
        })

    return {
        "marks_awarded": round(marks_awarded, 2),
        "marks_total": marks_total,
        "covered_points": J.as_str_list(payload.get("covered_points"), max_items=20, max_len=400),
        "missing_points": J.as_str_list(payload.get("missing_points"), max_items=20, max_len=400),
        "incorrect_points": J.as_str_list(payload.get("incorrect_points"), max_items=20, max_len=400),
        "weak_explanation_areas": J.as_str_list(payload.get("weak_explanation_areas"), max_items=15, max_len=400),
        "presentation_feedback": J.clean_text(payload.get("presentation_feedback"), max_len=1200),
        "teacher_feedback": J.clean_text(payload.get("teacher_feedback"), max_len=2000),
        "model_answer": J.clean_text(payload.get("model_answer"), max_len=6000),
        "improve_to_full_marks": J.clean_text(payload.get("improve_to_full_marks"), max_len=2000),
        "rubric_scores": rubric,
        "next_question_suggestion": J.clean_text(payload.get("next_question_suggestion"), max_len=600),
        "weakness_tags": weakness_tags,
        "fallback_used": False,
    }


# ---------------------------------------------------------------------------
# StudentWeaknessTrackerAgent (deterministic)
# ---------------------------------------------------------------------------
def derive_weaknesses(
    *,
    evaluation: Dict[str, Any],
    subject: str = "",
    class_level: str = "",
    chapter_name: str = "",
    topic: str = "",
) -> List[Dict[str, Any]]:
    """Turn one evaluation into durable weakness signals (no extra LLM)."""
    signals: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(weakness_type: str, summary: str, evidence: List[str], suggestion: str, topic_value: str = "") -> None:
        topic_final = (topic_value or topic or chapter_name or "").strip()
        key = (weakness_type, topic_final.lower())
        if not weakness_type or key in seen:
            return
        seen.add(key)
        signals.append({
            "topic": topic_final,
            "weakness_type": weakness_type,
            "weakness_summary": summary[:400],
            "evidence": [str(e)[:300] for e in (evidence or [])][:6],
            "improvement_suggestion": suggestion[:400],
        })

    # 1) Explicit tags from the evaluator.
    for tag in evaluation.get("weakness_tags") or []:
        if isinstance(tag, dict) and tag.get("weakness_type"):
            _add(
                str(tag["weakness_type"]).lower(),
                str(tag.get("note") or "Identified during answer evaluation."),
                [str(tag.get("note") or "")],
                evaluation.get("improve_to_full_marks") or "Revise this point and re-attempt.",
                topic_value=str(tag.get("topic") or ""),
            )

    # 2) Inferred from the structured result.
    missing = evaluation.get("missing_points") or []
    if missing:
        _add("missing_key_points", "Key marking points were missing from the answer.", missing,
             evaluation.get("improve_to_full_marks") or "Include the missing key points.")
    if evaluation.get("incorrect_points"):
        _add("concept_gap", "Some statements were incorrect or misleading.",
             evaluation.get("incorrect_points") or [],
             "Re-study the concept to correct these mistakes.")
    if evaluation.get("weak_explanation_areas"):
        _add("clarity", "Explanation was unclear or incomplete in places.",
             evaluation.get("weak_explanation_areas") or [],
             "Practice explaining the concept step by step.")

    rubric = evaluation.get("rubric_scores") or {}
    if J.clamp_float(rubric.get("exam_presentation"), 0.0, 1.0, 1.0) < 0.5:
        _add("presentation", "Answer presentation/structure needs improvement.", [],
             "Use headings, points, steps, and units; label any diagrams.")
    if J.clamp_float(rubric.get("step_logic"), 0.0, 1.0, 1.0) < 0.5:
        _add("step_logic", "Step-by-step reasoning was weak.", [],
             "Show each step of the derivation/working clearly.")

    marks_total = float(evaluation.get("marks_total") or 0)
    if marks_total > 0 and float(evaluation.get("marks_awarded") or 0) / marks_total < 0.5 and not signals:
        _add("incomplete", "Overall answer scored below half marks.",
             missing, "Cover more of the expected points and add depth.")

    return signals
