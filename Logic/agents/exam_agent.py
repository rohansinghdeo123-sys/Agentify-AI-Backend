# Logic/agents/exam_agent.py

"""
EXAM AGENT - MCQ Generation & Probable Questions

This agent supports both legacy text output and structured exam payloads.
Structured MCQs are returned in `data.questions` for the frontend.
Now uses the Knowledge Graph to extract common mistakes as distractors
and align questions with curriculum learning objectives.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from Logic.agent_event_bus import event_bus
from Logic.coach.model_gateway import model_gateway
from Logic.tools.answer_evaluator import evaluate_answer_quality
from Logic.tools.chemistry_formatter import format_chemistry_output
from Logic.tools.knowledge_search import search_knowledge_base
from Logic.tools.mcq_normalization import (
    normalize_mcq_answer,
    normalize_mcq_options,
    parse_text_mcqs,  # noqa: F401  (used by the legacy text-output fallback)
)
from Logic.knowledge_graph import knowledge_graph   # <-- NEW

logger = logging.getLogger("ai_educator.agents.exam")

# The shared gateway owns timeouts, retries, provider fallback, and cost
# records; GROQ_TUTOR_MODEL still selects the primary model.
MODEL_NAME = model_gateway.model_for("tutor")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = text.strip()

    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def normalize_structured_mcqs(payload: Optional[Dict[str, Any]], count: int = 5) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    questions = payload.get("questions")
    if not isinstance(questions, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(questions[:count]):
        if not isinstance(item, dict):
            continue

        question = str(item.get("question") or "").strip()
        if not question:
            continue

        correct = normalize_mcq_answer(item)
        if not correct:
            continue

        normalized.append(
            {
                "id": str(item.get("id") or f"Q{index + 1}"),
                "question": question,
                "options": normalize_mcq_options(item.get("options")),
                "correct": correct,
                "explanation": str(item.get("explanation") or "").strip(),
                "source": str(
                    item.get("source")
                    or item.get("reference")
                    or payload.get("source")
                    or payload.get("section_id")
                    or ""
                ).strip(),
            }
        )

    return normalized


def mcqs_to_legacy_text(questions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for index, item in enumerate(questions, start=1):
        lines.append(f"Q{index}. {item['question']}")
        lines.extend(item["options"])
        lines.append(f"Answer: {item['correct']}")
        lines.append(f"Explanation: {item.get('explanation', '')}")
        lines.append("")

    return "\n".join(lines).strip()


def build_structured_mcq_prompt(context: str, count: int = 5) -> str:
    return f"""
You are a friendly, patient AI exam creator for school students. Create exactly {count} multiple-choice questions from the provided content.

Use simple language and clear phrasing. Make sure the questions test understanding, not just memorisation.

When the content includes 'Common Mistakes', use those typical errors as WRONG options (distractors) to help students learn from mistakes.

Return valid JSON only.
Do not include markdown.
Do not include text before or after JSON.

JSON shape:
{{
  "questions": [
    {{
      "id": "Q1",
      "question": "Question text",
      "options": [
        {{ "key": "A", "text": "Option text" }},
        {{ "key": "B", "text": "Option text" }},
        {{ "key": "C", "text": "Option text" }},
        {{ "key": "D", "text": "Option text" }}
      ],
      "answer": "A",
      "explanation": "Short feedback explaining the correct answer in a friendly tone",
      "source": "Short topic or line reference from SECTION CONTENT"
    }}
  ]
}}

Rules:
- Exactly {count} questions.
- Exactly 4 options per question.
- One and only one correct answer per question.
- Where possible, include a common misconception as one of the wrong options.
- Explanations must be helpful and encouraging.
- Every question must be traceable to SECTION CONTENT.
- Add a short source/topic reference for every question.
- Use Unicode chemical subscripts and superscripts.

SECTION CONTENT:
{context}
""".strip()


def build_structured_probable_prompt(context: str) -> str:
    return f"""
You are a friendly, helpful AI exam setter for school students. Generate probable theory questions from the provided content.

Return valid JSON only.
Do not include markdown.
Do not include text before or after JSON.

JSON shape:
{{
  "questions": [
    {{ "id": "Q1", "marks": 3, "question": "Question text", "source": "Short topic or line reference" }},
    {{ "id": "Q2", "marks": 3, "question": "Question text", "source": "Short topic or line reference" }},
    {{ "id": "Q3", "marks": 3, "question": "Question text", "source": "Short topic or line reference" }},
    {{ "id": "Q4", "marks": 5, "question": "Question text", "source": "Short topic or line reference" }},
    {{ "id": "Q5", "marks": 5, "question": "Question text", "source": "Short topic or line reference" }}
  ]
}}

Rules:
- Generate exactly 5 questions.
- First 3 questions are 3 marks, last 2 are 5 marks.
- Questions should test deep understanding.
- Do not provide answers or explanations.
- Use friendly, clear language.
- Every question must be traceable to SECTION CONTENT.
- Add a short source/topic reference for every question.

SECTION CONTENT:
{context}
""".strip()


def normalize_probable_questions(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    questions = payload.get("questions")
    if not isinstance(questions, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(questions[:5]):
        if not isinstance(item, dict):
            continue

        question = str(item.get("question") or "").strip()
        if not question:
            continue

        fallback_marks = 3 if index < 3 else 5

        try:
            marks = int(item.get("marks") or fallback_marks)
        except Exception:
            marks = fallback_marks

        normalized.append(
            {
                "id": str(item.get("id") or f"Q{index + 1}"),
                "marks": marks,
                "question": question,
                "source": str(
                    item.get("source")
                    or item.get("reference")
                    or payload.get("source")
                    or payload.get("section_id")
                    or ""
                ).strip(),
            }
        )

    return normalized


def probable_to_text(questions: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['id']} ({item['marks']} Marks): {item['question']}"
        for item in questions
    )


def call_groq(prompt: str, temperature: float, max_tokens: int) -> str:
    kwargs = {
        "role": "tutor",
        "messages": [{"role": "user", "content": prompt}],
        "agent_name": "tutor_model",
        "task": "Generate structured exam questions from section context.",
        "student_visible": True,
        "safety_tier": "final_answer",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        # JSON mode removes the whole regex-rescue tier when supported.
        return model_gateway.complete(
            **kwargs, response_format={"type": "json_object"}
        ).strip()
    except Exception as exc:
        logger.warning("[EXAM] JSON mode unavailable, retrying as plain text: %s", exc)
        return model_gateway.complete(**kwargs).strip()


def exam_agent(request, exam_type: str = "mcq") -> dict:
    start_time = time.time()

    section_id = request.section_id
    question = request.question
    content_scope = getattr(request, "content_scope", None)
    uses_strict_catalog = (
        str((content_scope or {}).get("catalog_source") or "").strip().lower()
        in {"published", "planning_manifest"}
    )
    try:
        requested_count = max(1, min(int(getattr(request, "count", 5) or 5), 10))
    except Exception:
        requested_count = 5

    logger.info("[EXAM] Type: %s | Section: %s | Count: %s", exam_type, section_id, requested_count)

    event_bus.emit(
        "exam",
        "task_start",
        {
            "task": f"{exam_type.upper()} generation for {section_id}",
            "section": section_id,
            "exam_type": exam_type,
            "message": f"Starting {exam_type} generation for {section_id}",
        },
    )

    # ===== STEP 1: RETRIEVE from markdown =====
    event_bus.emit(
        "exam",
        "tool_call",
        {
            "step": "retrieve_markdown",
            "step_num": 1,
            "total_steps": 5,
            "tool": "knowledge_search",
            "message": f"Searching markdown knowledge base for {section_id}...",
        },
    )

    # Retrieve against the bare topic/section, not the long JSON instruction
    # carried in `question` — that boilerplate buries the real topic and
    # degrades retrieval.
    retrieval_query = (getattr(request, "topic", "") or section_id or question).strip()
    search_result = search_knowledge_base(
        section_id=section_id,
        question=retrieval_query,
        max_paragraphs=8,
        max_chars=4000,
        scope=content_scope,
    )

    if search_result.get("error"):
        not_found = (
            getattr(request, "required_not_found_response", "")
            or "I could not find this in your study material. Please upload or select the correct chapter/data."
        )
        event_bus.emit(
            "exam",
            "error",
            {
                "step": "retrieve_markdown",
                "message": f"Knowledge base lookup failed for section '{section_id}': {search_result['error']}",
            },
            severity="error",
        )
        return {
            "type": "exam",
            "answer": not_found,
            "data": None,
            "metadata": {"agent": "exam", "exam_type": exam_type, "status": "material_not_found"},
        }

    context = search_result["context"]

    event_bus.emit(
        "exam",
        "step",
        {
            "step": "retrieve_markdown_complete",
            "step_num": 1,
            "total_steps": 5,
            "message": f"Retrieved {search_result['paragraphs_found']} paragraphs from markdown",
        },
    )

    # ===== STEP 2: ENRICH with knowledge graph (extract common mistakes for distractors) =====
    event_bus.emit(
        "exam",
        "tool_call",
        {
            "step": "retrieve_graph",
            "step_num": 2,
            "total_steps": 5,
            "tool": "knowledge_graph",
            "message": "Searching knowledge graph for common mistakes and learning objectives...",
        },
    )

    concept_data = ""
    concepts_found = []
    if not uses_strict_catalog and knowledge_graph.concepts:
        exact = knowledge_graph.get_concept(section_id)
        if exact:
            concepts_found = [exact]
        else:
            keywords = [w.strip().lower() for w in question.replace("?", "").replace(".", "").split() if len(w) > 3]
            search_kw = keywords[0] if keywords else section_id
            concepts_found = knowledge_graph.search_by_keyword(search_kw, limit=3)

        if concepts_found:
            blocks = []
            for c in concepts_found:
                block = f"--- CONCEPT: {c['title']} ---\n"
                if c.get("common_mistakes"):
                    mistakes = "\n  ".join([f"Mistake: {m['mistake']} (Student thinks: {m['mistake']}) -> Correction: {m['correction']}" for m in c["common_mistakes"]])
                    block += "Common Mistakes (use these as WRONG options):\n  " + mistakes + "\n"
                if c.get("learning_objectives"):
                    block += "Learning Objectives:\n  " + "\n  ".join(c["learning_objectives"]) + "\n"
                blocks.append(block)
            concept_data = "\n\n".join(blocks)

    # Combine markdown and graph data
    enriched_context = context
    if concept_data:
        enriched_context += "\n\n--- COMMON MISTAKES & OBJECTIVES (use these for distractors) ---\n" + concept_data

    event_bus.emit(
        "exam",
        "step",
        {
            "step": "retrieve_graph_complete",
            "step_num": 2,
            "total_steps": 5,
            "message": f"Found {len(concepts_found)} concept(s) with common mistakes",
            "concepts_found": len(concepts_found),
        },
    )

    # ===== STEP 3: BUILD PROMPT =====
    if exam_type == "probable":
        prompt = build_structured_probable_prompt(enriched_context)
        temperature = 0.25
        max_tokens = 900
    else:
        prompt = build_structured_mcq_prompt(enriched_context, count=requested_count)
        temperature = 0.2
        # Scale output room with the requested question count so a 10-question
        # pack is not silently truncated at the 5-question token budget.
        max_tokens = min(3200, 300 * requested_count + 200)

    event_bus.emit(
        "exam",
        "tool_call",
        {
            "step": "generate",
            "step_num": 3,
            "total_steps": 5,
            "tool": "groq_llm",
            "message": f"Generating {exam_type} via {MODEL_NAME}...",
            "model": MODEL_NAME,
            "temperature": temperature,
        },
    )

    try:
        raw_answer = call_groq(prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.error("[EXAM] Groq API error: %s", e)
        event_bus.emit(
            "exam",
            "error",
            {
                "step": "generate",
                "message": f"LLM API error: {str(e)}",
            },
            severity="error",
        )

        event_bus.emit(
            "exam",
            "task_complete",
            {
                "status": "failed",
                "message": "Generation failed",
                "latency_ms": round((time.time() - start_time) * 1000),
            },
        )

        return {
            "type": "exam",
            "exam_type": exam_type,
            "answer": "AI service encountered an error. Please try again.",
            "data": None,
            "metadata": {"agent": "exam", "error": str(e)},
        }

    formatted_answer = format_chemistry_output(raw_answer)
    payload = extract_json_object(formatted_answer)

    if exam_type == "probable":
        probable_questions = normalize_probable_questions(payload)
        answer_text = probable_to_text(probable_questions) if probable_questions else formatted_answer
        data = {
            "questions": probable_questions,
            "text": answer_text,
        }
    else:
        mcq_questions = normalize_structured_mcqs(payload, count=requested_count)

        if len(mcq_questions) < requested_count:
            text_questions = parse_text_mcqs(formatted_answer, count=requested_count)
            if len(text_questions) > len(mcq_questions):
                mcq_questions = text_questions

        answer_text = mcqs_to_legacy_text(mcq_questions) if mcq_questions else formatted_answer
        data = {
            "questions": mcq_questions,
        }

    # ===== STEP 5: VALIDATE =====
    event_bus.emit(
        "exam",
        "tool_call",
        {
            "step": "validate",
            "step_num": 5,
            "total_steps": 5,
            "tool": "answer_evaluator",
            "message": "Validating exam content quality...",
        },
    )

    quality = evaluate_answer_quality(
        question=question,
        answer=answer_text,
        mode="exam",
        context=context,
    )

    latency_ms = round((time.time() - start_time) * 1000)

    event_bus.emit(
        "exam",
        "task_complete",
        {
            "status": "success",
            "message": f"{exam_type.upper()} delivered. Quality: {quality['score']}",
            "latency_ms": latency_ms,
            "quality_score": quality["score"],
            "quality_passed": quality["passed"],
        },
    )

    logger.info("[EXAM] Complete. Quality: %s | Latency: %sms", quality["score"], latency_ms)

    return {
        "type": "exam",
        "exam_type": exam_type,
        "answer": answer_text,
        "data": data,
        "metadata": {
            "agent": "exam",
            "exam_type": exam_type,
            "quality_score": quality["score"],
            "quality_passed": quality["passed"],
            "latency_ms": latency_ms,
            "structured": bool(data.get("questions")),
        },
    }
