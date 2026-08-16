# Logic/section_doubt.py

"""
SECTION DOUBT - Backward-Compatible Entry Point

Keeps the legacy section_doubt() function alive while adding structured
exam-generation helpers for the dashboard/study frontend.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from Logic.agent_router import route_to_agent
from Logic.agents.tutor_agent import reset_tutor_session
from Logic.knowledge_graph import knowledge_graph
from Logic.tools.knowledge_search import search_knowledge_base

# Canonical MCQ helpers live in one module and are re-exported here for the
# existing importers of this facade.
from Logic.tools.mcq_normalization import (  # noqa: F401
    mcq_options_are_valid,
    normalize_mcq_answer,
    normalize_mcq_options,
    normalize_option_key,
    parse_text_mcqs,
    shuffle_mcq_options,
)

logger = logging.getLogger("ai_educator.section_doubt")

MATERIAL_NOT_FOUND_MESSAGE = "I could not find this in your study material. Please upload or select the correct chapter/data."


class _LegacyRequest:
    """Adapter to convert function args into a request-like object."""

    def __init__(
        self,
        question,
        section_id,
        session_id,
        mode,
        difficulty,
        strict_grounding: bool = False,
        required_not_found_response: Optional[str] = None,
        count: int = 5,
        topic: Optional[str] = None,
        class_level: str = "",
        content_scope: Optional[Dict[str, Any]] = None,
    ):
        self.question = question
        self.section_id = section_id
        self.session_id = session_id
        self.mode = mode
        self.difficulty = difficulty
        # Bare topic for retrieval. The `question` passed by the structured
        # generators is a long JSON instruction; using it as the search query
        # buries the real topic, so retrieval keys off this instead.
        self.topic = (topic or section_id or "").strip()
        self.strict_grounding = strict_grounding
        self.retrieval_required = strict_grounding
        self.fallback_to_general_knowledge = not strict_grounding
        self.required_not_found_response = required_not_found_response or MATERIAL_NOT_FOUND_MESSAGE
        self.count = count
        self.class_level = class_level
        self.content_scope = dict(content_scope or {})
        self.subject = str(self.content_scope.get("subject") or "")
        self.chapter = str(self.content_scope.get("chapter") or "")
        self.learning_context = {"class_level": class_level} if class_level else {}


def normalize_section_id(section_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (section_id or "").strip().lower()).strip("_")
    aliases = {
        "basic_concepts_of_chemistry": "matter_definition",
        "basic_concept_of_chemistry": "matter_definition",
        "matter": "matter_definition",
        "hydrocarbon": "alkanes",
        "hydrocarbons": "alkanes",
        "aromatic_hydrocarbons": "aromatics",
    }
    return aliases.get(cleaned, cleaned)


def section_doubt(
    question: str,
    section_id: str,
    session_id: str,
    mode: str = "revision",
    difficulty: str = "medium",
    strict_grounding: bool = False,
    required_not_found_response: Optional[str] = None,
    class_level: str = "",
    content_scope: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Main entry point for the section-based AI tutor.

    Backward-compatible with old /section-ai calls.
    """
    if not section_id:
        return "Invalid section selected."

    section_id = normalize_section_id(section_id)
    not_found = required_not_found_response or MATERIAL_NOT_FOUND_MESSAGE

    if strict_grounding:
        search_result = search_knowledge_base(
            section_id=section_id,
            question=question or section_id,
            max_paragraphs=8,
            max_chars=4000,
            scope=content_scope,
        )
        if search_result.get("error") or not str(search_result.get("context") or "").strip():
            return not_found

    class_context = (
        f"Student class level: {class_level}. Match the vocabulary, depth, examples, and exam framing to this level.\n\n"
        if class_level
        else ""
    )
    request = _LegacyRequest(
        question=f"{class_context}{question}",
        section_id=section_id,
        session_id=session_id,
        mode=mode,
        difficulty=difficulty,
        strict_grounding=strict_grounding,
        required_not_found_response=not_found,
        class_level=class_level,
        content_scope=content_scope,
    )

    logger.info(
        "[SECTION_DOUBT] Routing to agent system: mode=%s, section=%s",
        mode,
        section_id,
    )

    result = route_to_agent(request)

    if isinstance(result, dict):
        answer = result.get("answer", "") or result.get("data", "")
    else:
        answer = str(result or "")

    return answer or "No response generated."


def reset_conversation(session_id: str):
    """Clear conversation memory for a session."""
    reset_tutor_session(session_id)
    logger.info("Session %s memory cleared.", session_id)


# =====================================================
# STRUCTURED GENERATION HELPERS
# =====================================================
def extract_json_payload(value: Any) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from agent output.

    Agents may return:
    - a dict
    - a raw JSON string
    - a string with surrounding text
    """
    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return None

    text = value.strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def unwrap_agent_data(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Agent routes commonly return:
    { "answer": "...", "data": { "questions": [...] } }

    This helper gives structured endpoints the nested data object when present.
    """
    if not isinstance(payload, dict):
        return payload

    data = payload.get("data")
    if isinstance(data, dict):
        return data

    return payload


def run_structured_agent(
    question: str,
    section_id: str,
    session_id: str,
    mode: str,
    difficulty: str,
    strict_grounding: bool = False,
    required_not_found_response: Optional[str] = None,
    count: int = 5,
    topic: Optional[str] = None,
    class_level: str = "",
    content_scope: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Any]:
    request = _LegacyRequest(
        question=question,
        section_id=normalize_section_id(section_id),
        session_id=session_id,
        mode=mode,
        difficulty=difficulty,
        strict_grounding=strict_grounding,
        required_not_found_response=required_not_found_response,
        count=count,
        topic=topic,
        class_level=class_level,
        content_scope=content_scope,
    )

    result = route_to_agent(request)

    if isinstance(result, dict):
        raw = result.get("data") or result.get("answer") or result
    else:
        raw = result

    payload = unwrap_agent_data(extract_json_payload(raw))
    return payload, raw


# (normalize_option_key / normalize_mcq_options / normalize_mcq_answer /
#  mcq_options_are_valid / shuffle_mcq_options are imported from
#  Logic.tools.mcq_normalization above.)


def normalize_mcq_questions(payload: Optional[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    questions = payload.get("questions")
    if not isinstance(questions, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(questions[:count]):
        if not isinstance(item, dict):
            continue

        question_text = str(item.get("question") or "").strip()
        if not question_text:
            continue

        options = normalize_mcq_options(item.get("options"))
        explanation = str(item.get("explanation") or "").strip()

        if not mcq_options_are_valid(options):
            continue
        if not explanation:
            continue
        correct = normalize_mcq_answer(item)
        if not correct:
            continue

        normalized.append(
            shuffle_mcq_options(
                {
                    "id": str(item.get("id") or f"Q{index + 1}"),
                    "question": question_text,
                    "options": options,
                    "correct": correct,
                    "explanation": explanation,
                    "source": str(
                        item.get("source")
                        or item.get("reference")
                        or payload.get("source")
                        or payload.get("section_id")
                        or ""
                    ).strip(),
                }
            )
        )

    return normalized


def _clean_grounded_fact(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^[#>\-\d\.\)\s]+", "", text).strip()
    text = re.sub(r"^\*\*([^*]+):?\*\*\s*:?\s*", r"\1: ", text)
    text = re.sub(r"^(definition|explanation):\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" -")


def _append_unique_fact(facts: List[str], value: Any, limit: int = 18) -> None:
    text = _clean_grounded_fact(value)
    words = re.findall(r"[A-Za-z0-9]+", text)
    lowered = text.lower()
    if len(text) < 18 or len(text) > 260 or len(words) < 5:
        return
    if text.endswith(("?", ":")) or re.match(r"""^['"]""", text):
        return
    if re.match(r"^[A-Z]\.\s+", text):
        return
    if lowered.startswith(("why ", "how ", "what ", "which ", "when ", "where ", "who ", "meaning ")):
        return
    if " meaning " in f" {lowered} " and len(words) < 8:
        return

    if lowered.startswith(("source:", "section content:", "common mistakes:", "key points:", "examples:", "formulas:")):
        return
    if lowered in {item.lower() for item in facts}:
        return
    if len(facts) < limit:
        facts.append(text)


def collect_grounded_facts(search_result: Dict[str, Any], section_id: str) -> List[str]:
    """Extract compact, traceable facts for deterministic exam recovery."""
    facts: List[str] = []
    concept = knowledge_graph.get_concept(section_id)

    if concept:
        _append_unique_fact(facts, concept.get("definition"))
        _append_unique_fact(facts, concept.get("core_explanation"))
        for field in ("key_points", "properties", "formulas", "examples", "applications"):
            for item in concept.get(field, []) or []:
                _append_unique_fact(facts, item)

    context = str(search_result.get("context") or "")
    for line in re.split(r"[\r\n]+", context):
        _append_unique_fact(facts, line)

    if len(facts) < 8:
        for sentence in re.split(r"(?<=[.!?])\s+", context):
            _append_unique_fact(facts, sentence)

    return facts


def build_grounded_fallback_probable_questions(
    search_result: Dict[str, Any],
    topic: str,
    section_id: str,
) -> List[Dict[str, Any]]:
    facts = collect_grounded_facts(search_result, section_id)
    if not facts:
        return []

    prompts = [
        f"Define {topic} and state its central idea.",
        f"State three important points about {topic}.",
        f"Explain this statement from the selected material: {facts[min(1, len(facts) - 1)]}",
        f"Write a detailed note on {topic} using the important properties, examples, or applications in the selected material.",
        f"Explain {topic} in detail and connect the major points given in the selected study material.",
    ]
    return [
        {
            "id": f"Q{index + 1}",
            "marks": 3 if index < 3 else 5,
            "question": prompt,
            "source": section_id,
        }
        for index, prompt in enumerate(prompts)
    ]


def build_mcq_instruction(topic: str, difficulty: str, count: int, class_level: str = "") -> str:
    learner_level = class_level or "school"
    return f"""
Generate exactly {count} high-quality {learner_level} Chemistry MCQs for topic: {topic}.

Return valid JSON only.
Do not include markdown.
Do not include any text before or after JSON.

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
      "explanation": "Short feedback explaining the correct answer",
      "source": "Topic or line reference from the selected study material"
    }}
  ]
}}

Rules:
- Exactly {count} questions.
- Exactly 4 options per question.
- One and only one correct answer per question.
- Options must be plausible.
- Explanations must be useful feedback after user selection.
- Every question must be directly supported by the selected study material.
- Add a short source/topic reference for every question.
- Use Unicode chemical subscripts and superscripts where needed.
- Use only the current section content available to the agent.
""".strip()


def generate_structured_mcqs(
    topic: str,
    section_id: str,
    session_id: str = "exam-session",
    difficulty: str = "medium",
    count: int = 5,
    strict_grounding: bool = False,
    required_not_found_response: Optional[str] = None,
    include_source: bool = False,
    class_level: str = "",
    content_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_count = max(1, min(int(count or 5), 10))
    safe_topic = (topic or section_id or "unknown").strip()
    safe_section_id = normalize_section_id(section_id or safe_topic)
    safe_difficulty = (difficulty or "medium").strip().lower()
    not_found = required_not_found_response or MATERIAL_NOT_FOUND_MESSAGE

    search_result = search_knowledge_base(
        section_id=safe_section_id,
        question=safe_topic,
        max_paragraphs=8,
        max_chars=4000,
        scope=content_scope,
    )
    source_label = f"{safe_section_id}"

    if search_result.get("error") or not str(search_result.get("context") or "").strip():
        return {
            "topic": safe_topic,
            "section_id": safe_section_id,
            "difficulty": safe_difficulty,
            "questions": [],
            "error": not_found,
            "raw_answer": "",
        }

    def _attempt() -> Tuple[List[Dict[str, Any]], Any]:
        payload, raw = run_structured_agent(
            question=build_mcq_instruction(safe_topic, safe_difficulty, safe_count, class_level),
            section_id=safe_section_id,
            session_id=session_id,
            mode="exam",
            difficulty=safe_difficulty,
            strict_grounding=strict_grounding,
            required_not_found_response=not_found,
            count=safe_count,
            topic=safe_topic,
            class_level=class_level,
            content_scope=content_scope,
        )
        attempt_questions = normalize_mcq_questions(payload, safe_count)
        if len(attempt_questions) < safe_count:
            if isinstance(raw, dict):
                fallback_text = raw.get("answer") or json.dumps(raw)
            else:
                fallback_text = str(raw or "")
            fallback_questions = parse_text_mcqs(fallback_text, safe_count)
            if len(fallback_questions) > len(attempt_questions):
                attempt_questions = fallback_questions
        return attempt_questions, raw

    questions, raw = _attempt()

    # When the model under-delivers, ask it again instead of fabricating
    # distractors by word mutation: every shipped question must be real.
    if len(questions) < safe_count:
        retry_questions, retry_raw = _attempt()
        seen = {item["question"].strip().lower() for item in questions}
        for item in retry_questions:
            if len(questions) >= safe_count:
                break
            key = item["question"].strip().lower()
            if key in seen:
                continue
            questions.append({**item, "id": f"Q{len(questions) + 1}"})
            seen.add(key)
        if not raw:
            raw = retry_raw

    generated_count = len(questions)
    fallback_used = generated_count < safe_count

    for item in questions:
        if include_source or not item.get("source"):
            item["source"] = item.get("source") or source_label

    if not questions:
        return {
            "topic": safe_topic,
            "section_id": safe_section_id,
            "difficulty": safe_difficulty,
            "questions": [],
            "error": "The selected study material did not produce a usable exam pack. Please try again or pick another section.",
            "raw_answer": str(raw or ""),
        }

    return {
        "topic": safe_topic,
        "section_id": safe_section_id,
        "difficulty": safe_difficulty,
        "class_level": class_level,
        "questions": questions[:safe_count],
        "raw_answer": "" if questions else str(raw or ""),
        "fallback_used": fallback_used,
        "model_questions": generated_count,
    }


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

        question_text = str(item.get("question") or "").strip()
        if not question_text:
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
                "question": question_text,
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


def build_probable_instruction(topic: str, difficulty: str, class_level: str = "") -> str:
    learner_level = class_level or "school"
    return f"""
Generate probable {learner_level} exam theory questions for topic: {topic}.

Return valid JSON only.
Do not include markdown.
Do not include any text before or after JSON.

JSON shape:
{{
  "questions": [
    {{ "id": "Q1", "marks": 3, "question": "Question text", "source": "Topic or line reference" }},
    {{ "id": "Q2", "marks": 3, "question": "Question text", "source": "Topic or line reference" }},
    {{ "id": "Q3", "marks": 3, "question": "Question text", "source": "Topic or line reference" }},
    {{ "id": "Q4", "marks": 5, "question": "Question text", "source": "Topic or line reference" }},
    {{ "id": "Q5", "marks": 5, "question": "Question text", "source": "Topic or line reference" }}
  ]
}}

Rules:
- Generate exactly 5 questions.
- First 3 questions are 3 marks.
- Last 2 questions are 5 marks.
- Do not provide answers.
- Do not provide hints.
- Do not provide explanations.
- Every question must be directly supported by the selected study material.
- Add a short source/topic reference for every question.
- Use only the current section content available to the agent.
""".strip()


def format_probable_text(questions: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['id']} ({item['marks']} Marks): {item['question']}"
        for item in questions
    )


def generate_structured_probable_questions(
    topic: str,
    section_id: str,
    session_id: str = "probable-session",
    difficulty: str = "medium",
    strict_grounding: bool = False,
    required_not_found_response: Optional[str] = None,
    include_source: bool = False,
    class_level: str = "",
    content_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_topic = (topic or section_id or "unknown").strip()
    safe_section_id = normalize_section_id(section_id or safe_topic)
    safe_difficulty = (difficulty or "medium").strip().lower()
    not_found = required_not_found_response or MATERIAL_NOT_FOUND_MESSAGE

    search_result = search_knowledge_base(
        section_id=safe_section_id,
        question=safe_topic,
        max_paragraphs=8,
        max_chars=4000,
        scope=content_scope,
    )

    if search_result.get("error") or not str(search_result.get("context") or "").strip():
        return {
            "topic": safe_topic,
            "section_id": safe_section_id,
            "difficulty": safe_difficulty,
            "questions": [],
            "text": not_found,
            "error": not_found,
            "raw_answer": "",
        }

    payload, raw = run_structured_agent(
        question=build_probable_instruction(safe_topic, safe_difficulty, class_level),
        section_id=safe_section_id,
        session_id=session_id,
        mode="probable",
        difficulty=safe_difficulty,
        strict_grounding=strict_grounding,
        required_not_found_response=not_found,
        topic=safe_topic,
        class_level=class_level,
        content_scope=content_scope,
    )

    questions = normalize_probable_questions(payload)
    generated_count = len(questions)
    fallback_used = len(questions) < 5
    if fallback_used:
        grounded_fallback = build_grounded_fallback_probable_questions(
            search_result=search_result,
            topic=safe_topic,
            section_id=safe_section_id,
        )
        existing_ids = {item.get("id") for item in questions}
        for item in grounded_fallback:
            if len(questions) >= 5:
                break
            if item.get("id") in existing_ids:
                item = {**item, "id": f"Q{len(questions) + 1}"}
            questions.append(item)
            existing_ids.add(item.get("id"))

    for item in questions:
        if include_source or not item.get("source"):
            item["source"] = item.get("source") or safe_section_id

    if strict_grounding and len(questions) < 5:
        return {
            "topic": safe_topic,
            "section_id": safe_section_id,
            "difficulty": safe_difficulty,
            "questions": [],
            "text": not_found,
            "error": not_found,
            "raw_answer": "",
        }

    return {
        "topic": safe_topic,
        "section_id": safe_section_id,
        "difficulty": safe_difficulty,
        "class_level": class_level,
        "questions": questions,
        "text": format_probable_text(questions),
        "raw_answer": "" if questions else str(raw or ""),
        "fallback_used": fallback_used,
        "model_questions": generated_count,
    }
