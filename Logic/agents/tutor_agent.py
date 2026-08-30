# Logic/agents/tutor_agent.py

"""
TUTOR AGENT — The ASK AI Brain (with Admin Telemetry)

This is a TRUE AGENT that follows the Think → Act → Observe → Respond cycle:
1. THINK: Analyze the student's question
2. ACT: Search the knowledge base (markdown) AND the knowledge graph (JSON concepts)
3. OBSERVE: Evaluate the retrieved context quality, enrich with structured concept data
4. RESPOND: Generate a structured answer using the enriched context
5. VALIDATE: Check answer quality before returning

Every step emits events to the Agent Event Bus for real-time admin monitoring.
"""

import time
import logging
from prompts.agent_prompts import TUTOR_AGENT_PROMPT
from database import SessionLocal
from models import AgentChatMemory
from Logic.coach.model_gateway import model_gateway
from Logic.tools.knowledge_search import search_knowledge_base
from Logic.tools.chemistry_formatter import format_chemistry_output
from Logic.tools.answer_evaluator import evaluate_answer_quality
from Logic.agent_event_bus import event_bus
from Logic.knowledge_graph import knowledge_graph   # <-- NEW import

logger = logging.getLogger("ai_educator.agents.tutor")

# The shared gateway owns timeouts, retries, provider fallback, and cost
# records; GROQ_TUTOR_MODEL still selects the primary model.
MODEL_NAME = model_gateway.model_for("tutor")

# Doubt memory is persisted in the agent_chat_memory table (keyed by
# session_id), so it survives restarts and stays consistent across instances
# instead of living in a per-process dict. Sessions are owned/validated at the
# router. Each DB session is short-lived and never held across the LLM call.
_MEMORY_TURNS = 10  # last N messages (user + assistant) replayed to the model


def _load_session_memory(session_id: str, limit: int = _MEMORY_TURNS) -> list:
    if not session_id:
        return []
    db = SessionLocal()
    try:
        rows = (
            db.query(AgentChatMemory)
            .filter(AgentChatMemory.session_id == session_id)
            .order_by(AgentChatMemory.id.desc())
            .limit(limit)
            .all()
        )
        return [{"role": row.role, "content": row.content} for row in reversed(rows)]
    except Exception as exc:
        logger.warning("Could not load tutor session memory | session_id=%s error=%s", session_id, exc)
        return []
    finally:
        db.close()


def _save_turn(
    session_id: str,
    question: str,
    answer: str,
    *,
    content_scope: dict | None = None,
) -> int | None:
    if not session_id:
        return None
    db = SessionLocal()
    try:
        db.add(AgentChatMemory(session_id=session_id, role="user", content=question))
        assistant = AgentChatMemory(
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata_json=(
                {
                    "event_type": "study_answer",
                    "catalog_source": str(content_scope.get("catalog_source") or ""),
                    "curriculum_key": str(content_scope.get("curriculum_key") or ""),
                    "unit_id": str(
                        content_scope.get("planning_unit_id")
                        or content_scope.get("section_id")
                        or ""
                    ),
                    "chapter_slug": str(content_scope.get("chapter_slug") or ""),
                    "primary_topic_id": str(content_scope.get("section_id") or ""),
                    "subject": str(content_scope.get("subject") or ""),
                    "class_level": str(content_scope.get("class_level") or ""),
                }
                if str((content_scope or {}).get("catalog_source") or "").lower()
                in {"planning_manifest", "published"}
                else {}
            ),
        )
        db.add(assistant)
        db.flush()
        interaction_id = int(assistant.id)
        db.commit()
        return interaction_id
    except Exception as exc:
        db.rollback()
        logger.warning("Could not save tutor session memory | session_id=%s error=%s", session_id, exc)
        return None
    finally:
        db.close()


def tutor_agent(request) -> dict:
    """
    Agentic Tutor: Think → Retrieve → Generate → Validate → Respond
    """
    start_time = time.time()

    question = request.question
    section_id = request.section_id
    session_id = request.session_id
    content_scope = getattr(request, "content_scope", None)
    uses_strict_catalog = (
        str((content_scope or {}).get("catalog_source") or "").strip().lower()
        in {"published", "planning_manifest"}
    )

    logger.info(f"[TUTOR] Processing: '{question}' | Section: {section_id}")

    # ===== EMIT: Task Start =====
    event_bus.emit("tutor", "task_start", {
        "task": f"Answer: {question[:60]}...",
        "section": section_id,
        "session": session_id,
        "message": f"Processing question about {section_id}",
    }, session_id=session_id)

    # ===== STEP 1: THINK — Analyze the question =====
    event_bus.emit("tutor", "step", {
        "step": "think",
        "step_num": 1,
        "total_steps": 7,
        "message": "Analyzing question intent and complexity...",
    }, session_id=session_id)

    # ===== STEP 2: ACT — Search knowledge base (markdown) =====
    event_bus.emit("tutor", "tool_call", {
        "step": "retrieve_markdown",
        "step_num": 2,
        "total_steps": 7,
        "tool": "knowledge_search",
        "message": f"Searching markdown knowledge base for: {question[:40]}...",
    }, session_id=session_id)

    search_result = search_knowledge_base(
        section_id=section_id,
        question=question,
        max_paragraphs=5,
        max_chars=3000,
        scope=content_scope,
    )

    if search_result.get("error"):
        not_found = (
            getattr(request, "required_not_found_response", "")
            or "I could not find this in your study material. Please upload or select the correct chapter/data."
        )
        event_bus.emit("tutor", "error", {
            "step": "retrieve_markdown",
            "message": f"Knowledge base lookup failed for section '{section_id}': {search_result['error']}",
        }, session_id=session_id, severity="error")
        return {
            "type": "tutor",
            "answer": not_found,
            "metadata": {"agent": "tutor", "step": "retrieval_failed", "status": "material_not_found"},
        }

    context = search_result["context"]
    basics = search_result.get("basics_context", "")

    event_bus.emit("tutor", "step", {
        "step": "retrieve_markdown_complete",
        "step_num": 2,
        "total_steps": 7,
        "message": f"Retrieved {search_result['paragraphs_found']} paragraphs from markdown",
        "paragraphs": search_result["paragraphs_found"],
        "keywords": search_result["keywords_used"][:5],
    }, session_id=session_id)

    # ===== STEP 2b: ACT — Search knowledge graph (JSON concepts) =====
    event_bus.emit("tutor", "tool_call", {
        "step": "retrieve_graph",
        "step_num": 3,
        "total_steps": 7,
        "tool": "knowledge_graph",
        "message": f"Searching knowledge graph for concepts related to: {question[:40]}...",
    }, session_id=session_id)

    # Attempt to find relevant concepts by keyword or section_id
    concept_data = ""
    concepts_found = []
    # Published catalog sessions are grounded exclusively in the selected
    # versioned material. The bundled graph is a legacy fallback and may
    # represent another chapter/version.
    if not uses_strict_catalog and knowledge_graph.concepts:
        # First try exact match via section_id (which may be a concept_id)
        exact = knowledge_graph.get_concept(section_id)
        if exact:
            concepts_found = [exact]
        else:
            # Otherwise search by keywords extracted from the question
            # Simple keyword extraction (we could use a more sophisticated method later)
            keywords = [w.strip().lower() for w in question.replace("?", "").replace(".", "").split() if len(w) > 3]
            # Use the first meaningful keyword to search
            search_kw = keywords[0] if keywords else section_id
            concepts_found = knowledge_graph.search_by_keyword(search_kw, limit=3)

        if concepts_found:
            # Build a structured text block from the concepts
            blocks = []
            for c in concepts_found:
                block = f"--- CONCEPT: {c['title']} (ID: {c['concept_id']}) ---\n"
                block += f"Definition: {c.get('definition', '')}\n"
                block += f"Explanation: {c.get('core_explanation', '')}\n"
                if c.get("key_points"):
                    block += "Key Points:\n  " + "\n  ".join(c["key_points"]) + "\n"
                if c.get("formulas"):
                    block += "Formulas:\n  " + "\n  ".join(c["formulas"]) + "\n"
                if c.get("examples"):
                    block += "Examples:\n  " + "\n  ".join(c["examples"]) + "\n"
                if c.get("common_mistakes"):
                    mistakes = "\n  ".join([f"Mistake: {m['mistake']} -> Correction: {m['correction']}" for m in c["common_mistakes"]])
                    block += "Common Mistakes:\n  " + mistakes + "\n"
                blocks.append(block)
            concept_data = "\n\n".join(blocks)
            logger.info(f"[TUTOR] Found {len(concepts_found)} concept(s) in knowledge graph: {[c['concept_id'] for c in concepts_found]}")
        else:
            logger.info("[TUTOR] No matching concepts found in knowledge graph")

    event_bus.emit("tutor", "step", {
        "step": "retrieve_graph_complete",
        "step_num": 3,
        "total_steps": 7,
        "message": f"Knowledge graph search yielded {len(concepts_found)} concept(s)",
        "concepts_found": len(concepts_found),
        "concept_ids": [c["concept_id"] for c in concepts_found] if concepts_found else [],
    }, session_id=session_id)

    # ===== STEP 3: OBSERVE — Build context-aware prompt with enriched content =====
    event_bus.emit("tutor", "step", {
        "step": "build_prompt",
        "step_num": 4,
        "total_steps": 7,
        "message": "Building context-aware prompt with markdown, graph concepts, and memory...",
    }, session_id=session_id)

    # Combine markdown context with structured concept data
    enriched_context = context
    if concept_data:
        enriched_context += "\n\n--- STRUCTURED KNOWLEDGE (from official curriculum) ---\n" + concept_data

    # Render the versioned registry prompt. The template carries LaTeX braces,
    # so placeholders are substituted with .replace() rather than .format().
    system_prompt = (
        TUTOR_AGENT_PROMPT
        .replace("{context}", enriched_context)
        .replace("{basics}", basics or "None provided.")
    )

    memory = _load_session_memory(session_id)

    messages = [
        {"role": "system", "content": system_prompt},
    ] + memory + [
        {"role": "user", "content": question},
    ]

    # ===== STEP 4: RESPOND — Generate answer =====
    event_bus.emit("tutor", "tool_call", {
        "step": "generate",
        "step_num": 5,
        "total_steps": 7,
        "tool": "groq_llm",
        "message": f"Generating answer via {MODEL_NAME}...",
        "model": MODEL_NAME,
        "temperature": 0.25,
    }, session_id=session_id)

    try:
        raw_answer = model_gateway.complete(
            role="tutor",
            messages=messages,
            agent_name="tutor_model",
            task="Answer a Study Lab doubt from retrieved section context.",
            student_visible=True,
            safety_tier="final_answer",
            temperature=0.3,
            max_tokens=700,
        ).strip()
    except Exception as e:
        logger.error(f"[TUTOR] LLM error: {e}")
        event_bus.emit("tutor", "error", {
            "step": "generate",
            "message": f"LLM API error: {str(e)}",
            "error": str(e),
        }, session_id=session_id, severity="error")

        event_bus.emit("tutor", "task_complete", {
            "status": "failed",
            "message": "Generation failed",
            "latency_ms": round((time.time() - start_time) * 1000),
        }, session_id=session_id)

        return {
            "type": "tutor",
            "answer": "AI service encountered an error. Please try again.",
            "metadata": {"agent": "tutor", "step": "generation_failed", "error": str(e)},
        }

    # ===== STEP 5: VALIDATE — Format and evaluate =====
    event_bus.emit("tutor", "tool_call", {
        "step": "validate",
        "step_num": 6,
        "total_steps": 7,
        "tool": "chemistry_formatter",
        "message": "Applying chemistry formatting and quality validation...",
    }, session_id=session_id)

    formatted_answer = format_chemistry_output(raw_answer)

    quality = evaluate_answer_quality(
        question=question,
        answer=formatted_answer,
        mode="ask",
        context=context,
    )

    # If quality is too low, retry with more context (only if we haven't already used graph data)
    if not quality["passed"] and search_result["paragraphs_found"] < 3 and not concepts_found:
        event_bus.emit("tutor", "step", {
            "step": "retry",
            "step_num": 6,
            "total_steps": 7,
            "message": f"Quality check failed (score={quality['score']}). Retrying with more markdown context...",
        }, session_id=session_id, severity="warning")

        retry_result = search_knowledge_base(
            section_id=section_id,
            question=question,
            max_paragraphs=8,
            max_chars=4000,
            scope=content_scope,
        )
        if retry_result["context"]:
            retry_context = retry_result["context"]
            if concept_data:
                retry_context += "\n\n--- STRUCTURED KNOWLEDGE ---\n" + concept_data
            retry_prompt = (
                TUTOR_AGENT_PROMPT
                .replace("{context}", retry_context)
                .replace("{basics}", basics or "None provided.")
            )
            retry_messages = [
                {"role": "system", "content": retry_prompt},
                {"role": "user", "content": question},
            ]
            try:
                retry_answer = model_gateway.complete(
                    role="tutor",
                    messages=retry_messages,
                    agent_name="tutor_model",
                    task="Retry a low-quality Study Lab answer with more context.",
                    student_visible=True,
                    safety_tier="final_answer",
                    temperature=0.2,
                    max_tokens=600,
                )
                formatted_answer = format_chemistry_output(retry_answer.strip())
                quality = evaluate_answer_quality(
                    question=question,
                    answer=formatted_answer,
                    mode="ask",
                )
                event_bus.emit("tutor", "step", {
                    "step": "retry_complete",
                    "message": f"Retry quality: score={quality['score']}",
                }, session_id=session_id)
            except Exception as exc:
                # Keep the original answer, but record that the quality retry failed.
                logger.warning("Tutor quality retry failed; keeping first answer | session_id=%s error=%s", session_id, exc)

    # ===== STEP 7: UPDATE MEMORY =====
    interaction_id = _save_turn(
        session_id,
        question,
        formatted_answer,
        content_scope=content_scope,
    )

    latency_ms = round((time.time() - start_time) * 1000)

    # ===== EMIT: Task Complete =====
    event_bus.emit("tutor", "task_complete", {
        "status": "success",
        "message": f"Answer delivered. Quality: {quality['score']}",
        "latency_ms": latency_ms,
        "quality_score": quality["score"],
        "quality_passed": quality["passed"],
        "paragraphs_retrieved": search_result["paragraphs_found"],
        "concepts_from_graph": len(concepts_found),
    }, session_id=session_id)

    logger.info(f"[TUTOR] Complete. Quality: {quality['score']} | Latency: {latency_ms}ms")

    return {
        "type": "tutor",
        "answer": formatted_answer,
        "metadata": {
            "agent": "tutor",
            "quality_score": quality["score"],
            "quality_passed": quality["passed"],
            "paragraphs_retrieved": search_result["paragraphs_found"],
            "concepts_from_graph": len(concepts_found),
            "keywords_used": search_result["keywords_used"][:5],
            "latency_ms": latency_ms,
            "interaction_id": (
                f"study_answer:{interaction_id}" if interaction_id is not None else None
            ),
        },
    }


def reset_tutor_session(session_id: str):
    """Clear tutor memory for a session."""
    db = SessionLocal()
    try:
        db.query(AgentChatMemory).filter(AgentChatMemory.session_id == session_id).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Could not reset tutor session memory | session_id=%s error=%s", session_id, exc)
    finally:
        db.close()
    event_bus.emit("tutor", "state_change", {
        "state": "idle",
        "message": f"Session {session_id} memory cleared",
    }, session_id=session_id)
    logger.info(f"[TUTOR] Session {session_id} cleared.")
