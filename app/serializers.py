"""Pure serialization and formatting helpers (no database access)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.request_models import CoachConversationPatch
from app.security import session_id_belongs_to_user
from models import (
    AICoachDailySignal,
    AICoachInteraction,
    AICoachMemory,
    AICoachProfile,
    AdminAuditLog,
    ModelToolTrace,
    TestHistory,
    UserProfile,
    UserProgress,
)


def normalize_topic(topic: str) -> str:
    cleaned = (topic or "unknown").strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    aliases = {
        "basic_concepts_of_chemistry": "matter_definition",
        "basic_concept_of_chemistry": "matter_definition",
        "matter": "matter_definition",
        "hydrocarbon": "alkanes",
        "hydrocarbons": "alkanes",
        "aromatic_hydrocarbons": "aromatics",
    }
    return aliases.get(cleaned, cleaned)


def progress_payload(user: UserProgress) -> Dict[str, Any]:
    return {
        "user_id": user.user_id,
        "total_tests": int(user.total_tests or 0),
        "total_questions": int(user.total_questions or 0),
        "total_correct": int(user.total_correct or 0),
        "xp": int(user.xp or 0),
        "streak": int(user.streak or 0),
        "level": int(user.level),
        "accuracy": round(float(user.accuracy), 1),
        "focus_score": round(float(user.focus_score or 0), 1),
        "consistency_index": round(float(user.consistency_index or 0), 1),
        "learning_efficiency": round(float(user.learning_efficiency or 0), 1),
    }


def utc_naive(value: Optional[datetime]) -> Optional[datetime]:
    if not value:
        return None
    if value.tzinfo:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def format_test_session(test: TestHistory, class_level: str = "") -> Dict[str, Any]:
    questions = int(test.total_questions or 0)
    correct = int(test.score or 0)
    seconds = int(test.time_spent_seconds or 0)
    duration_minutes = round(seconds / 60)

    if seconds > 0 and duration_minutes == 0:
        duration_minutes = 1

    accuracy = round((correct / questions) * 100) if questions else 0
    session_date = test.date.isoformat() if test.date else None
    timestamp = (
        datetime.combine(test.date, datetime.min.time()).isoformat()
        if test.date
        else None
    )
    started_at = getattr(test, "started_at", None)
    completed_at = getattr(test, "completed_at", None)
    confidence_before = getattr(test, "confidence_before", None)
    confidence_after = getattr(test, "confidence_after", None)
    confidence_change = (
        round(float(confidence_after) - float(confidence_before), 1)
        if confidence_before is not None and confidence_after is not None
        else None
    )
    replay_data = test.details.replay_data if getattr(test, "details", None) else {}
    replay_question_count = 0
    if isinstance(replay_data, dict):
        questions_payload = replay_data.get("questions")
        if isinstance(questions_payload, list):
            replay_question_count = len(questions_payload)

    return {
        "id": str(test.id),
        "subject": "Chemistry",
        "class_level": class_level,
        "topic": test.topic or "unknown",
        "duration": duration_minutes,
        "questions": questions,
        "correct": correct,
        "xp": int(test.xp_earned or 0),
        "focusScore": round(float(test.focus_score or 0)),
        "date": session_date,
        "timestamp": timestamp,
        "status": "completed",
        "performance": accuracy,
        "time_spent_seconds": seconds,
        "accuracy_rate": round(float(test.accuracy_rate or accuracy), 1),
        "focus_score": round(float(test.focus_score or 0), 1),
        "session_type": test.session_type or "exam",
        "started_at": iso_or_none(started_at),
        "completed_at": iso_or_none(completed_at),
        "startedAt": iso_or_none(started_at),
        "completedAt": iso_or_none(completed_at),
        "response_latency_ms": int(getattr(test, "response_latency_ms", 0) or 0),
        "responseLatencyMs": int(getattr(test, "response_latency_ms", 0) or 0),
        "hint_count": int(getattr(test, "hint_count", 0) or 0),
        "hintCount": int(getattr(test, "hint_count", 0) or 0),
        "retry_count": int(getattr(test, "retry_count", 0) or 0),
        "retryCount": int(getattr(test, "retry_count", 0) or 0),
        "confidence_before": confidence_before,
        "confidence_after": confidence_after,
        "confidenceBefore": confidence_before,
        "confidenceAfter": confidence_after,
        "confidence_change": confidence_change,
        "confidenceChange": confidence_change,
        "has_replay": bool(replay_question_count),
        "replay_question_count": replay_question_count,
        "replay_data": replay_data or {},
    }


def serialize_coach_profile(coach: AICoachProfile) -> Dict[str, Any]:
    return {
        "coach_id": coach.coach_id,
        "user_id": coach.user_id,
        "coach_name": coach.coach_name,
        "coach_tone": coach.coach_tone,
        "coach_style": coach.coach_style,
        "coach_status": coach.coach_status,
        "student_display_name": coach.student_display_name,
        "target_exam": coach.target_exam,
        "target_exam_date": coach.target_exam_date,
        "preferred_subjects": coach.preferred_subjects or [],
        "weak_topics_snapshot": coach.weak_topics_snapshot or [],
        "strengths_snapshot": coach.strengths_snapshot or [],
        "active_goals": coach.active_goals or [],
        "motivation_profile": coach.motivation_profile or {},
        "study_preferences": coach.study_preferences or {},
        "long_term_summary": coach.long_term_summary or "",
        "daily_strategy": coach.daily_strategy or "",
        "next_best_action": coach.next_best_action or "",
        "last_learning_cycle_at": coach.last_learning_cycle_at,
        "last_interaction_at": coach.last_interaction_at,
        "created_at": coach.created_at,
        "updated_at": coach.updated_at,
    }


def serialize_coach_memory(memory: AICoachMemory) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "coach_id": memory.coach_id,
        "user_id": memory.user_id,
        "memory_type": memory.memory_type,
        "title": memory.title,
        "summary": memory.summary,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "source": memory.source,
        "metadata_json": memory.metadata_json or {},
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
    }


def serialize_daily_signal(signal: Optional[AICoachDailySignal]) -> Optional[Dict[str, Any]]:
    if not signal:
        return None

    return {
        "user_id": signal.user_id,
        "coach_id": signal.coach_id,
        "signal_date": signal.signal_date,
        "sessions_count": signal.sessions_count,
        "questions_attempted": signal.questions_attempted,
        "accuracy": signal.accuracy,
        "focus_score": signal.focus_score,
        "xp_earned": signal.xp_earned,
        "weakest_topics": signal.weakest_topics or [],
        "strongest_topics": signal.strongest_topics or [],
        "recommendation": signal.recommendation,
        "risk_level": signal.risk_level,
    }


# ================= COACH CONVERSATION SERIALIZATION =================
def interaction_metadata(row: AICoachInteraction) -> Dict[str, Any]:
    return row.metadata_json if isinstance(row.metadata_json, dict) else {}


def interaction_session_id(row: AICoachInteraction) -> str:
    return str(interaction_metadata(row).get("session_id") or "").strip()


def conversation_title_from(message: str) -> str:
    compact = " ".join(str(message or "").split())
    if not compact:
        return "New study chat"
    return compact[:54] + ("..." if len(compact) > 54 else "")


def serialize_coach_interaction_message(row: AICoachInteraction) -> Dict[str, Any]:
    metadata = interaction_metadata(row)
    role = "coach" if row.role == "assistant" else "user"
    payload: Dict[str, Any] = {
        "role": role,
        "content": row.message or "",
        "timestamp": row.created_at.strftime("%H:%M") if row.created_at else "",
    }
    if role == "coach":
        answer_blocks = metadata.get("answer_blocks")
        sources = metadata.get("sources")
        if isinstance(answer_blocks, list):
            payload["blocks"] = answer_blocks
        if isinstance(sources, dict):
            payload["sources"] = sources
        orchestration = metadata.get("orchestration") if isinstance(metadata.get("orchestration"), dict) else {}
        if "socratic" in orchestration:
            payload["socratic"] = bool(orchestration.get("socratic"))
    return payload


def conversation_metadata(rows: List[AICoachInteraction]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for row in rows:
        metadata = interaction_metadata(row)
        for key in ("conversation_title", "conversation_pinned", "conversation_archived", "conversation_title_locked"):
            if key in metadata:
                merged[key] = metadata[key]
    return merged


def serialize_coach_conversation(session_id: str, rows: List[AICoachInteraction]) -> Dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda item: item.id or 0)
    metadata = conversation_metadata(sorted_rows)
    first_user = next((row for row in sorted_rows if row.role == "user" and row.message), sorted_rows[0])
    last_row = sorted_rows[-1]
    title = str(metadata.get("conversation_title") or conversation_title_from(first_user.message))
    last_metadata = interaction_metadata(last_row)
    learning_context = last_metadata.get("learning_context") if isinstance(last_metadata.get("learning_context"), dict) else {}
    payload = {
        "id": session_id.replace(f"coach-{last_row.user_id}-", "", 1) if session_id.startswith(f"coach-{last_row.user_id}-") else session_id,
        "sessionId": session_id,
        "title": title,
        "updatedAt": last_row.created_at.isoformat() if last_row.created_at else datetime.utcnow().isoformat(),
        "chapter": str(learning_context.get("selected_chapter") or "Open tutor"),
        "topic": str(learning_context.get("selected_topic") or "Any subject"),
        "messages": [serialize_coach_interaction_message(row) for row in sorted_rows],
        "pinned": bool(metadata.get("conversation_pinned")),
        "archived": bool(metadata.get("conversation_archived")),
        "titleLocked": bool(metadata.get("conversation_title_locked")),
        "messageCount": len(sorted_rows),
    }
    catalog_source = str(learning_context.get("catalog_source") or "").strip().lower()
    if (
        learning_context.get("scope") == "selected_study_material_only"
        or catalog_source == "planning_manifest"
    ):
        chapter_id = str(learning_context.get("selected_chapter_id") or "").strip()
        topic_id = str(
            learning_context.get("selected_topic_id")
            or learning_context.get("section_id")
            or ""
        ).strip()
        chapter_label = str(
            learning_context.get("selected_chapter")
            or learning_context.get("chapter")
            or chapter_id
        ).strip()
        topic_label = str(
            learning_context.get("selected_topic")
            or learning_context.get("topic")
            or topic_id
        ).strip()
        if chapter_id and topic_id:
            payload["scope"] = {
                "source": "syllabus",
                "catalogSource": catalog_source or "starter",
                "subject": str(learning_context.get("selected_subject") or "").strip(),
                "classLevel": str(learning_context.get("class_level") or "").strip(),
                "chapterId": chapter_id,
                "chapterLabel": chapter_label,
                "topicId": topic_id,
                "topicLabel": topic_label,
            }
    return payload


def group_conversation_rows(rows: List[AICoachInteraction]) -> Dict[str, List[AICoachInteraction]]:
    from collections import defaultdict

    grouped: Dict[str, List[AICoachInteraction]] = defaultdict(list)
    for row in rows:
        grouped[interaction_session_id(row)].append(row)
    return grouped


def session_id_from_conversation_id(user_id: str, conversation_id: str) -> str:
    raw = str(conversation_id or "").strip()
    if session_id_belongs_to_user(raw, user_id):
        return raw
    return f"coach-{user_id}-{raw}"


def apply_conversation_patch(row: AICoachInteraction, patch: CoachConversationPatch) -> None:
    metadata = dict(interaction_metadata(row))
    if patch.title is not None:
        metadata["conversation_title"] = patch.title.strip() or "New study chat"
    if patch.pinned is not None:
        metadata["conversation_pinned"] = bool(patch.pinned)
    if patch.archived is not None:
        metadata["conversation_archived"] = bool(patch.archived)
    if patch.titleLocked is not None:
        metadata["conversation_title_locked"] = bool(patch.titleLocked)
    row.metadata_json = metadata


# ================= ADMIN SERIALIZATION =================
def serialize_audit_log(row: AdminAuditLog) -> Dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "actor_uid": row.actor_uid or "",
        "actor_email": row.actor_email or "",
        "action": row.action or "",
        "target_type": row.target_type or "",
        "target_id": row.target_id or "",
        "status": row.status or "",
        "metadata": row.metadata_json or {},
    }


def trace_payload(row: ModelToolTrace) -> Dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "turn_id": row.turn_id or "",
        "session_id": row.session_id or "",
        "user_id": row.user_id or "",
        "trace_type": row.trace_type or "",
        "name": row.name or "",
        "provider": row.provider or "",
        "model": row.model or "",
        "status": row.status or "",
        "latency_ms": int(row.latency_ms or 0),
        "estimated_input_tokens": int(row.estimated_input_tokens or 0),
        "estimated_output_tokens": int(row.estimated_output_tokens or 0),
        "estimated_cost_usd": float(row.estimated_cost_usd or 0.0),
        "metadata": row.metadata_json or {},
    }


def student_payload(row: UserProgress, profile: Optional[UserProfile] = None) -> Dict[str, Any]:
    return {
        "user_id": row.user_id,
        "display_name": profile.display_name if profile else "",
        "class_level": profile.class_level if profile else "",
        "onboarding_completed": bool(profile.onboarding_completed) if profile else False,
        "xp": int(row.xp or 0),
        "level": int(row.level or 1),
        "streak": int(row.streak or 0),
        "total_tests": int(row.total_tests or 0),
        "total_questions": int(row.total_questions or 0),
        "total_correct": int(row.total_correct or 0),
        "accuracy": round(float(row.accuracy or 0.0), 1),
        "last_active_date": row.last_active_date.isoformat() if row.last_active_date else None,
        "focus_score": round(float(row.focus_score or 0.0), 1),
        "consistency_index": round(float(row.consistency_index or 0.0), 1),
        "learning_efficiency": round(float(row.learning_efficiency or 0.0), 1),
    }
