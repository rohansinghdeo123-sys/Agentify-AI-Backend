"""Request bodies for endpoints that are not defined in ``schemas``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class SectionAIRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2500)
    section_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=220)
    mode: str = Field(default="revision", max_length=40)
    difficulty: str = Field(default="medium", max_length=40)
    subject: Optional[str] = Field(default=None, max_length=120)
    chapter: Optional[str] = Field(default=None, max_length=180)
    topic: Optional[str] = Field(default=None, max_length=180)
    class_level: Optional[str] = Field(default=None, max_length=64)
    catalog_source: Optional[
        Literal["published", "planning_manifest", "builtin", "starter"]
    ] = None
    system_guardrail: Optional[str] = Field(default=None, max_length=8000)
    strict_grounding: bool = False
    retrieval_required: bool = False
    fallback_to_general_knowledge: bool = True
    required_not_found_response: Optional[str] = Field(default=None, max_length=500)


class PlanningLearningEventRequest(BaseModel):
    interaction_id: str = Field(min_length=3, max_length=220)
    event_type: Literal["study_answer"] = "study_answer"


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=160)
    user_id: Optional[str] = None


class CoachConversationPatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=72)
    pinned: Optional[bool] = None
    archived: Optional[bool] = None
    titleLocked: Optional[bool] = None


class AgentCommandRequest(BaseModel):
    agent_id: str
    command: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class AgentMessageRequest(BaseModel):
    agent_id: str
    message: str
    section_id: str = "alkanes"
    session_id: str = "admin"
    mode: Optional[str] = None
    system_message: Optional[str] = None


class ContentIngestFolderRequest(BaseModel):
    root_path: Optional[str] = Field(default=None, max_length=600)
    replace_existing_extraction: bool = True
    run_in_background: bool = True


class ContentConceptImportRequest(BaseModel):
    concepts: Any
    replace_existing: bool = True


class ContentGenerateConceptsRequest(BaseModel):
    replace_existing: bool = True
    # Smaller batches mean fewer concepts per LLM response, which keeps the JSON
    # output well under the token cap and avoids mid-array truncation.
    max_batch_chars: int = Field(default=6000, ge=2500, le=16000)
    run_in_background: bool = True


class ContentEmbedRequest(BaseModel):
    chapter_id: Optional[int] = Field(default=None, ge=1)
    run_in_background: bool = True


class AdminAuditRequest(BaseModel):
    action: str = Field(min_length=2, max_length=120)
    target_type: str = Field(default="console", max_length=80)
    target_id: str = Field(default="", max_length=220)
    status: str = Field(default="success", max_length=40)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdminActionRequest(BaseModel):
    action: str
    target_type: str = ""
    target_id: str = ""
    confirmed: bool = False
    payload: Dict[str, Any] = Field(default_factory=dict)


class GenerateMCQRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=180)
    section_id: Optional[str] = Field(default=None, max_length=160)
    session_id: str = Field(default="exam-session", min_length=1, max_length=220)
    difficulty: str = Field(default="medium", max_length=40)
    count: int = Field(default=5, ge=1, le=10)
    subject: Optional[str] = Field(default=None, max_length=120)
    chapter: Optional[str] = Field(default=None, max_length=180)
    class_level: Optional[str] = Field(default=None, max_length=64)
    catalog_source: Optional[
        Literal["published", "planning_manifest", "builtin", "starter"]
    ] = None
    system_guardrail: Optional[str] = Field(default=None, max_length=8000)
    strict_grounding: bool = False
    retrieval_required: bool = False
    fallback_to_general_knowledge: bool = True
    required_not_found_response: Optional[str] = Field(default=None, max_length=500)
    include_source: bool = False
    require_four_options: bool = True
    require_explanation: bool = True


class GenerateProbableRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=180)
    section_id: Optional[str] = Field(default=None, max_length=160)
    session_id: str = Field(default="probable-session", min_length=1, max_length=220)
    difficulty: str = Field(default="medium", max_length=40)
    subject: Optional[str] = Field(default=None, max_length=120)
    chapter: Optional[str] = Field(default=None, max_length=180)
    class_level: Optional[str] = Field(default=None, max_length=64)
    catalog_source: Optional[
        Literal["published", "planning_manifest", "builtin", "starter"]
    ] = None
    system_guardrail: Optional[str] = Field(default=None, max_length=8000)
    strict_grounding: bool = False
    retrieval_required: bool = False
    fallback_to_general_knowledge: bool = True
    required_not_found_response: Optional[str] = Field(default=None, max_length=500)
    include_source: bool = False


class ArtifactGenerateRequest(BaseModel):
    section_id: str = Field(min_length=1, max_length=160)
    topic: Optional[str] = Field(default=None, max_length=180)
    artifact_type: str = Field(default="auto", max_length=50)
    subject: Optional[str] = Field(default=None, max_length=120)
    chapter: Optional[str] = Field(default=None, max_length=180)
    class_level: Optional[str] = Field(default=None, max_length=64)
    catalog_source: Optional[
        Literal["published", "planning_manifest", "builtin", "starter"]
    ] = None
    system_guardrail: Optional[str] = Field(default=None, max_length=8000)
    strict_grounding: bool = False
    retrieval_required: bool = False
    fallback_to_general_knowledge: bool = True
    required_not_found_response: Optional[str] = Field(default=None, max_length=500)


class SubmitSessionRequest(BaseModel):
    user_id: str
    topic: str
    subject: str = "Chemistry"
    score: int = Field(ge=0)
    total_questions: int = Field(gt=0)
    xp_earned: Optional[int] = None
    time_spent_seconds: int = Field(default=0, ge=0)
    focus_score: float = Field(default=0.0, ge=0, le=100)
    session_type: str = "exam"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_latency_ms: int = Field(default=0, ge=0)
    hint_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    confidence_before: Optional[float] = Field(default=None, ge=0, le=100)
    confidence_after: Optional[float] = Field(default=None, ge=0, le=100)
    replay_data: Optional[Dict[str, Any]] = None
