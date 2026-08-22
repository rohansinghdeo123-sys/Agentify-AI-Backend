from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from database import Base


def _utcnow_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# =========================================================
# AGENT MEMORY
# =========================================================
class AgentChatMemory(Base):
    """Stores conversation history for the agentic system."""
    __tablename__ = "agent_chat_memory"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)


# =========================================================
# USER PROGRESS
# =========================================================
class UserProgress(Base):
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    total_tests = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    total_correct = Column(Integer, default=0)
    xp = Column(Integer, default=0)
    streak = Column(Integer, default=0)
    last_active_date = Column(Date, nullable=True)

    focus_score = Column(Float, default=0.0)
    consistency_index = Column(Float, default=0.0)
    learning_efficiency = Column(Float, default=0.0)

    coach_profile = relationship(
        "AICoachProfile",
        back_populates="progress",
        uselist=False,
        primaryjoin="UserProgress.user_id == foreign(AICoachProfile.user_id)",
    )

    @property
    def accuracy(self):
        if self.total_questions == 0:
            return 0.0
        return (self.total_correct / self.total_questions) * 100

    @property
    def level(self):
        return (self.xp // 100) + 1


# =========================================================
# USER PROFILE
# =========================================================
class UserProfile(Base):
    """Canonical account profile populated during post-login onboarding."""

    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True)
    email = Column(String, default="", index=True)
    display_name = Column(String, default="")
    class_level = Column(String, default="", index=True)
    onboarding_completed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)


# =========================================================
# TEST HISTORY
# =========================================================
class TestHistory(Base):
    __tablename__ = "test_history"
    __table_args__ = (
        # Rival rankings scan by date window; weekly stats scan (user, window).
        Index("ix_test_history_date", "date"),
        Index("ix_test_history_user_date", "user_id", "date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    date = Column(Date)
    topic = Column(String)
    score = Column(Integer)
    total_questions = Column(Integer)
    xp_earned = Column(Integer)

    time_spent_seconds = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)
    focus_score = Column(Float, default=0.0)
    session_type = Column(String, default="exam")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    response_latency_ms = Column(Integer, default=0)
    hint_count = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    confidence_before = Column(Float, nullable=True)
    confidence_after = Column(Float, nullable=True)

    details = relationship(
        "SessionDetail",
        back_populates="test",
        uselist=False,
        cascade="all, delete-orphan",
    )


# =========================================================
# SESSION DETAIL
# =========================================================
class SessionDetail(Base):
    __tablename__ = "session_details"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("test_history.id"))

    replay_data = Column(JSON)

    test = relationship("TestHistory", back_populates="details")


# =========================================================
# TOPIC PERFORMANCE
# =========================================================
class TopicPerformance(Base):
    __tablename__ = "topic_performance"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    topic = Column(String, index=True)
    attempts = Column(Integer, default=0)
    correct = Column(Integer, default=0)
    weak = Column(Boolean, default=False)

    last_practiced = Column(DateTime, default=datetime.utcnow)
    avg_time_per_question = Column(Float, default=0.0)
    trend_score = Column(Float, default=0.0)

    @property
    def accuracy(self):
        if self.attempts == 0:
            return 0.0
        return (self.correct / self.attempts) * 100


# =========================================================
# PLANNING LEARNING EVIDENCE
# =========================================================
class PlanningLearningEvent(Base):
    """Append-only proof that a registered Planning unit was studied.

    This records exposure/learning only. It intentionally contains no client-
    writable mastery field; scored Exam analytics remain the mastery authority.
    """

    __tablename__ = "planning_learning_events"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "curriculum_key",
            "unit_id",
            "interaction_id",
            name="uq_planning_learning_event_identity",
        ),
        Index(
            "ix_planning_learning_events_user_curriculum",
            "user_id",
            "curriculum_key",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    curriculum_key = Column(String, nullable=False, index=True)
    unit_id = Column(String, nullable=False, index=True)
    interaction_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False, default="study_answer")
    source_session_id = Column(String, nullable=False, default="")
    created_at = Column(DateTime, default=_utcnow_naive, nullable=False)


# =========================================================
# PERSONAL AI COACH PROFILE
# =========================================================
class AICoachProfile(Base):
    """
    One durable AI coach identity per user.
    This is the coach's stable personality, preferences, and long-term learning state.
    """
    __tablename__ = "ai_coach_profiles"

    id = Column(Integer, primary_key=True, index=True)
    coach_id = Column(String, unique=True, index=True)
    user_id = Column(String, unique=True, index=True)

    coach_name = Column(String, default="Astra")
    coach_tone = Column(String, default="focused_supportive")
    coach_style = Column(String, default="exam_oriented")
    coach_status = Column(String, default="active")

    student_display_name = Column(String, nullable=True)
    target_exam = Column(String, nullable=True)
    target_exam_date = Column(Date, nullable=True)

    preferred_subjects = Column(JSON, default=list)
    weak_topics_snapshot = Column(JSON, default=list)
    strengths_snapshot = Column(JSON, default=list)
    active_goals = Column(JSON, default=list)

    motivation_profile = Column(JSON, default=dict)
    study_preferences = Column(JSON, default=dict)
    last_recommendation = Column(JSON, nullable=True)

    long_term_summary = Column(Text, default="")
    daily_strategy = Column(Text, default="")
    next_best_action = Column(Text, default="Start with a short focused study session.")

    last_learning_cycle_at = Column(DateTime, nullable=True)
    last_interaction_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    progress = relationship(
        "UserProgress",
        back_populates="coach_profile",
        uselist=False,
        primaryjoin="foreign(AICoachProfile.user_id) == UserProgress.user_id",
    )

    memories = relationship(
        "AICoachMemory",
        back_populates="coach",
        cascade="all, delete-orphan",
    )

    interactions = relationship(
        "AICoachInteraction",
        back_populates="coach",
        cascade="all, delete-orphan",
    )


# =========================================================
# PERSONAL AI COACH MEMORY
# =========================================================
class AICoachMemory(Base):
    """
    Stores compact coach memory, not raw endless chat.
    Use this for daily learning, user patterns, study preferences, and weak-topic context.
    """
    __tablename__ = "ai_coach_memories"

    id = Column(Integer, primary_key=True, index=True)
    coach_id = Column(String, ForeignKey("ai_coach_profiles.coach_id"), index=True)
    user_id = Column(String, index=True)

    memory_type = Column(String, default="study_pattern")
    title = Column(String, default="")
    summary = Column(Text, default="")
    importance = Column(Float, default=0.5)
    confidence = Column(Float, default=0.5)

    source = Column(String, default="coach")
    metadata_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    coach = relationship("AICoachProfile", back_populates="memories")


# =========================================================
# PERSONAL AI COACH INTERACTIONS
# =========================================================
class AICoachInteraction(Base):
    """
    Stores coach conversations and advice events per user.
    This gives every user their own coach history without mixing sessions.
    """
    __tablename__ = "ai_coach_interactions"

    id = Column(Integer, primary_key=True, index=True)
    coach_id = Column(String, ForeignKey("ai_coach_profiles.coach_id"), index=True)
    user_id = Column(String, index=True)

    role = Column(String)
    message = Column(Text)
    intent = Column(String, default="general")
    mode = Column(String, default="coach")
    quality_score = Column(Float, default=0.0)

    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    coach = relationship("AICoachProfile", back_populates="interactions")


# =========================================================
# PERSONAL AI COACH DAILY LEARNING SIGNALS
# =========================================================
class AICoachDailySignal(Base):
    """
    One daily snapshot per user that lets the coach improve its advice over time.
    It can be generated after sessions or by a daily background job later.
    """
    __tablename__ = "ai_coach_daily_signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    coach_id = Column(String, index=True)

    signal_date = Column(Date, default=datetime.utcnow)
    sessions_count = Column(Integer, default=0)
    questions_attempted = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    focus_score = Column(Float, default=0.0)
    xp_earned = Column(Integer, default=0)

    weakest_topics = Column(JSON, default=list)
    strongest_topics = Column(JSON, default=list)
    recommendation = Column(Text, default="")
    risk_level = Column(String, default="normal")

    created_at = Column(DateTime, default=datetime.utcnow)


# =========================================================
# DURABLE OBSERVABILITY
# =========================================================
class ObservabilityEvent(Base):
    """
    Durable copy of agent events that were previously only kept in memory.
    Ops can now recover recent activity after restarts and inspect historical failures.
    """
    __tablename__ = "observability_events"

    id = Column(Integer, primary_key=True, index=True)
    event_version = Column(Integer, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    agent_id = Column(String, index=True)
    event_type = Column(String, index=True)
    severity = Column(String, default="info", index=True)
    session_id = Column(String, default="", index=True)
    source = Column(String, default="event_bus")

    summary = Column(Text, default="")
    latency_ms = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    data_json = Column(JSON, default=dict)


class ModelToolTrace(Base):
    """
    Durable trace row for model calls, tool calls, and full coach turns.
    Costs are estimated unless the provider later returns exact token accounting.
    """
    __tablename__ = "model_tool_traces"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user_id = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    turn_id = Column(String, nullable=True, index=True)

    trace_type = Column(String, index=True)  # model, tool, turn
    name = Column(String, index=True)
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    status = Column(String, default="success", index=True)

    latency_ms = Column(Integer, default=0)
    estimated_input_tokens = Column(Integer, default=0)
    estimated_output_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    metadata_json = Column(JSON, default=dict)

class DailyQuotaUsage(Base):
    """Persistent per-user daily usage counters for AI routes."""
    __tablename__ = "daily_quota_usage"

    id = Column(Integer, primary_key=True, index=True)
    quota_key = Column(String, unique=True, index=True, nullable=False)
    user_hash = Column(String, nullable=False, index=True)
    quota_name = Column(String, nullable=False, index=True)
    quota_date = Column(Date, nullable=False, index=True)
    count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =========================================================
# PRODUCTION CONTENT PIPELINE
# =========================================================
class ContentChapter(Base):
    """
    Registry row for one approved/draft study source chapter.
    Raw PDFs remain the source of truth; concepts/chunks are derived layers.
    """
    __tablename__ = "content_chapters"

    id = Column(Integer, primary_key=True, index=True)
    board = Column(String, default="NCERT", index=True)
    class_level = Column(String, default="", index=True)
    subject = Column(String, default="", index=True)
    book_name = Column(String, default="")
    chapter_number = Column(Integer, nullable=True, index=True)
    chapter_name = Column(String, default="", index=True)
    slug = Column(String, unique=True, index=True)

    pdf_path = Column(Text, default="")
    source_hash = Column(String, default="", index=True)
    status = Column(String, default="uploaded", index=True)
    version = Column(String, default="v1")
    # Hash of the PDF that is currently live for students; lets approval bump
    # the version when a re-ingested PDF replaces the published source.
    published_source_hash = Column(String, default="")

    page_count = Column(Integer, default=0)
    extracted_page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    concept_count = Column(Integer, default=0)
    coverage_score = Column(Float, default=0.0)
    extraction_quality = Column(Float, default=0.0)
    validation_report = Column(JSON, default=dict)

    approved_by = Column(String, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)

    pages = relationship(
        "ContentPage",
        back_populates="chapter",
        cascade="all, delete-orphan",
    )
    concepts = relationship(
        "ContentConcept",
        back_populates="chapter",
        cascade="all, delete-orphan",
    )
    chunks = relationship(
        "ContentChunk",
        back_populates="chapter",
        cascade="all, delete-orphan",
    )


class ContentPage(Base):
    """Page-level extracted text with source page mapping."""
    __tablename__ = "content_pages"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("content_chapters.id"), index=True)
    page_number = Column(Integer, index=True)
    text = Column(Text, default="")
    char_count = Column(Integer, default=0)
    extraction_quality = Column(Float, default=0.0)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow_naive)

    chapter = relationship("ContentChapter", back_populates="pages")


class ContentConcept(Base):
    """Structured teaching layer generated from approved source text."""
    __tablename__ = "content_concepts"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("content_chapters.id"), index=True)
    concept_id = Column(String, index=True)
    title = Column(String, default="", index=True)
    definition = Column(Text, default="")
    core_explanation = Column(Text, default="")
    key_points = Column(JSON, default=list)
    examples = Column(JSON, default=list)
    formulas = Column(JSON, default=list)
    properties = Column(JSON, default=list)
    applications = Column(JSON, default=list)
    common_mistakes = Column(JSON, default=list)
    prerequisites = Column(JSON, default=list)
    related_concepts = Column(JSON, default=list)
    learning_objectives = Column(JSON, default=list)
    source_pages = Column(JSON, default=list)
    difficulty_level = Column(Integer, default=1)
    blooms_taxonomy = Column(String, default="")
    typical_exam_weightage = Column(String, default="")
    importance_level = Column(String, default="")
    raw_json = Column(JSON, default=dict)
    validation_issues = Column(JSON, default=list)
    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)

    chapter = relationship("ContentChapter", back_populates="concepts")


class ContentChunk(Base):
    """Searchable RAG chunk derived from extracted source text."""
    __tablename__ = "content_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("content_chapters.id"), index=True)
    chunk_id = Column(String, unique=True, index=True)
    text = Column(Text, default="")
    page_start = Column(Integer, nullable=True, index=True)
    page_end = Column(Integer, nullable=True, index=True)
    section_title = Column(String, default="", index=True)
    token_estimate = Column(Integer, default=0)
    lexical_terms = Column(JSON, default=list)
    embedding = Column(JSON, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow_naive)

    chapter = relationship("ContentChapter", back_populates="chunks")


class ContentIngestionJob(Base):
    """Auditable ingestion/generation/validation job log."""
    __tablename__ = "content_ingestion_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    job_type = Column(String, default="ingest", index=True)
    status = Column(String, default="queued", index=True)
    source_path = Column(Text, default="")
    summary = Column(JSON, default=dict)
    error = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)


# =========================================================
# ADMIN AUDIT
# =========================================================
class AdminAuditLog(Base):
    """Founder/admin action audit trail for protected console operations."""
    __tablename__ = "admin_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=_utcnow_naive, index=True)
    actor_uid = Column(String, default="", index=True)
    actor_email = Column(String, default="", index=True)
    action = Column(String, default="", index=True)
    target_type = Column(String, default="", index=True)
    target_id = Column(String, default="", index=True)
    status = Column(String, default="success", index=True)
    ip_address = Column(String, default="")
    user_agent = Column(Text, default="")
    metadata_json = Column(JSON, default=dict)


# =========================================================
# AGENT RUNTIME
# =========================================================
class AgentRuntimeRun(Base):
    """
    Durable execution envelope for one controlled agent workflow.
    A run can represent a coach turn today and an autonomous mission later.
    """
    __tablename__ = "agent_runtime_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True)
    turn_id = Column(String, nullable=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=True, index=True)

    workflow_name = Column(String, default="study_coach_turn", index=True)
    lead_agent = Column(String, default="lead_coach_orchestrator", index=True)
    mode = Column(String, default="coach")
    intent = Column(String, default="general", index=True)
    status = Column(String, default="running", index=True)

    started_at = Column(DateTime, default=_utcnow_naive, index=True)
    completed_at = Column(DateTime, nullable=True)
    latency_ms = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)
    grounding_status = Column(String, default="not_required", index=True)

    final_answer_excerpt = Column(Text, default="")
    state_json = Column(JSON, default=dict)
    metadata_json = Column(JSON, default=dict)


class AgentRuntimeStep(Base):
    """One ordered backend-controlled step inside an agent runtime run."""
    __tablename__ = "agent_runtime_steps"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    step_name = Column(String, index=True)
    agent_name = Column(String, default="", index=True)
    status = Column(String, default="success", index=True)
    step_order = Column(Integer, default=0)

    started_at = Column(DateTime, default=_utcnow_naive, index=True)
    completed_at = Column(DateTime, nullable=True)
    latency_ms = Column(Integer, default=0)
    input_json = Column(JSON, default=dict)
    output_json = Column(JSON, default=dict)
    error = Column(Text, default="")


class AgentRuntimeMessage(Base):
    """Structured internal message or task packet between agent roles."""
    __tablename__ = "agent_runtime_messages"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    created_at = Column(DateTime, default=_utcnow_naive, index=True)

    sender_agent = Column(String, index=True)
    receiver_agent = Column(String, index=True)
    message_type = Column(String, index=True)
    task = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    required_action = Column(Text, default="")
    evidence_json = Column(JSON, default=dict)
    result_json = Column(JSON, default=dict)


class AgentRuntimeToolCall(Base):
    """Durable record for deterministic tools selected during a run."""
    __tablename__ = "agent_runtime_tool_calls"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    tool_name = Column(String, index=True)
    agent_name = Column(String, default="", index=True)
    status = Column(String, default="success", index=True)

    started_at = Column(DateTime, default=_utcnow_naive, index=True)
    completed_at = Column(DateTime, nullable=True)
    latency_ms = Column(Integer, default=0)
    input_json = Column(JSON, default=dict)
    output_json = Column(JSON, default=dict)
    error = Column(Text, default="")


class AgentRuntimeHandoff(Base):
    """Durable handoff contract between agent roles."""
    __tablename__ = "agent_runtime_handoffs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    created_at = Column(DateTime, default=_utcnow_naive, index=True)

    from_agent = Column(String, index=True)
    to_agent = Column(String, index=True)
    reason = Column(Text, default="")
    status = Column(String, default="requested", index=True)
    input_json = Column(JSON, default=dict)
    result_json = Column(JSON, default=dict)


# =========================================================
# EXAM INTELLIGENCE — UPLOADED PAPERS & PATTERN ANALYSIS
# =========================================================
class UploadedExamPaper(Base):
    """One question paper a student uploaded for pattern analysis.

    Extracted text and analysis are the durable source of truth: the raw file
    on disk may be lost on ephemeral hosts, but stored text lets us re-analyze.
    ``storage_path`` is internal and must never be serialized to clients.
    """
    __tablename__ = "uploaded_exam_papers"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)

    class_level = Column(String, default="", index=True)
    subject = Column(String, default="", index=True)
    chapter_id = Column(Integer, nullable=True, index=True)
    chapter_name = Column(String, default="")
    exam_type = Column(String, default="unknown", index=True)
    paper_title = Column(String, default="")

    file_name = Column(String, default="")
    file_type = Column(String, default="")
    file_size = Column(Integer, default=0)
    storage_path = Column(Text, default="")  # internal only — never serialized

    upload_status = Column(String, default="uploaded", index=True)
    parse_status = Column(String, default="pending", index=True)
    uploaded_at = Column(DateTime, default=_utcnow_naive)
    parsed_at = Column(DateTime, nullable=True)

    extracted_text = Column(Text, default="")
    extraction_confidence = Column(Float, default=0.0)
    extracted_question_count = Column(Integer, default=0)
    analysis_json = Column(JSON, default=dict)
    warnings_json = Column(JSON, default=list)

    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)

    questions = relationship(
        "ExtractedExamQuestion",
        back_populates="paper",
        cascade="all, delete-orphan",
    )


class ExtractedExamQuestion(Base):
    """One question structured out of an uploaded paper by the analyzer agent."""
    __tablename__ = "extracted_exam_questions"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("uploaded_exam_papers.id"), index=True)
    user_id = Column(String, index=True, nullable=False)

    question_number = Column(String, default="")
    section_name = Column(String, default="")
    question_text = Column(Text, default="")
    marks = Column(Float, nullable=True)  # null when not detectable
    question_type = Column(String, default="")
    intent = Column(String, default="")
    difficulty = Column(String, default="")

    class_level = Column(String, default="")
    subject = Column(String, default="")
    chapter_id = Column(Integer, nullable=True)
    chapter_name = Column(String, default="")
    topic = Column(String, default="", index=True)

    concept_tags_json = Column(JSON, default=list)
    expected_answer_style = Column(String, default="")
    confidence_score = Column(Float, default=0.0)
    raw_block = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow_naive)

    paper = relationship("UploadedExamPaper", back_populates="questions")


class ExamPatternAnalysis(Base):
    """Aggregated pattern intelligence across one or more uploaded papers."""
    __tablename__ = "exam_pattern_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)

    class_level = Column(String, default="", index=True)
    subject = Column(String, default="", index=True)
    chapter_id = Column(Integer, nullable=True, index=True)
    chapter_name = Column(String, default="")

    source_paper_ids_json = Column(JSON, default=list)
    total_questions = Column(Integer, default=0)
    total_marks = Column(Float, nullable=True)
    marks_distribution_json = Column(JSON, default=dict)
    question_type_distribution_json = Column(JSON, default=dict)
    chapter_weightage_json = Column(JSON, default=dict)
    topic_frequency_json = Column(JSON, default=dict)
    repeated_concepts_json = Column(JSON, default=list)
    difficulty_distribution_json = Column(JSON, default=dict)
    pattern_summary = Column(Text, default="")
    confidence_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)


class ProbableQuestionSet(Base):
    """Most-probable practice questions derived from observed patterns.

    Output is always framed as 'most probable', never guaranteed (see disclaimer).
    """
    __tablename__ = "probable_question_sets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)

    class_level = Column(String, default="", index=True)
    subject = Column(String, default="", index=True)
    chapter_id = Column(Integer, nullable=True, index=True)
    chapter_name = Column(String, default="")

    source_analysis_ids_json = Column(JSON, default=list)
    generation_mode = Column(String, default="mixed", index=True)
    probable_questions_json = Column(JSON, default=list)
    priority_topics_json = Column(JSON, default=list)
    strategy_summary = Column(Text, default="")
    disclaimer = Column(Text, default="")
    confidence_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=_utcnow_naive)


# =========================================================
# EXAM INTELLIGENCE — WRITTEN ANSWER PRACTICE
# =========================================================
class WrittenPracticeSession(Base):
    """A descriptive (non-MCQ) answer-writing practice session for one student."""
    __tablename__ = "written_practice_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)

    class_level = Column(String, default="")
    subject = Column(String, default="", index=True)
    chapter_id = Column(Integer, nullable=True)
    chapter_name = Column(String, default="")
    topic = Column(String, default="")
    marks_focus = Column(String, default="")  # e.g. "1", "3", "5", "long"

    session_status = Column(String, default="active", index=True)
    started_at = Column(DateTime, default=_utcnow_naive)
    completed_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, default=dict)

    attempts = relationship(
        "WrittenAnswerAttempt",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class WrittenAnswerAttempt(Base):
    """One descriptive question a student attempted within a practice session."""
    __tablename__ = "written_answer_attempts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("written_practice_sessions.id"), index=True)
    user_id = Column(String, index=True, nullable=False)

    question_text = Column(Text, default="")
    question_type = Column(String, default="")
    marks_total = Column(Float, default=0.0)
    student_answer = Column(Text, default="")
    expected_points_json = Column(JSON, default=list)

    # Denormalized scope so weakness recalculation is self-contained.
    class_level = Column(String, default="")
    subject = Column(String, default="")
    chapter_id = Column(Integer, nullable=True)
    chapter_name = Column(String, default="")
    topic = Column(String, default="", index=True)

    submitted_at = Column(DateTime, nullable=True)
    evaluation_status = Column(String, default="pending", index=True)
    created_at = Column(DateTime, default=_utcnow_naive)

    session = relationship("WrittenPracticeSession", back_populates="attempts")
    feedback = relationship(
        "WrittenAnswerFeedback",
        back_populates="attempt",
        uselist=False,
        cascade="all, delete-orphan",
    )


class WrittenAnswerFeedback(Base):
    """Teacher-style rubric evaluation for one written answer attempt."""
    __tablename__ = "written_answer_feedback"

    id = Column(Integer, primary_key=True, index=True)
    attempt_id = Column(Integer, ForeignKey("written_answer_attempts.id"), index=True)
    user_id = Column(String, index=True, nullable=False)

    marks_awarded = Column(Float, default=0.0)
    marks_total = Column(Float, default=0.0)
    score_percentage = Column(Float, default=0.0)

    covered_points_json = Column(JSON, default=list)
    missing_points_json = Column(JSON, default=list)
    incorrect_points_json = Column(JSON, default=list)
    weak_explanation_json = Column(JSON, default=list)
    presentation_feedback = Column(Text, default="")
    teacher_feedback = Column(Text, default="")
    model_answer = Column(Text, default="")
    improve_to_full_marks = Column(Text, default="")
    rubric_scores_json = Column(JSON, default=dict)
    next_question_suggestion = Column(Text, default="")

    created_at = Column(DateTime, default=_utcnow_naive)

    attempt = relationship("WrittenAnswerAttempt", back_populates="feedback")


class RivalPairing(Base):
    """One student's fixed rival assignment for one competition week.

    Pairings are written symmetrically (one row per participant) so both
    students see the same match. Rows are immutable for the week; the weekly
    refresh writes new rows for the next week_start and resolves old ones.
    """

    __tablename__ = "rival_pairings"
    __table_args__ = (
        UniqueConstraint("week_start", "user_id", name="uq_rival_week_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    week_start = Column(Date, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    rival_user_id = Column(String, index=True, nullable=True)

    # Matchmaking snapshot at pairing time (monthly exam performance).
    status = Column(String, default="active", index=True)  # active | unmatched
    match_basis = Column(String, default="monthly_exam")   # monthly_exam | newcomer | fallback
    user_seed_score = Column(Float, default=0.0)
    rival_seed_score = Column(Float, default=0.0)
    user_seed_rank = Column(Integer, nullable=True)
    rival_seed_rank = Column(Integer, nullable=True)

    # End-of-week resolution (idempotent; applied when the next week begins).
    resolved = Column(Boolean, default=False, index=True)
    outcome = Column(String, default="")  # won | lost | tied | unmatched
    my_week_xp = Column(Integer, default=0)
    rival_week_xp = Column(Integer, default=0)
    reward_xp = Column(Integer, default=0)
    resolved_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=_utcnow_naive)


class StudentExamWeakness(Base):
    """Durable per-student weakness signal aggregated from written evaluations."""
    __tablename__ = "student_exam_weaknesses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)

    class_level = Column(String, default="")
    subject = Column(String, default="", index=True)
    chapter_id = Column(Integer, nullable=True)
    chapter_name = Column(String, default="")
    topic = Column(String, default="", index=True)

    weakness_type = Column(String, default="", index=True)
    weakness_summary = Column(Text, default="")
    evidence_json = Column(JSON, default=list)
    frequency_count = Column(Integer, default=1)
    last_seen_at = Column(DateTime, default=_utcnow_naive)
    improvement_suggestion = Column(Text, default="")

    created_at = Column(DateTime, default=_utcnow_naive)
    updated_at = Column(DateTime, default=_utcnow_naive, onupdate=_utcnow_naive)
