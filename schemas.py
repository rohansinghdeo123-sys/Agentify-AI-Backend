from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


ALLOWED_CLASS_LEVELS = {
    "",
    "Class 6",
    "Class 7",
    "Class 8",
    "Class 9",
    "Class 10",
    "Class 11",
    "Class 12",
    "Other",
}


# =========================================================
# USER PROFILE SCHEMAS
# =========================================================
class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=2, max_length=80)
    class_level: Optional[str] = Field(default=None, max_length=32)
    onboarding_completed: Optional[bool] = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if len(cleaned) < 2 or not any(character.isalpha() for character in cleaned):
            raise ValueError("Enter a valid name")
        return cleaned

    @field_validator("class_level")
    @classmethod
    def validate_class_level(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if cleaned not in ALLOWED_CLASS_LEVELS:
            raise ValueError("Select a valid class level")
        return cleaned


class UserProfileResponse(BaseModel):
    user_id: str
    email: str = ""
    display_name: str = ""
    class_level: str = ""
    onboarding_completed: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# =========================================================
# PROGRESS SCHEMAS
# =========================================================
class ProgressBase(BaseModel):
    user_id: str
    total_tests: int = Field(ge=0, default=0)
    total_questions: int = Field(ge=0, default=0)
    total_correct: int = Field(ge=0, default=0)
    xp: int = Field(ge=0, default=0)
    streak: int = Field(ge=0, default=0)


class ProgressUpdate(ProgressBase):
    pass


class ProgressResponse(ProgressBase):
    level: int = 1
    accuracy: float = 0.0
    focus_score: float = 0.0
    consistency_index: float = 0.0
    learning_efficiency: float = 0.0

    model_config = {"from_attributes": True}


# =========================================================
# TEST HISTORY & SESSION SCHEMAS
# =========================================================
class TestHistoryCreate(BaseModel):
    user_id: str
    topic: str
    score: int = Field(ge=0)
    total_questions: int = Field(ge=1)
    xp_earned: int = Field(ge=0)
    time_spent_seconds: int = 0
    focus_score: float = 0.0
    session_type: str = "exam"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_latency_ms: int = Field(default=0, ge=0)
    hint_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    confidence_before: Optional[float] = Field(default=None, ge=0, le=100)
    confidence_after: Optional[float] = Field(default=None, ge=0, le=100)
    replay_data: Optional[Dict[str, Any]] = None


class TestHistoryResponse(BaseModel):
    id: int
    date: date
    topic: Optional[str] = None
    score: int
    total_questions: int
    xp_earned: int
    time_spent_seconds: int
    accuracy_rate: float
    focus_score: float
    session_type: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_latency_ms: int = 0
    hint_count: int = 0
    retry_count: int = 0
    confidence_before: Optional[float] = None
    confidence_after: Optional[float] = None

    model_config = {"from_attributes": True}


class SessionReplayResponse(BaseModel):
    id: int
    topic: str
    date: date
    replay_data: Dict[str, Any]

    model_config = {"from_attributes": True}


# =========================================================
# TOPIC PERFORMANCE SCHEMAS
# =========================================================
class TopicPerformanceResponse(BaseModel):
    topic: str
    attempts: int
    correct: int
    accuracy: float
    weak: bool
    last_practiced: datetime
    avg_time_per_question: float
    trend_score: float

    model_config = {"from_attributes": True}


# =========================================================
# ADVANCED ANALYTICS SCHEMAS
# =========================================================
class AnalyticsInsight(BaseModel):
    type: str
    message: str
    severity: str
    action_label: Optional[str] = None
    action_trigger: Optional[str] = None


class AdvancedAnalyticsResponse(BaseModel):
    summary: Dict[str, Any]
    topic_heatmap: List[Dict[str, Any]]
    performance_trends: List[Dict[str, Any]]
    weak_areas: List[Dict[str, Any]]
    insights: List[AnalyticsInsight]
    cognitive_metrics: Dict[str, float]
    predictive_stats: Dict[str, Any]


# =========================================================
# AGENT AI SCHEMAS
# =========================================================
class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    section_id: str = ""
    session_id: str
    mode: str = "revision"
    difficulty: str = "medium"


class AgentResponse(BaseModel):
    answer: str
    tools_used: List[str] = []
    session_id: str


class AgentChatMemoryBase(BaseModel):
    session_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata_json: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


# =========================================================
# PERSONAL AI COACH SCHEMAS
# =========================================================
class CoachBootstrapRequest(BaseModel):
    user_id: str
    student_display_name: Optional[str] = None
    preferred_subjects: List[str] = Field(default_factory=list)
    target_exam: Optional[str] = None
    target_exam_date: Optional[date] = None


class CoachProfileResponse(BaseModel):
    coach_id: str
    user_id: str
    coach_name: str
    coach_tone: str
    coach_style: str
    coach_status: str
    student_display_name: Optional[str] = None
    target_exam: Optional[str] = None
    target_exam_date: Optional[date] = None
    preferred_subjects: List[str] = Field(default_factory=list)
    weak_topics_snapshot: List[Dict[str, Any]] = Field(default_factory=list)
    strengths_snapshot: List[Dict[str, Any]] = Field(default_factory=list)
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    motivation_profile: Dict[str, Any] = Field(default_factory=dict)
    study_preferences: Dict[str, Any] = Field(default_factory=dict)
    long_term_summary: str = ""
    daily_strategy: str = ""
    next_best_action: str = ""
    last_learning_cycle_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CoachMemoryResponse(BaseModel):
    id: int
    coach_id: str
    user_id: str
    memory_type: str
    title: str
    summary: str
    importance: float
    confidence: float
    source: str
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CoachAttachmentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    mime_type: str = Field(min_length=1, max_length=80)
    size_bytes: int = Field(default=0, ge=0, le=6_500_000)
    data_url: str = Field(min_length=1, max_length=8_800_000)


class CoachChatRequest(BaseModel):
    user_id: str
    message: str = Field(min_length=1, max_length=2500)
    original_message: Optional[str] = None
    grounding_context_prompt: Optional[str] = None
    mode: str = "coach"
    intent: str = "general"
    subject: Optional[str] = None
    chapter: Optional[str] = None
    topic: Optional[str] = None
    section_id: Optional[str] = None
    session_id: Optional[str] = None
    mentor_directive: Optional[str] = None
    system_guardrail: Optional[str] = None
    strict_grounding: bool = False
    retrieval_required: bool = False
    fallback_to_general_knowledge: bool = True
    required_not_found_response: Optional[str] = None
    student_state: Dict[str, Any] = Field(default_factory=dict)
    adaptive_strategy: Dict[str, Any] = Field(default_factory=dict)
    learning_context: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[CoachAttachmentRequest] = Field(default_factory=list, max_length=5)
    direct_answer: bool = False
    socratic_mode: bool = True


class CoachChatResponse(BaseModel):
    coach_id: str
    coach_name: str
    answer: str
    next_best_action: str
    daily_strategy: str
    memory_used: List[Dict[str, Any]] = Field(default_factory=list)
    analytics_snapshot: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoachDailySignalResponse(BaseModel):
    user_id: str
    coach_id: str
    signal_date: date
    sessions_count: int
    questions_attempted: int
    accuracy: float
    focus_score: float
    xp_earned: int
    weakest_topics: List[Dict[str, Any]] = Field(default_factory=list)
    strongest_topics: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: str
    risk_level: str

    model_config = {"from_attributes": True}


class CoachDashboardResponse(BaseModel):
    profile: CoachProfileResponse
    memories: List[CoachMemoryResponse] = Field(default_factory=list)
    daily_signal: Optional[CoachDailySignalResponse] = None
    analytics_snapshot: Dict[str, Any] = Field(default_factory=dict)


class AutonomousStudyRequest(BaseModel):
    current_chapter: str = Field(min_length=1, max_length=240)
    subject: str = Field(min_length=1, max_length=120)
    class_level: str = Field(min_length=1, max_length=64)
    chapter_proficiency: Literal[
        "new_to_it",
        "know_a_little",
        "know_the_basics",
        "mostly_confident",
    ] = "know_a_little"
    # A student's choice is distinct from duration supplied by an existing
    # study session.  When both are absent, Planning chooses a focused route of
    # at most 30 minutes; ``no_limit`` still remains content-sized.
    study_time_today: Optional[
        Literal["15", "30", "60", "120_plus", "no_limit"]
    ] = None
    session_duration_minutes: Optional[int] = Field(default=None, ge=15, le=120)

    model_config = {"extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def migrate_retired_planning_choices(cls, value: Any) -> Any:
        """Load historical clients without keeping retired controls alive.

        ``Fast Track`` is the only legacy value that needs semantic migration:
        it now resolves to the single Quick Revision behaviour represented by
        ``mostly_confident``. Old knowledge values receive their closest new
        proficiency. Numeric historical time choices are normalized into the
        current presets. Exam-target and style fields remain ignored.
        """
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        raw_time = payload.get("study_time_today")
        if raw_time is not None and raw_time != "":
            if isinstance(raw_time, bool):
                # Let the field validator return the normal validation error.
                pass
            elif isinstance(raw_time, (int, float)) and float(raw_time).is_integer():
                numeric_time = int(raw_time)
                payload["study_time_today"] = (
                    "120_plus" if numeric_time == 120 else str(numeric_time)
                )
        if payload.get("chapter_proficiency") in {None, ""}:
            legacy_goal = str(payload.get("learning_goal") or "").strip().lower()
            legacy_goal = legacy_goal.replace(" ", "_").replace("-", "_")
            legacy_knowledge = str(payload.get("current_knowledge") or "").strip().lower()
            legacy_knowledge = legacy_knowledge.replace(" ", "_").replace("-", "_")
            if legacy_goal in {"fast_track", "quick_revision"}:
                payload["chapter_proficiency"] = "mostly_confident"
            else:
                payload["chapter_proficiency"] = {
                    "new": "new_to_it",
                    "weak_basics": "know_a_little",
                    "some_idea": "know_a_little",
                    "know_basics": "know_the_basics",
                }.get(legacy_knowledge, "know_a_little")
        return payload

    @field_validator("study_time_today", mode="before")
    @classmethod
    def normalize_study_time_today(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            return {
                "15_min": "15",
                "15_mins": "15",
                "15_minute": "15",
                "15_minutes": "15",
                "30_min": "30",
                "30_mins": "30",
                "30_minute": "30",
                "30_minutes": "30",
                "1_hour": "60",
                "1_hr": "60",
                "60_min": "60",
                "60_mins": "60",
                "60_minutes": "60",
                "120": "120_plus",
                "120+": "120_plus",
                "2_hours": "120_plus",
                "2+_hours": "120_plus",
                "2_plus_hours": "120_plus",
                "2_hours+": "120_plus",
                "unlimited": "no_limit",
            }.get(normalized, normalized)
        return value

    @field_validator("chapter_proficiency", mode="before")
    @classmethod
    def normalize_chapter_proficiency(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        return {
            "new": "new_to_it",
            "some_idea": "know_a_little",
            "know_basics": "know_the_basics",
            "fast_track": "mostly_confident",
            "quick_revision": "mostly_confident",
        }.get(normalized, normalized)

    @field_validator("current_chapter")
    @classmethod
    def normalize_planning_chapter(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Select a chapter before generating a plan")
        return normalized

    @field_validator("subject", "class_level")
    @classmethod
    def normalize_planning_scope_label(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Select a class and subject before generating a plan")
        return normalized

class PlanningFocusArea(BaseModel):
    focus_area_id: str = Field(min_length=1, max_length=160)
    unit_ids: List[str] = Field(min_length=1, max_length=16)
    # Rollout alias for saved clients that only understand one unit ID.
    unit_id: str = Field(min_length=1, max_length=240)
    unit_titles: List[str] = Field(min_length=1, max_length=16)
    title: str = Field(min_length=1, max_length=240)
    subtopics: List[str] = Field(min_length=1, max_length=4)
    focus_level: Literal["high", "medium", "light"]
    reason: str = Field(min_length=1, max_length=180)
    guidance: str = Field(min_length=1, max_length=180)


class PlanningGuidanceStep(BaseModel):
    sequence: int = Field(ge=1, le=5)
    title: str = Field(min_length=1, max_length=80)
    instruction: str = Field(min_length=1, max_length=220)
    focus_unit_ids: List[str] = Field(min_length=1, max_length=80)


class PlanningCoverage(BaseModel):
    status: Literal["complete"] = "complete"
    included_unit_ids: List[str] = Field(min_length=1, max_length=80)
    unit_count: int = Field(ge=1, le=80)


class PlanningEstimatedMinutes(BaseModel):
    min: int = Field(ge=5, le=180)
    max: int = Field(ge=5, le=180)


class PlanningNcertSection(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=180)


class PlanningNcertSubtopic(BaseModel):
    id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=240)
    section_id: str = Field(min_length=1, max_length=80)


class PlanningCurriculumMetadata(BaseModel):
    key: str = Field(min_length=1, max_length=180)
    source: str = Field(min_length=1, max_length=240)
    edition: str = Field(min_length=1, max_length=120)
    chapter_number: int = Field(ge=1, le=200)
    content_order_locked: Literal[True] = True
    source_reference: Dict[str, Any] = Field(default_factory=dict)


class PlanningConcept(BaseModel):
    id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=240)
    status: Literal[
        "not_started",
        "recommended",
        "learning",
        "practising",
        "needs_review",
        "mastered",
    ]
    evidence_count: int = Field(ge=0)


class PlanningLearningUnit(BaseModel):
    id: str = Field(min_length=1, max_length=180)
    order: int = Field(ge=1, le=80)
    title: str = Field(min_length=1, max_length=180)
    short_description: str = Field(min_length=1, max_length=400)
    ncert_sections: List[PlanningNcertSection] = Field(min_length=1, max_length=40)
    ncert_subtopics: List[PlanningNcertSubtopic] = Field(min_length=1, max_length=80)
    concepts: List[PlanningConcept] = Field(min_length=1, max_length=80)
    skills: List[str] = Field(default_factory=list, max_length=40)
    practice: List[str] = Field(default_factory=list, max_length=40)
    importance: Literal["very_high", "high", "moderate", "low"]
    difficulty: Literal["foundation", "steady", "challenging"]
    estimated_minutes: PlanningEstimatedMinutes
    prerequisite_unit_ids: List[str] = Field(default_factory=list, max_length=40)
    dependent_unit_ids: List[str] = Field(default_factory=list, max_length=40)
    learning_types: List[str] = Field(default_factory=list, max_length=20)
    depth: Literal["overview", "working", "mastery"]
    exam_relevance: Literal["very_high", "high", "moderate", "low"]
    conceptual_importance: Literal["very_high", "high", "moderate", "low"]
    why_it_matters: str = Field(min_length=1, max_length=500)
    learning_route: List[str] = Field(min_length=2, max_length=4)
    mastery_criteria: List[str] = Field(min_length=1, max_length=20)
    status: Literal[
        "not_started",
        "recommended",
        "learning",
        "practising",
        "needs_review",
        "mastered",
    ]
    primary_topic_id: str = Field(min_length=1, max_length=180)


class PlanningNextStep(BaseModel):
    unit_id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=180)
    reason: str = Field(min_length=1, max_length=500)
    estimated_minutes: PlanningEstimatedMinutes
    importance: Literal["very_high", "high", "moderate", "low"]
    learning_types: List[str] = Field(min_length=1, max_length=20)
    approach: List[str] = Field(min_length=2, max_length=4)
    outcome: str = Field(min_length=1, max_length=500)


class PlanningDailyRouteItem(BaseModel):
    unit_id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=180)
    activity: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=400)
    role: Literal["main_focus", "quick_check"]
    minutes: int = Field(ge=5, le=180)
    scope: Literal["partial", "full_unit"]


class PlanningDailyRoute(BaseModel):
    source: Literal["default_focus", "student_choice", "session_state"]
    budget_minutes: Optional[int] = Field(default=None, ge=15, le=120)
    estimated_minutes: PlanningEstimatedMinutes
    total_minutes: int = Field(ge=5, le=180)
    items: List[PlanningDailyRouteItem] = Field(min_length=1, max_length=20)


class PlanningProgress(BaseModel):
    mastered_units: int = Field(ge=0, le=80)
    learning_units: int = Field(ge=0, le=80)
    practising_units: int = Field(ge=0, le=80)
    needs_review_units: int = Field(ge=0, le=80)
    recommended_units: int = Field(ge=0, le=1)
    total_units: int = Field(ge=1, le=80)
    percentage: int = Field(ge=0, le=100)


class PlanningPortfolioChapterSelection(BaseModel):
    chapter_ref: str = Field(min_length=1, max_length=240)
    chapter_proficiency: Literal[
        "new_to_it",
        "know_a_little",
        "know_the_basics",
        "mostly_confident",
    ] = "know_a_little"

    model_config = {"extra": "forbid"}

    @field_validator("chapter_ref")
    @classmethod
    def normalize_chapter_ref(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Select a chapter before building a portfolio")
        return normalized

    @field_validator("chapter_proficiency", mode="before")
    @classmethod
    def normalize_proficiency(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        return {
            "new": "new_to_it",
            "some_idea": "know_a_little",
            "know_basics": "know_the_basics",
            "fast_track": "mostly_confident",
            "quick_revision": "mostly_confident",
        }.get(normalized, normalized)


class PlanningPortfolioRequest(BaseModel):
    class_level: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=120)
    chapters: List[PlanningPortfolioChapterSelection] = Field(
        min_length=1,
        max_length=6,
    )
    study_time_today: Optional[
        Literal["15", "30", "60", "120_plus", "no_limit"]
    ] = None
    session_duration_minutes: Optional[int] = Field(default=None, ge=15, le=120)

    model_config = {"extra": "forbid"}

    @field_validator("class_level", "subject")
    @classmethod
    def normalize_scope_label(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Select a class and subject before building a portfolio")
        return normalized

    @field_validator("study_time_today", mode="before")
    @classmethod
    def normalize_study_time_today(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and float(value).is_integer():
            numeric = int(value)
            return "120_plus" if numeric == 120 else str(numeric)
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            return {
                "15_min": "15",
                "15_mins": "15",
                "15_minute": "15",
                "15_minutes": "15",
                "30_min": "30",
                "30_mins": "30",
                "30_minute": "30",
                "30_minutes": "30",
                "1_hour": "60",
                "1_hr": "60",
                "60_min": "60",
                "60_mins": "60",
                "60_minutes": "60",
                "120": "120_plus",
                "120+": "120_plus",
                "2_hours": "120_plus",
                "2+_hours": "120_plus",
                "2_plus_hours": "120_plus",
                "2_hours+": "120_plus",
                "unlimited": "no_limit",
            }.get(normalized, normalized)
        return value


class PlanningPortfolioSelectionFactor(BaseModel):
    id: Literal[
        "status_urgency",
        "chapter_continuity",
        "importance",
        "exam_relevance",
        "effort_sizing",
        "selection_order",
    ]
    label: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=120)
    score: int = Field(ge=0, le=1000)
    explanation: str = Field(min_length=1, max_length=320)


class PlanningPortfolioChapterRoadmap(BaseModel):
    curriculum_key: str = Field(min_length=1, max_length=180)
    chapter_slug: str = Field(min_length=1, max_length=180)
    chapter: str = Field(min_length=1, max_length=240)
    chapter_proficiency: Literal[
        "new_to_it",
        "know_a_little",
        "know_the_basics",
        "mostly_confident",
    ]
    roadmap_version: Literal["planning_roadmap_v2"]
    curriculum: PlanningCurriculumMetadata
    learning_units: List[PlanningLearningUnit] = Field(min_length=1, max_length=80)
    next_step: PlanningNextStep
    progress: PlanningProgress
    coverage: PlanningCoverage
    completion_criteria: List[str] = Field(min_length=1, max_length=20)
    candidate_score: int = Field(ge=0, le=5000)
    selection_factors: List[PlanningPortfolioSelectionFactor] = Field(
        min_length=1,
        max_length=8,
    )
    selected_for_today: bool


class PlanningPortfolioGlobalNextStep(PlanningNextStep):
    curriculum_key: str = Field(min_length=1, max_length=180)
    chapter_slug: str = Field(min_length=1, max_length=180)
    chapter: str = Field(min_length=1, max_length=240)
    chapter_proficiency: Literal[
        "new_to_it",
        "know_a_little",
        "know_the_basics",
        "mostly_confident",
    ]
    selection_reason: str = Field(min_length=1, max_length=500)
    candidate_score: int = Field(ge=0, le=5000)


class PlanningPortfolioRouteItem(PlanningDailyRouteItem):
    curriculum_key: str = Field(min_length=1, max_length=180)
    chapter_slug: str = Field(min_length=1, max_length=180)
    chapter: str = Field(min_length=1, max_length=240)


class PlanningPortfolioTodayRoute(BaseModel):
    source: Literal["default_focus", "student_choice", "session_state"]
    budget_minutes: Optional[int] = Field(default=None, ge=15, le=120)
    estimated_minutes: PlanningEstimatedMinutes
    total_minutes: int = Field(ge=5, le=180)
    items: List[PlanningPortfolioRouteItem] = Field(min_length=1, max_length=20)


class PlanningPortfolioAggregateProgress(BaseModel):
    mastered_units: int = Field(ge=0, le=480)
    active_units: int = Field(ge=0, le=480)
    needs_review_units: int = Field(ge=0, le=480)
    total_units: int = Field(ge=1, le=480)
    percentage: int = Field(ge=0, le=100)


class PlanningPortfolioResponse(BaseModel):
    portfolio_version: Literal["planning_portfolio_v1"]
    user_id: str = Field(min_length=1, max_length=240)
    class_level: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=120)
    requested_chapter_count: int = Field(ge=1, le=6)
    chapter_count: int = Field(ge=1, le=6)
    deduplicated_chapter_count: int = Field(ge=0, le=5)
    study_time_today: Optional[
        Literal["15", "30", "60", "120_plus", "no_limit"]
    ] = None
    session_duration_minutes: Optional[int] = Field(default=None, ge=15, le=120)
    chapters: List[PlanningPortfolioChapterRoadmap] = Field(min_length=1, max_length=6)
    global_next_step: PlanningPortfolioGlobalNextStep
    today_route: PlanningPortfolioTodayRoute
    selection_factors: List[PlanningPortfolioSelectionFactor] = Field(
        min_length=1,
        max_length=8,
    )
    aggregate_progress: PlanningPortfolioAggregateProgress


class AutonomousStudyResponse(BaseModel):
    mission_id: str
    status: str
    subject: str
    chapter: str = ""
    plan_scope: str = "chapter"
    brief_version: Literal["chapter_focus_v1"] = "chapter_focus_v1"
    chapter_summary: str = ""
    focus_areas: List[PlanningFocusArea] = Field(default_factory=list)
    guidance_steps: List[PlanningGuidanceStep] = Field(default_factory=list)
    completion_signal: str = ""
    coverage: Optional[PlanningCoverage] = None
    roadmap_version: Optional[Literal["planning_roadmap_v2"]] = None
    chapter_proficiency: Literal[
        "new_to_it",
        "know_a_little",
        "know_the_basics",
        "mostly_confident",
    ] = "know_a_little"
    study_time_today: Optional[
        Literal["15", "30", "60", "120_plus", "no_limit"]
    ] = None
    session_duration_minutes: Optional[int] = Field(default=None, ge=15, le=120)
    class_level: str = ""
    chapter_slug: str = ""
    curriculum: Optional[PlanningCurriculumMetadata] = None
    learning_units: List[PlanningLearningUnit] = Field(default_factory=list)
    next_step: Optional[PlanningNextStep] = None
    daily_route: Optional[PlanningDailyRoute] = None
    progress: Optional[PlanningProgress] = None
    completion_criteria: List[str] = Field(default_factory=list)
    learning_unit_count: int = 0
    # Retained as a response alias for older saved clients. Chapter Planning
    # now places the chapter label here; requests no longer accept a topic.
    target_topic: str
    target_source: str
    mission_type: str = "study"
    priority: str = "medium"
    mastery_band: str = "unknown"
    estimated_minutes: int = 0
    mission_goal: str = ""
    prerequisite_check: Dict[str, Any] = Field(default_factory=dict)
    high_priority_concepts: List[str] = Field(default_factory=list)
    fast_revision_strategy: List[str] = Field(default_factory=list)
    weakness_detection_points: List[str] = Field(default_factory=list)
    final_confidence_check: List[str] = Field(default_factory=list)
    primary_agent: str
    mode: str
    difficulty: str
    objective: str
    why: str
    steps: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    study_plan: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostic_question: Dict[str, Any] = Field(default_factory=dict)
    adaptive_roadmap: List[Dict[str, Any]] = Field(default_factory=list)
    agent_sequence: List[Dict[str, Any]] = Field(default_factory=list)
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list)
    student_state: Dict[str, Any] = Field(default_factory=dict)
    completion_report: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    analytics_summary: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0


# =========================================================
# LEADERBOARD SCHEMAS
# =========================================================
class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    xp: int
    streak: int
    total_tests: int

    model_config = {"from_attributes": True}


# =========================================================
# SYSTEM HEALTH SCHEMA
# =========================================================
class HealthResponse(BaseModel):
    status: str
    database: bool
    version: str = "2.0.0-bloomberg"
