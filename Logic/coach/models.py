"""Typed internal models for the unified Study Lab coach."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class QueryUnderstanding:
    intent: str
    answer_format: str
    is_conversational: bool = False
    is_follow_up: bool = False
    needs_retrieval: bool = False
    retrieval_policy: str = "none"
    requires_grounding: bool = False
    reasoning_mode: str = "general_reasoning"
    needs_memory: bool = True
    needs_quality_review: bool = True
    requested_tools: List[str] = field(default_factory=list)
    anchor_terms: List[str] = field(default_factory=list)
    topic_shift: bool = False
    confidence: float = 0.72
    teaching_strategy: str = ""
    scenario_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    context: str = ""
    section_id: str = ""
    source: str = ""
    paragraphs_found: int = 0
    keywords_used: List[str] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)
    supported: bool = False
    error: str = ""
    retrieval_mode: str = ""
    matched_sections: List[str] = field(default_factory=list)
    semantic_matches: int = 0
    best_semantic_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoachPlan:
    route: str
    intent: str
    answer_format: str
    tools: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    score: float
    passed: bool
    relevance: float
    grounding: float
    completeness: float
    clarity: float
    student_friendliness: float
    formatting: float
    intent_satisfaction: float
    readability: float
    hallucination_risk: float
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
