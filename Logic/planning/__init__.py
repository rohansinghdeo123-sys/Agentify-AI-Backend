"""Curriculum-grounded planning primitives.

Planning deliberately keeps curriculum data, recommendation logic, and HTTP
presentation separate.  The public autonomous-study route remains the rollout
entry point while these modules provide the versioned roadmap contract.
"""

from .curriculum_registry import (
    PlanningCurriculumError,
    load_planning_curriculum,
    resolve_planning_curriculum,
)
from .recommendation_engine import build_planning_roadmap

__all__ = [
    "PlanningCurriculumError",
    "build_planning_roadmap",
    "load_planning_curriculum",
    "resolve_planning_curriculum",
]
