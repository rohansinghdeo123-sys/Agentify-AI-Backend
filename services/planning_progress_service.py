"""Durable, server-owned Study evidence for deterministic Planning roadmaps."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from Logic.planning.curriculum_registry import (
    load_planning_curricula,
    resolve_planning_curriculum,
)
from models import PlanningLearningEvent
from services.catalog_service import (
    find_published_planning_curriculum_by_key,
    resolve_published_planning_curriculum,
)


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def resolve_learning_event_scope(
    scope: Mapping[str, Any],
    db: Optional[Session] = None,
) -> Optional[Dict[str, str]]:
    """Resolve an explicit manifest or published Planning handoff to one unit."""
    catalog_source = _normalized(scope.get("catalog_source"))
    if catalog_source not in {"planning_manifest", "published"}:
        return None
    chapter_ref = str(
        scope.get("chapter_slug")
        or scope.get("selected_chapter_id")
        or scope.get("chapter")
        or ""
    ).strip()
    subject = str(scope.get("subject") or scope.get("selected_subject") or "") or None
    class_level = str(scope.get("class_level") or "") or None
    if catalog_source == "planning_manifest":
        curriculum = resolve_planning_curriculum(
            chapter_ref=chapter_ref,
            subject=subject,
            class_level=class_level,
        )
    else:
        if db is None:
            return None
        curriculum = resolve_published_planning_curriculum(
            db,
            chapter_ref=chapter_ref,
            subject=subject,
            class_level=class_level,
        )
    if curriculum is None:
        return None

    requested = {
        _normalized(value)
        for value in (
            scope.get("planning_unit_id"),
            scope.get("selected_topic_id"),
            scope.get("section_id"),
            scope.get("primary_topic_id"),
        )
        if _normalized(value)
    }
    matches = [
        unit
        for unit in curriculum["units"]
        if requested.intersection(
            {
                _normalized(unit["id"]),
                _normalized(unit["title"]),
                _normalized(unit["primary_topic_id"]),
                *(_normalized(value) for value in unit.get("legacy_topic_ids") or []),
                *(
                    _normalized(value)
                    for concept in unit.get("concepts") or []
                    for value in (concept.get("id"), concept.get("title"))
                ),
            }
        )
    ]
    if len(matches) != 1:
        return None
    unit = matches[0]
    return {
        "curriculum_key": str(curriculum["curriculum_key"]),
        "unit_id": str(unit["id"]),
        "chapter_slug": str(curriculum["chapter_slug"]),
        "primary_topic_id": str(unit["primary_topic_id"]),
    }


def _event_count(
    db: Session,
    *,
    user_id: str,
    curriculum_key: str,
    unit_id: str,
) -> int:
    return int(
        db.query(func.count(PlanningLearningEvent.id))
        .filter(
            PlanningLearningEvent.user_id == user_id,
            PlanningLearningEvent.curriculum_key == curriculum_key,
            PlanningLearningEvent.unit_id == unit_id,
            PlanningLearningEvent.event_type == "study_answer",
        )
        .scalar()
        or 0
    )


def _event_response(
    db: Session,
    event: PlanningLearningEvent,
    *,
    recorded: bool,
    idempotent: bool,
) -> Dict[str, Any]:
    return {
        "recorded": recorded,
        "idempotent": idempotent,
        "interaction_id": event.interaction_id,
        "event_count": _event_count(
            db,
            user_id=event.user_id,
            curriculum_key=event.curriculum_key,
            unit_id=event.unit_id,
        ),
        "status": "learning",
        "curriculum_key": event.curriculum_key,
        "unit_id": event.unit_id,
    }


def record_study_answer_event(
    db: Session,
    *,
    user_id: str,
    interaction_id: str,
    scope: Mapping[str, Any],
    source_session_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Idempotently record a server-completed, curriculum-grounded answer."""
    resolved = resolve_learning_event_scope(scope, db=db)
    if resolved is None or not user_id.strip() or not interaction_id.strip():
        return None
    identity = {
        "user_id": user_id.strip(),
        "curriculum_key": resolved["curriculum_key"],
        "unit_id": resolved["unit_id"],
        "interaction_id": interaction_id.strip(),
    }
    existing = db.query(PlanningLearningEvent).filter_by(**identity).first()
    if existing is not None:
        return _event_response(db, existing, recorded=False, idempotent=True)

    event = PlanningLearningEvent(
        **identity,
        event_type="study_answer",
        source_session_id=source_session_id.strip(),
    )
    db.add(event)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.query(PlanningLearningEvent).filter_by(**identity).first()
        if existing is None:
            raise
        return _event_response(db, existing, recorded=False, idempotent=True)
    db.refresh(event)
    return _event_response(db, event, recorded=True, idempotent=False)


def confirm_study_answer_event(
    db: Session,
    *,
    user_id: str,
    interaction_id: str,
) -> Optional[Dict[str, Any]]:
    """Confirm an event that the backend already created at answer completion."""
    event = (
        db.query(PlanningLearningEvent)
        .filter(
            PlanningLearningEvent.user_id == user_id,
            PlanningLearningEvent.interaction_id == interaction_id,
            PlanningLearningEvent.event_type == "study_answer",
        )
        .first()
    )
    if event is None:
        return None
    curricula = {
        str(curriculum["curriculum_key"]): curriculum
        for curriculum in load_planning_curricula()
    }
    curriculum = curricula.get(event.curriculum_key)
    if curriculum is None:
        curriculum = find_published_planning_curriculum_by_key(
            db,
            event.curriculum_key,
        )
    if curriculum is None or event.unit_id not in {
        str(unit["id"]) for unit in curriculum["units"]
    }:
        return None
    return _event_response(db, event, recorded=False, idempotent=True)


def planning_learning_states(
    db: Session,
    *,
    user_id: str,
    curriculum_key: str,
    valid_unit_ids: Sequence[str],
    unit_aliases: Optional[Mapping[str, str]] = None,
) -> list[SimpleNamespace]:
    """Project durable exposure into the engine's canonical persisted-state shape."""
    aliases = {
        _normalized(alias): canonical
        for alias, canonical in (unit_aliases or {}).items()
        if _normalized(alias) and canonical in set(valid_unit_ids)
    }
    canonical_by_identity = {
        **{_normalized(unit_id): unit_id for unit_id in valid_unit_ids},
        **aliases,
    }
    rows = (
        db.query(
            PlanningLearningEvent.unit_id,
            func.count(PlanningLearningEvent.id),
        )
        .filter(
            PlanningLearningEvent.user_id == user_id,
            PlanningLearningEvent.curriculum_key == curriculum_key,
            PlanningLearningEvent.event_type == "study_answer",
        )
        .group_by(PlanningLearningEvent.unit_id)
        .all()
    )
    counts: Dict[str, int] = {}
    for unit_id, event_count in rows:
        canonical = canonical_by_identity.get(_normalized(unit_id))
        if canonical and int(event_count or 0) > 0:
            counts[canonical] = counts.get(canonical, 0) + int(event_count or 0)
    return [
        SimpleNamespace(
            unit_id=unit_id,
            status="learning",
            evidence_count=event_count,
            mastery_score=None,
        )
        for unit_id, event_count in counts.items()
    ]
