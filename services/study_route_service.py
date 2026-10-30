"""One persistent route, backed by the existing curriculum and mastery engine.

The itinerary is captured at start so refresh cannot move a learner mid-unit.
Mastery remains derived from TestHistory, TopicPerformance and study exposure.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from sqlalchemy.orm import Session

from Logic.analytics_engine import update_topic_performance
from Logic.planning.portfolio_engine import PlanningPortfolioError, build_planning_portfolio
from Logic.planning.curriculum_registry import resolve_planning_curriculum
from models import AICoachProfile, ContentChapter, ContentConcept, StudyRouteState, TestHistory
from services.catalog_service import resolve_published_planning_curriculum, resolve_catalog_topic
from services.planning_progress_service import record_study_answer_event


class StudyRouteError(ValueError):
    pass


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _get_state(db, user_id, *, lock=False):
    query = db.query(StudyRouteState).filter(StudyRouteState.user_id == user_id)
    if lock:
        query = query.with_for_update()
    return query.order_by(StudyRouteState.route_date.desc(), StudyRouteState.id.desc()).first()


def _build_portfolio_from_state(db, state):
    try:
        return build_planning_portfolio(
            db, user_id=state.user_id, class_level=state.class_level,
            subject=state.subject, selections=state.selections_json or [],
            study_time_today=state.study_time_today,
            session_duration_minutes=state.session_duration_minutes,
        )
    except PlanningPortfolioError as exc:
        raise StudyRouteError(str(exc)) from exc


def _chapter(portfolio, state):
    return next((c for c in portfolio['chapters']
                 if c['curriculum_key'] == state.active_curriculum_key), portfolio['chapters'][0])


def _unit(chapter, unit_id):
    found = next((u for u in chapter['learning_units'] if u['id'] == unit_id), None)
    if found is None:
        raise StudyRouteError('The requested unit is not in the active chapter')
    return found


def _curriculum(db, state):
    args = dict(chapter_ref=state.active_chapter_slug, subject=state.subject, class_level=state.class_level)
    return (resolve_planning_curriculum(**args)
            or resolve_published_planning_curriculum(db, **args))


def learning_notes(db, state, unit):
    """Replaceable notes adapter: approved concept key points, then unit skills.

    This intentionally returns plain bullets until Amit supplies the final format.
    """
    ids = [c['id'] for c in unit.get('concepts', [])]
    from Logic.content_pipeline import APPROVED_STATUSES
    chapter = db.query(ContentChapter).filter(
        ContentChapter.slug == state.active_chapter_slug,
        ContentChapter.status.in_(APPROVED_STATUSES),
    ).first()
    points = []
    if chapter and ids:
        concepts = db.query(ContentConcept).filter(
            ContentConcept.chapter_id == chapter.id, ContentConcept.concept_id.in_(ids),
        ).order_by(ContentConcept.id).all()
        points = [str(p).strip() for c in concepts for p in (c.key_points or []) if isinstance(p, str) and p.strip()]
    return list(dict.fromkeys(points or unit.get('skills') or [unit['short_description']]))[:6]


def _itinerary(portfolio):
    return list(dict.fromkeys(i['unit_id'] for i in portfolio['today_route']['items']))


def start_route(db: Session, *, user_id: str, class_level: str, subject: str,
                selections: Sequence[Mapping[str, Any]], study_time_today=None,
                session_duration_minutes=None):
    if not user_id.strip() or not selections:
        raise StudyRouteError('A student and chapter selections are required')
    # Validate before writing; use the existing request/engine vocabulary.
    from schemas import StudyRouteStartRequest
    from pydantic import ValidationError
    try:
        payload = StudyRouteStartRequest(class_level=class_level, subject=subject,
            chapters=list(selections), study_time_today=study_time_today,
            session_duration_minutes=session_duration_minutes)
    except ValidationError as exc:
        raise StudyRouteError(str(exc)) from exc
    selections = [c.model_dump() for c in payload.chapters]
    state = _get_state(db, user_id, lock=True)
    same = state is not None and (
        state.class_level == class_level and state.subject == subject
        and state.selections_json == selections
        and state.study_time_today == payload.study_time_today
        and state.session_duration_minutes == session_duration_minutes)
    if same and state.status != 'completed':
        return read_current_route(db, user_id=user_id)
    previous_effective = dict(state.effective_proficiency_json or {}) if state else {}
    previous_review = list(state.needs_review_concept_ids_json or []) if state else []
    if state is None or state.route_date != _today():
        state = StudyRouteState(user_id=user_id, route_date=_today())
        db.add(state)
    state.class_level, state.subject = class_level, subject
    state.selections_json = selections
    state.study_time_today = payload.study_time_today
    state.session_duration_minutes = session_duration_minutes
    portfolio = _build_portfolio_from_state(db, state)
    step = portfolio['global_next_step']
    state.active_curriculum_key = step['curriculum_key']
    state.active_chapter_slug = step['chapter_slug']
    state.current_unit_id = step['unit_id']
    state.current_step_index = 0
    state.current_step_role = 'main_focus'
    state.runtime_json = {'today_route': portfolio['today_route'], 'unit_ids': _itinerary(portfolio),
                          'completed_unit_ids': [], 'receipts': {}}
    state.last_position_json = {}
    state.override_reason = state.override_until_unit_id = ''
    state.effective_proficiency_json = previous_effective
    state.needs_review_concept_ids_json = previous_review
    coach = db.query(AICoachProfile).filter(AICoachProfile.user_id == user_id).first()
    state.goal_json = ({'active_goals': coach.active_goals or [], 'target_exam': coach.target_exam,
                        'target_exam_date': coach.target_exam_date.isoformat() if coach.target_exam_date else None}
                       if coach else {})
    state.status = 'not_started'
    state.started_at = None
    state.last_active_at = _now()
    state.completed_at = None
    db.commit()
    return read_current_route(db, user_id=user_id, portfolio=portfolio)


def read_current_route(db, *, user_id, portfolio=None):
    state = _get_state(db, user_id)
    if state is None:
        raise StudyRouteError('No saved learning route')
    portfolio = portfolio or _build_portfolio_from_state(db, state)
    chapter = _chapter(portfolio, state)
    unit = _unit(chapter, state.current_unit_id)
    runtime = dict(state.runtime_json or {})
    if not runtime.get('today_route'):
        runtime.update(today_route=portfolio['today_route'], unit_ids=_itinerary(portfolio), completed_unit_ids=[], receipts={})
        state.runtime_json = runtime
        db.commit()
    route = runtime['today_route']
    ids = runtime.get('unit_ids') or _itinerary(portfolio)
    total = max(1, len(ids))
    index = min(int(state.current_step_index or 0), total - 1)
    item = next((i for i in route['items'] if i['unit_id'] == unit['id'] and i['role'] == 'main_focus'), {})
    position = dict(state.last_position_json or {})
    check = runtime.get('check') or {}
    answered = len(position.get('check_answers', {})) if check.get('unit_id') == unit['id'] else 0
    progress = 100 if unit['status'] == 'mastered' else min(90, int(answered / max(1, len(check.get('questions', []))) * 90))
    if position.get('conversation_id') and not progress:
        progress = 10
    override = {'active': bool(state.override_reason), 'reason': state.override_reason or '',
        'reason_label': 'Temporary prerequisite refresh' if state.override_reason == 'prerequisite_gap' else 'Targeted revision',
        'until_unit_id': state.override_until_unit_id or '',
        'return_unit_id': (runtime.get('branch') or {}).get('unit_id', ''),
        'explanation': (runtime.get('branch') or {}).get('reason', '')}
    next_step = chapter['next_step']
    why = override['explanation'] if override['active'] else (
        item.get('reason') or (next_step['reason'] if next_step['unit_id'] == unit['id'] else unit['why_it_matters']))
    return {'user_id': user_id, 'route_date': state.route_date.isoformat(),
        'class_level': state.class_level, 'subject': state.subject, 'status': state.status,
        'study_time_today': state.study_time_today, 'session_duration_minutes': state.session_duration_minutes,
        'goal': state.goal_json or {}, 'portfolio': portfolio,
        'completed_unit_ids': list(dict.fromkeys([*runtime.get('completed_unit_ids', []),
            *(u['id'] for c in portfolio['chapters'] for u in c['learning_units'] if u['status'] == 'mastered')])),
        'portfolio_version': portfolio['portfolio_version'], 'chapters': portfolio['chapters'],
        'global_next_step': portfolio['global_next_step'], 'today_route': route,
        'selection_factors': portfolio['selection_factors'], 'aggregate_progress': portfolio['aggregate_progress'],
        'current_step': {'step_index': index, 'step_total': total, 'step_label': f'Step {index + 1} of {total}',
            'current_unit_id': unit['id'], 'current_unit_title': unit['title'],
            'current_unit_role': 'main_focus', 'current_unit_activity': item.get('activity', ''),
            'current_unit_minutes': item.get('minutes', unit['estimated_minutes']['min']),
            'current_unit_scope': item.get('scope', 'partial'),
            'recommended_approach': unit['learning_route'], 'why_now': why,
            'outcome': unit['mastery_criteria'][0] if unit['mastery_criteria'] else '',
            'learning_types': unit['learning_types'], 'unit_depth': unit['depth'],
            'unit_difficulty': unit['difficulty'], 'unit_importance': unit['importance'],
            'prerequisites': unit['prerequisite_unit_ids'], 'unit_status': unit['status'],
            'last_position': position, 'key_notes': learning_notes(db, state, unit),
            'progress_percentage': progress},
        'override': override, 'effective_proficiency': {c['curriculum_key']: c['effective_proficiency'] for c in portfolio['chapters']},
        'needs_review_concept_ids': [u['primary_topic_id'] for c in portfolio['chapters'] for u in c['learning_units'] if u['status'] == 'needs_review'],
        'started_at': state.started_at, 'last_active_at': state.last_active_at, 'completed_at': state.completed_at}


def build_study_lab_context(db, *, user_id):
    r = read_current_route(db, user_id=user_id)
    s = r['current_step']
    return {'user_id': user_id, 'route_date': r['route_date'], 'status': r['status'],
        'class_level': r['class_level'], 'subject': r['subject'],
        'study_time_today': r['study_time_today'], 'session_duration_minutes': r['session_duration_minutes'],
        'current_topic': s['current_unit_title'], 'step_label': s['step_label'],
        'step_index': s['step_index'], 'step_total': s['step_total'], 'why_now': s['why_now'],
        'recommended_approach': s['recommended_approach'], 'recommended_depth': s['unit_depth'],
        'recommended_minutes': s['current_unit_minutes'], 'key_notes': s['key_notes'],
        'progress_percentage': s['progress_percentage'], 'learning_types': s['learning_types'],
        'prerequisites': s['prerequisites'], 'current_unit_id': s['current_unit_id'],
        'current_unit_status': s['unit_status'], 'progress': r['aggregate_progress'],
        'goal': r['goal'], 'override': r['override'], 'effective_proficiency': r['effective_proficiency'],
        'needs_review_concept_ids': r['needs_review_concept_ids'], 'last_position': s['last_position'],
        'next_step': r['global_next_step'], 'today_route': r['today_route'], 'chapters': r['chapters']}


def prepare_check(db, *, user_id):
    """Generate once and keep the answer key private, including across refresh."""
    from Logic.section_doubt import generate_structured_mcqs
    from app.security import enforce_user_quota
    state = _get_state(db, user_id, lock=True)
    if not state:
        raise StudyRouteError('No saved learning route')
    runtime = dict(state.runtime_json or {})
    check = runtime.get('check') or {}
    if check.get('unit_id') != state.current_unit_id:
        enforce_user_quota(user_id, 'exam')
        curriculum = _curriculum(db, state)
        unit = next(u for u in curriculum['units'] if u['id'] == state.current_unit_id)
        count = min(10, max(3, int(unit.get('practice', {}).get('minimum_items') or 3)))
        source = 'published' if curriculum.get('source', {}).get('authority') == 'approved_content_pipeline' else 'planning_manifest'
        scope = dict(catalog_source=source, chapter_slug=state.active_chapter_slug,
            chapter=state.active_chapter_slug, subject=state.subject, class_level=state.class_level,
            planning_unit_id=unit['id'], selected_topic_id=unit['primary_topic_id'])
        resolved = resolve_catalog_topic(db, unit['id'], subject=state.subject,
            chapter=state.active_chapter_slug, topic=unit['primary_topic_id'],
            class_level=state.class_level, catalog_source=source)
        if resolved:
            scope.update(resolved)
        pack = generate_structured_mcqs(topic=unit['title'], section_id=unit['primary_topic_id'],
            session_id=f'{user_id}_study_check', count=count, strict_grounding=True,
            class_level=state.class_level, content_scope=scope)
        questions = pack.get('questions') or []
        if not questions:
            raise StudyRouteError(pack.get('error') or 'No approved questions available for this unit')
        check = {'id': str(uuid4()), 'unit_id': unit['id'], 'questions': questions, 'started_at': _now().isoformat()}
        runtime['check'] = check
        state.runtime_json = runtime
        db.commit()
    return {'id': check['id'], 'unit_id': check['unit_id'], 'questions': [
        {k: q[k] for k in ('id', 'question', 'options')} for q in check['questions']]}


def _response(db, state, completion):
    route = read_current_route(db, user_id=state.user_id)
    chapter = _chapter(route['portfolio'], state)
    unit = _unit(chapter, state.current_unit_id)
    return {'completion': completion, 'route': route, 'next_action': {
        'title': unit['title'], 'reason': route['current_step']['why_now'],
        'estimated_minutes': unit['estimated_minutes'], 'approach': unit['learning_route'],
        'outcome': unit['mastery_criteria'][0] if unit['mastery_criteria'] else '',
        'done_for_today': state.status == 'completed'}}


def advance_route(db, *, user_id, completed_unit_id, interaction_id, answers=None,
                  correct_answers=0, total_questions=1, time_spent_seconds=0,
                  confidence_after=None, xp_earned=None, last_position=None):
    state = _get_state(db, user_id, lock=True)
    if not state:
        raise StudyRouteError('No saved learning route')
    runtime = dict(state.runtime_json or {})
    receipt = (runtime.get('receipts') or {}).get(interaction_id)
    if receipt:
        if receipt['unit_id'] != completed_unit_id:
            raise StudyRouteError('This assessment receipt belongs to a different unit')
        return _response(db, state, receipt)
    if completed_unit_id != state.current_unit_id:
        raise StudyRouteError('Only the current unit can be assessed; refresh your route')
    check = runtime.get('check') or {}
    if check.get('id') != interaction_id or check.get('unit_id') != completed_unit_id:
        raise StudyRouteError('Complete the current quick check before saving progress')
    questions = check['questions']
    answers = answers or {}
    if set(answers) != {q['id'] for q in questions}:
        raise StudyRouteError('Answer every question before submitting the check')
    for q in questions:
        valid = {o[:1] for o in q['options']}
        if answers[q['id']] not in valid:
            raise StudyRouteError('Invalid answer option')
    correct = sum(answers[q['id']] == q['correct'] for q in questions)
    total = len(questions)
    accuracy = correct / total * 100
    portfolio = _build_portfolio_from_state(db, state)
    chapter = _chapter(portfolio, state)
    unit = _unit(chapter, completed_unit_id)
    now = _now()
    elapsed = max(0, int((now - datetime.fromisoformat(check['started_at'])).total_seconds()))
    history = TestHistory(user_id=user_id, date=now.date(), topic=unit['primary_topic_id'],
        score=correct, total_questions=total, accuracy_rate=accuracy, xp_earned=correct,
        time_spent_seconds=min(elapsed, 86400), session_type='practice', completed_at=now,
        confidence_after=confidence_after)
    db.add(history)
    update_topic_performance(db, user_id, unit['primary_topic_id'], correct, total, elapsed, commit=False)
    scope = {'catalog_source': 'planning_manifest', 'chapter_slug': state.active_chapter_slug,
             'class_level': state.class_level, 'subject': state.subject, 'planning_unit_id': unit['id']}
    event = record_study_answer_event(db, user_id=user_id, interaction_id=interaction_id,
        scope=scope, source_session_id=str((state.last_position_json or {}).get('conversation_id', '')), commit=False)
    if event is None:
        scope['catalog_source'] = 'published'
        event = record_study_answer_event(db, user_id=user_id, interaction_id=interaction_id, scope=scope, commit=False)
    if event is None:
        db.rollback()
        raise StudyRouteError('This assessment could not be mapped to the curriculum')
    db.flush()
    fresh = _build_portfolio_from_state(db, state)
    refreshed_unit = _unit(_chapter(fresh, state), completed_unit_id)
    mastered = refreshed_unit['status'] == 'mastered'
    completion = {'unit_id': unit['id'], 'unit_title': unit['title'], 'accuracy_rate': round(accuracy, 2),
        'correct_answers': correct, 'total_questions': total, 'mastered': mastered,
        'feedback': [{'id': q['id'], 'correct': q['correct'], 'explanation': q['explanation'],
                      'is_correct': answers[q['id']] == q['correct']} for q in questions]}
    runtime['receipts'] = {**runtime.get('receipts', {}), interaction_id: completion}
    runtime.pop('check', None)
    branch = runtime.get('branch')
    if mastered:
        completed = list(runtime.get('completed_unit_ids', []))
        if unit['id'] not in completed:
            completed.append(unit['id'])
        runtime['completed_unit_ids'] = completed
        if branch:
            state.current_unit_id = branch['unit_id']
            state.current_step_index = branch['step_index']
            state.last_position_json = branch['position']
            runtime.pop('branch', None)
            state.override_reason = state.override_until_unit_id = ''
        else:
            step = fresh['global_next_step']
            state.active_curriculum_key, state.active_chapter_slug = step['curriculum_key'], step['chapter_slug']
            state.current_unit_id = step['unit_id']
            state.current_step_index = int(state.current_step_index or 0) + 1
            state.last_position_json = {}
    else:
        state.last_position_json = {k: v for k, v in (state.last_position_json or {}).items() if k != 'check_answers'}
    state.status = 'in_progress'
    state.completed_at = None
    state.last_active_at = now
    state.runtime_json = runtime
    # Weak prerequisite evidence prompts a temporary, explained branch.
    if accuracy < 60 and not branch and unit['prerequisite_unit_ids']:
        _set_branch(state, chapter, 'prerequisite_gap', unit['prerequisite_unit_ids'][0],
                    'Your last check showed a gap. Review this prerequisite, then return to ' + unit['title'] + '.')
    if mastered and not branch:
        remaining = set(runtime.get('unit_ids', [])) - set(runtime.get('completed_unit_ids', []))
        if not remaining:
            state.status = 'completed'
            state.completed_at = now
    db.commit()
    return _response(db, state, completion)


def _set_branch(state, chapter, reason, target, explanation):
    unit = _unit(chapter, state.current_unit_id)
    target_unit = _unit(chapter, target)
    by_id = {u['id']: u for u in chapter['learning_units']}
    prerequisites = set()
    def visit(u):
        for dep in u['prerequisite_unit_ids']:
            if dep not in prerequisites:
                prerequisites.add(dep)
                visit(by_id[dep])
    visit(unit)
    if reason == 'prerequisite_gap' and target not in prerequisites:
        raise StudyRouteError('Branch target must be a prerequisite of the current unit')
    if reason == 'revision_required' and (target != unit['id'] or unit['status'] != 'needs_review'):
        raise StudyRouteError('Revision must target the current unit with demonstrated review needs')
    if target_unit['order'] > unit['order']:
        raise StudyRouteError('A branch cannot skip forward in NCERT order')
    runtime = dict(state.runtime_json or {})
    if runtime.get('branch'):
        raise StudyRouteError('Finish the active intervention before branching again')
    runtime['branch'] = {'unit_id': state.current_unit_id, 'step_index': state.current_step_index,
                         'position': state.last_position_json or {}, 'reason': explanation}
    runtime.pop('check', None)
    state.runtime_json = runtime
    state.override_reason, state.override_until_unit_id = reason, target
    state.current_unit_id = target
    state.last_position_json = {}
    state.status = 'in_progress'


def branch_route(db, *, user_id, reason, target_unit_id):
    if reason not in {'prerequisite_gap', 'revision_required'}:
        raise StudyRouteError('Unsupported intervention reason')
    state = _get_state(db, user_id, lock=True)
    if not state:
        raise StudyRouteError('No saved learning route')
    portfolio = _build_portfolio_from_state(db, state)
    _set_branch(state, _chapter(portfolio, state), reason, target_unit_id,
                'Review this unit before returning to the original learning route.')
    db.commit()
    return read_current_route(db, user_id=user_id)


def complete_today(db, *, user_id):
    state = _get_state(db, user_id, lock=True)
    if not state:
        raise StudyRouteError('No saved learning route')
    # Stopping for the day never claims that unfinished units are mastered.
    if state.status != 'completed':
        state.status = 'paused'
    state.last_active_at = _now()
    db.commit()
    return read_current_route(db, user_id=user_id)


def record_heartbeat(db, *, user_id, last_position=None):
    state = _get_state(db, user_id, lock=True)
    if not state:
        raise StudyRouteError('No saved learning route')
    position = dict(last_position or {})
    if position.get('unit_id') and position['unit_id'] != state.current_unit_id:
        raise StudyRouteError('Position belongs to a different unit')
    if state.status == 'completed':
        return read_current_route(db, user_id=user_id)
    state.last_position_json = {**(state.last_position_json or {}), **position}
    state.status = 'in_progress'
    state.started_at = state.started_at or _now()
    state.last_active_at = _now()
    db.commit()
    return read_current_route(db, user_id=user_id)
