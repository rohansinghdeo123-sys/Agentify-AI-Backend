"""Manual topic ranges are authoritative; a session budget never rewrites them."""
from copy import deepcopy
from models import TopicDurationOverride


def apply_topic_durations(db, curriculum):
    result = deepcopy(curriculum)
    if db is None:
        return result
    overrides = {r.unit_id:r for r in db.query(TopicDurationOverride).filter_by(curriculum_key=curriculum['curriculum_key']).all()}
    for unit in result['units']:
        if unit['id'] in overrides:
            row = overrides[unit['id']]
            unit['estimated_minutes'] = {'min':row.minimum_minutes,'max':row.maximum_minutes}
            unit['duration_source'] = 'manual'
    return result


def set_topic_duration(db, *, curriculum, unit_id, minimum, maximum, actor_uid):
    if type(minimum) is not int or type(maximum) is not int or not 5 <= minimum <= maximum <= 600:
        raise ValueError('Supply separate ordered minimum/maximum minute values between 5 and 600')
    unit = next((u for u in curriculum['units'] if u['id'] == unit_id), None)
    if not unit:
        raise ValueError('The duration topic is outside the selected chapter')
    # These are the explicit Chapter 1 administrator values from the prompt.
    # Future topics remain configurable without changing this reviewed baseline.
    if curriculum['curriculum_key'] == 'ncert_class_11_chemistry_chapter_01':
        from Logic.planning.curriculum_registry import resolve_planning_curriculum
        canonical = resolve_planning_curriculum(chapter_ref=curriculum['chapter_slug'],subject=curriculum['subject'],class_level=curriculum['class_level'])
        baseline = next(u['estimated_minutes'] for u in canonical['units'] if u['id'] == unit_id)
        if {'min':minimum,'max':maximum} != baseline:
            raise ValueError('The six reviewed Chapter 1 ranges must remain unchanged')
    row = db.query(TopicDurationOverride).filter_by(curriculum_key=curriculum['curriculum_key'],unit_id=unit_id).first()
    if not row:
        row = TopicDurationOverride(curriculum_key=curriculum['curriculum_key'],unit_id=unit_id)
        db.add(row)
    row.minimum_minutes, row.maximum_minutes, row.configured_by = minimum, maximum, actor_uid
    db.flush()
    return {'unit_id':unit_id,'estimated_minutes':{'min':minimum,'max':maximum},'duration_source':'manual'}
