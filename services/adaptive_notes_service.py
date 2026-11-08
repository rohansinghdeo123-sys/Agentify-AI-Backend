"""Universal source-first notes, shared persisted variants and private reading anchors."""
from __future__ import annotations
from copy import deepcopy
import re
from sqlalchemy.exc import IntegrityError
from models import AdaptiveNoteLesson, AICoachProfile
from services.notes_material_service import retrieve_material, digest
from services.notes_scope_service import resolve_notes_curriculum, resolve_notes_topics, same_scope, identity

EDITION = 'adaptive_notes_v2'


def record_reading_position(db, state, position, curriculum, topic):
    version, index = position.get('notes_version'),position['notes_page']
    row = db.query(AdaptiveNoteLesson).filter_by(content_version=version,curriculum_key=curriculum['curriculum_key'],unit_id=topic).first()
    if not row:
        raise ValueError('Unknown or unscoped notes edition; reopen this topic')
    pages = row.payload_json['pages']
    if not 0 <= index < len(pages):
        raise ValueError('Notes page is outside this generated lesson')
    page = pages[index]
    if position.get('notes_page_id') and position['notes_page_id'] != page['id']:
        raise ValueError('Notes page identity changed; reopen the current lesson')
    if 'notes_total_pages' in position and (type(position['notes_total_pages']) is not int or position['notes_total_pages'] != len(pages)):
        raise ValueError('Notes page count does not match its source-bound edition')
    completed = position.get('notes_completed_page_ids',[])
    valid = {p['id'] for p in pages}
    if not isinstance(completed,list) or len(completed)>len(pages) or any(not isinstance(p,str) or p not in valid for p in completed):
        raise ValueError('Completed pages must belong to this notes edition')
    runtime = deepcopy(state.runtime_json or {})
    ledger = runtime.setdefault('notes_reading_by_version',{}).setdefault(version,{})
    prior = ledger.get(topic,{})
    merged = [p['id'] for p in pages if p['id'] in set(prior.get('completed_page_ids',[]))|set(completed)]
    navigation_id, sequence = position.get('notes_navigation_id'),position.get('notes_navigation_seq')
    if sequence is not None and (type(sequence) is not int or sequence < 0):
        raise ValueError('Invalid notes navigation sequence')
    stale = bool(navigation_id and navigation_id == prior.get('navigation_id') and sequence is not None and sequence <= prior.get('navigation_seq',-1))
    if stale:
        prior['completed_page_ids'] = merged
        ledger[topic] = prior
    else:
        ledger[topic] = {'page_id':page['id'],'page_index':index,'completed_page_ids':merged,
                         'navigation_id':navigation_id,'navigation_seq':sequence if sequence is not None else 0}
        runtime.setdefault('notes_anchor_by_unit',{})[topic] = {'content_anchor':page.get('content_anchor'),
            'source_ids':[r['id'] for r in page.get('source_refs',[])],'subtopic_ids':page['source_section_ids']}
        runtime.setdefault('notes_positions_by_version',{}).setdefault(version,{})[topic] = index
    state.runtime_json = runtime


def learner_settings(db, user_id, state, curriculum, unit):
    proficiency, style, allocated = 'new_to_it','balanced',None
    if state and same_scope(curriculum,state.subject,state.class_level):
        selected = next((s for s in state.selections_json or [] if (resolve_notes_curriculum(db,s.get('chapter_ref'),subject=state.subject,class_level=state.class_level) or {}).get('curriculum_key') == curriculum['curriculum_key']),None)
        if selected:
            proficiency = selected.get('chapter_proficiency') or proficiency
            from services.study_route_service import _build_portfolio_from_state
            portfolio = _build_portfolio_from_state(db,state)
            chapter = next((c for c in portfolio['chapters'] if c['curriculum_key'] == curriculum['curriculum_key']),{})
            proficiency = chapter.get('effective_proficiency',{}).get('effective',proficiency)
            allocated = sum(i['minutes'] for i in (state.runtime_json or {}).get('today_route',{}).get('items',[])
                            if i.get('unit_id') == unit['id'] and i.get('role') == 'main_focus') or None
    profile = db.query(AICoachProfile).filter_by(user_id=user_id).first()
    prefs = profile.study_preferences or {} if profile else {}
    preference = str(prefs.get('preferred_style') or prefs.get('answer_style') or prefs.get('feedback_depth') or '').lower()
    if any(w in preference for w in ['concise','brief','short']): style = 'concise'
    elif any(w in preference for w in ['detail','step_by_step','deep']): style = 'detailed'
    return proficiency,style,allocated


def compiled_lesson(db, curriculum, unit, *, proficiency='new_to_it',style='balanced',material=None,requested_version=None):
    from services.adaptive_content_compiler import compile_adaptive_lesson, COMPILER_VERSION
    material = material or retrieve_material(db,curriculum,unit)
    duration = unit['estimated_minutes']
    key = digest([EDITION,COMPILER_VERSION,curriculum['curriculum_key'],unit['id'],proficiency,style,
                  material['source_fingerprint'],duration])
    version = EDITION+'_'+key[:40]
    if requested_version:
        pinned = db.query(AdaptiveNoteLesson).filter_by(content_version=requested_version,curriculum_key=curriculum['curriculum_key'],unit_id=unit['id']).first()
        if (pinned and pinned.proficiency == proficiency and pinned.explanation_style == style
                and pinned.payload_json.get('compiler_version') == COMPILER_VERSION
                and pinned.source_fingerprint == material['source_fingerprint']
                and (pinned.minimum_minutes,pinned.maximum_minutes) == (duration['min'],duration['max'])):
            return deepcopy(pinned.payload_json),material
    row = db.query(AdaptiveNoteLesson).filter_by(content_version=version).first()
    if row:
        return deepcopy(row.payload_json),material
    if not material['sections']:
        return {'pages':[],'content_version':version,'resolution':'review_required','proficiency':proficiency,
                'explanation_style':style,'source_fingerprint':material['source_fingerprint'],
                'review_required':material['conflicts'] or [{'reason':'The selected mapped subtopics need an indexed, reviewed source section.'}]},material
    payload = compile_adaptive_lesson(material,proficiency=proficiency,style=style,duration=duration)
    payload.update(content_version=version,resolution='lesson',learning_depth=payload['adaptation']['depth'],
        source_provenance=[{k:s.get(k) for k in ['id','title','source_type','source_document','source_pages','subtopic_ids']} for s in material['sections']],
        review_required=[*material['conflicts'],*([{'reason':'Additional mapped source sections are still needed.','subtopic_ids':material['missing_subtopics']}] if material['missing_subtopics'] else [])])
    row = AdaptiveNoteLesson(content_version=version,curriculum_key=curriculum['curriculum_key'],chapter_slug=curriculum['chapter_slug'],
        unit_id=unit['id'],proficiency=proficiency,explanation_style=style,source_fingerprint=material['source_fingerprint'],
        minimum_minutes=duration['min'],maximum_minutes=duration['max'],payload_json=payload)
    try:
        with db.begin_nested():
            db.add(row); db.flush()
    except IntegrityError:
        row = db.query(AdaptiveNoteLesson).filter_by(content_version=version).one()
        payload = row.payload_json
    return deepcopy(payload),material


def reading_progress(state, unit, payload):
    pages = payload['pages']
    if not pages:
        return {'saved_page':0,'saved_page_id':None,'completed_page_ids':[],'progress_rebased':False}
    runtime = state.runtime_json or {} if state else {}
    version = payload['content_version']
    saved = runtime.get('notes_reading_by_version',{}).get(version,{}).get(unit['id'],{})
    page_ids = [p['id'] for p in pages]
    current = saved.get('page_id')
    completed = [p for p in page_ids if p in saved.get('completed_page_ids',[])]
    rebased = False
    if current not in page_ids:
        anchor = runtime.get('notes_anchor_by_unit',{}).get(unit['id'],{})
        current = next((p['id'] for p in pages if p.get('content_anchor') == anchor.get('content_anchor')),None)
        if not current and anchor:
            current = next((p['id'] for p in pages if {r['id'] for r in p.get('source_refs',[])}.intersection(anchor.get('source_ids',[]))),None)
        if not current and anchor.get('subtopic_ids'):
            current = next((p['id'] for p in pages if set(p['source_section_ids']).intersection(anchor['subtopic_ids'])),None)
        rebased = bool(anchor)
        if not current and not anchor:
            old = runtime.get('notes_positions_by_version',{}).get('ncert_teacher_v1',{}).get(unit['id'])
            if type(old) is int and 0 <= old < 10:
                from services.teacher_notes_service import _load_lesson
                try:
                    ids = set(_load_lesson(unit['id'])['pages'][old]['source_section_ids'])
                    current = next((p['id'] for p in pages if ids.intersection(p['source_section_ids'])),None)
                    rebased = bool(current)
                except (ValueError,KeyError,FileNotFoundError):
                    pass
        current = current or page_ids[0]
    return {'saved_page':page_ids.index(current),'saved_page_id':current,'completed_page_ids':completed,'progress_rebased':rebased}


def session_portion(pages, unit, *, start_page=0, session_minutes=None):
    if not pages or session_minutes is None or session_minutes <= 0:
        return None
    start_page = max(0,min(len(pages)-1,int(start_page)))
    weights = [1+sum(b['kind'] in {'worked_example','derivation'} for b in p['blocks'])+len(p.get('interrupts',[]))/2+len(p.get('questions',[]))/3 for p in pages]
    remaining, end = session_minutes,start_page
    for i in range(start_page,len(pages)):
        share = unit['estimated_minutes']['min']*weights[i]/sum(weights)
        if i>start_page and share>remaining+1e-9: break
        end = i; remaining -= share
        if remaining <= 0: break
    return {'start_page':start_page,'end_page':end,'allocated_minutes':session_minutes,
            'additional_practice_needed':end<len(pages)-1 or remaining < -1e-9}


def notes_response(db, chapter_ref, topic_id, user_id, *, subject=None,class_level=None,content_version=None):
    curriculum = resolve_notes_curriculum(db,chapter_ref,subject=subject,class_level=class_level)
    if not curriculum:
        return None
    from services.study_route_service import _get_state, _today, refresh_notes_session_anchor
    state = _get_state(db,user_id)
    if state and (state.runtime_json or {}).get('notes_session_date') != _today().isoformat():
        state = _get_state(db,user_id,lock=True); refresh_notes_session_anchor(db,state)
    units = resolve_notes_topics(db,curriculum,topic_id,state.current_unit_id if state else None)
    if not units:
        from fastapi import HTTPException
        raise HTTPException(404,detail='This topic has no unambiguous binding to the selected chapter. Reopen it from Planning.')
    chapter = {'slug':curriculum['chapter_slug'],'name':curriculum['chapter_title'],'class_level':curriculum['class_level'],
               'subject':curriculum['subject'],'chapter_number':curriculum['chapter_number']}
    if len(units)>1:
        return {'chapter':chapter,'resolution':'topic_choice','requested_topic_id':topic_id,'topic_title':'Choose a learning topic',
                'pages':[],'concepts':[],'can_save_position':False,'topic_options':[{'id':u['id'],'title':u['title'],
                'estimated_minutes':dict(u['estimated_minutes'])} for u in units]}
    unit = units[0]
    proficiency,style,allocation = learner_settings(db,user_id,state,curriculum,unit)
    payload, material = compiled_lesson(db,curriculum,unit,proficiency=proficiency,style=style,requested_version=content_version)
    progress = reading_progress(state,unit,payload)
    if not content_version and progress['saved_page'] == 0 and topic_id:
        chosen = [s['id'] for s in unit['ncert_subtopics'] if identity(topic_id) == identity(s['id']) and topic_id != unit['id']]
        if chosen:
            entry = next((i for i,p in enumerate(payload['pages']) if set(chosen).intersection(p['source_section_ids'])),0)
            if payload['pages']:
                progress.update(saved_page=entry,saved_page_id=payload['pages'][entry]['id'])
    selected = bool(state and same_scope(curriculum,state.subject,state.class_level) and any((resolve_notes_curriculum(db,s.get('chapter_ref'),subject=state.subject,class_level=state.class_level) or {}).get('curriculum_key') == curriculum['curriculum_key'] for s in state.selections_json or []))
    runtime = state.runtime_json or {} if state else {}
    starts = runtime.get('notes_session_start_positions',{}).get(payload['content_version'],{})
    if selected and unit['id'] not in starts:
        # Freeze the day's first reading anchor. Advancing a page must not
        # silently advance the whole recommended session window on reload.
        state = _get_state(db,user_id,lock=True)
        runtime = deepcopy(state.runtime_json or {})
        starts = runtime.setdefault('notes_session_start_positions',{}).setdefault(payload['content_version'],{})
        starts.setdefault(unit['id'],progress['saved_page'])
        state.runtime_json = runtime
    start = starts.get(unit['id'],progress['saved_page'])
    return {**payload,**progress,'chapter':chapter,'topic_id':unit['id'],'topic_title':unit['title'],'requested_topic_id':topic_id,
        'estimated_minutes':dict(unit['estimated_minutes']),'subtopics':unit['ncert_subtopics'],
        'catalog_source':'published' if curriculum['source']['authority']=='approved_content_pipeline' else 'planning_manifest',
        'active_unit_id':state.current_unit_id if state else None,'can_save_position':selected,'allocated_session_minutes':allocation,
        'session_portion':session_portion(payload['pages'],unit,start_page=start,session_minutes=allocation),
        'source_types':list(dict.fromkeys(s['source_type'] for s in material['sections'])),'concepts':[]}


def retrieve_context(db, chapter_ref, topic_id, *, subject=None,class_level=None,question='',subtopic_ids=None,page_number=None,page_id=None,content_version=None,proficiency='new_to_it',style='balanced'):
    curriculum = resolve_notes_curriculum(db,chapter_ref,subject=subject,class_level=class_level)
    if not curriculum: return None
    units = resolve_notes_topics(db,curriculum,topic_id)
    if len(units) != 1: return None
    unit = units[0]
    allowed = {s['id'] for s in unit['ncert_subtopics']}
    requested = set(subtopic_ids or allowed)
    if not requested <= allowed: raise ValueError('Subtopics outside selected Planning topic')
    material = retrieve_material(db,curriculum,unit,requested if not (page_id or page_number) else None)
    payload,_ = compiled_lesson(db,curriculum,unit,proficiency=proficiency,style=style,material=material,requested_version=content_version)
    pages = payload['pages']
    if page_id:
        pages = [p for p in pages if p['id'] == page_id]
    elif page_number is not None:
        if type(page_number) is not int or page_number < 1: raise ValueError('Invalid notes page')
        pages = [p for p in pages if p['page_number'] == page_number]
    pages = [p for p in pages if set(p['source_section_ids']) <= requested]
    if not (page_id or page_number) and question:
        terms = set(re.findall(r'[a-z]{3,}',question.lower()))
        pages = sorted(pages,key=lambda p:-len(terms.intersection(re.findall(r'[a-z]{3,}',p['text'].lower()))))[:4]
        pages.sort(key=lambda p:p['page_number'])
    context = '\n\n'.join('['+p['source']+'; '+p['title']+']\n'+'\n\n'.join(
        '['+b['source_type']+'] '+(b.get('expanded_text') or b['text']) for b in p['blocks']) for p in pages)
    return {'context':context,'source':'adaptive_source_notes','retrieval_mode':'master_first_scoped_notes',
        'unit_id':unit['id'],'section_id':unit['primary_topic_id'],'paragraphs_found':len(pages),
        'keywords_used':sorted(set(re.findall(r'[a-z]{3,}',question.lower()))),'matched_sections':[p['id'] for p in pages],
        'source_pages':sorted({n for p in pages for n in p['source_pages']}),'content_order_locked':True,
        'provenance':[r for p in pages for r in p['source_refs']],'source_types':payload.get('source_types',[]),
        'conflicts':material['conflicts'],'missing_subtopics':material['missing_subtopics']}


def context_for_scope(scope, topic_id, question):
    """Shared tutor/tool entry point; uses an independent read/cache transaction."""
    if scope.get('catalog_source') not in {'planning_manifest','published'}:
        return None
    chapter_ref = scope.get('chapter_slug') or scope.get('chapter')
    if not chapter_ref:
        return None
    version = str(scope.get('notes_content_version') or '')
    if version == 'ncert_teacher_v1':
        from services.teacher_notes_service import retrieve_teacher_context
        return retrieve_teacher_context(chapter_ref,topic_id,question=question,
            subtopic_ids=scope.get('subtopic_ids'),page_number=scope.get('notes_page_number'),
            proficiency=scope.get('notes_proficiency') or 'new_to_it',style=scope.get('notes_explanation_style') or 'balanced')
    if scope.get('catalog_source') == 'published' and not version.startswith(EDITION):
        # Revision/Exam keep their existing approved hybrid retrieval. Only a
        # notes-bound tutor request adopts its exact source-bound page context.
        return None
    from database import SessionLocal
    db = SessionLocal()
    try:
        curriculum = resolve_notes_curriculum(db,chapter_ref,subject=scope.get('subject'),class_level=scope.get('class_level'))
        if not curriculum:
            return None
        units = resolve_notes_topics(db,curriculum,topic_id)
        if len(units) != 1:
            return None
        result = retrieve_context(db,chapter_ref,units[0]['id'],subject=scope.get('subject'),class_level=scope.get('class_level'),
            question=question,subtopic_ids=scope.get('subtopic_ids'),
            page_number=scope.get('notes_page_number') if version.startswith(EDITION) else None,
            page_id=scope.get('notes_page_id') if version.startswith(EDITION) else None,
            content_version=version if version.startswith(EDITION) else None,
            proficiency=scope.get('notes_proficiency') or 'new_to_it',style=scope.get('notes_explanation_style') or 'balanced')
        db.commit()
        return result
    finally:
        db.close()
