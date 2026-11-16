"""Future-subject contract through real catalog, notes, source and route services.

The approved Physics/Mathematics records below are realistic test fixtures,
not a claim that those future textbooks have already been uploaded/published.
No model, embedding provider, Chemistry assets or live student data is used.
"""
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.security import verify_firebase_user
from database import Base, get_db
from models import AdaptiveNoteLesson, AICoachProfile, ContentChapter, ContentConcept, TestHistory as History
from routers.study import router as study_router
from services.adaptive_content_compiler import PROFICIENCIES
from services.adaptive_notes_service import compiled_lesson, retrieve_context
from services.notes_config_service import set_topic_duration
from services.notes_material_service import retrieve_material
from services.notes_scope_service import resolve_notes_curriculum
from services.study_route_service import start_route, record_heartbeat, _get_state


PHYSICS = [
    ('velocity','Velocity and displacement','Velocity is displacement per unit time.',
     'Displacement connects an initial position to a final position. Its direction matters. A change of direction can change velocity even if speed remains steady. Reading a motion description therefore requires both magnitude and direction.',
     'Average velocity = displacement / elapsed time.',
     'A displacement of 12 m east in 3 s gives an average velocity of 4 m s⁻¹ east.'),
    ('acceleration','Acceleration and changing velocity','Acceleration is the rate of change of velocity.',
     'Acceleration describes a change of velocity rather than simply fast motion. Both speed and direction belong to velocity. Straight motion can accelerate when speed changes. Uniform circular motion accelerates because its velocity direction changes.',
     'Average acceleration = change in velocity / elapsed time.',
     'A velocity change from 2 m s⁻¹ to 8 m s⁻¹ in 3 s gives 2 m s⁻² along the stated positive direction.'),
    ('inertia','Inertia and mass','Inertia is resistance to a change of an object’s state of motion.',
     'Mass measures inertia. The same push changes the motion of a loaded trolley less readily than that of an empty trolley. This comparison connects resistance to acceleration with mass. Motion itself does not require a continuing unbalanced push.',
     '', 'An empty trolley responds more readily than an otherwise similar loaded trolley to the same net force.'),
    ('first_law','Newton’s first law','An object retains rest or uniform straight-line motion unless a net external force acts.',
     'The law describes the zero-net-force case. Individual forces can still be present when they balance. A moving object with no net force keeps a constant velocity. A change of velocity signals an unbalanced interaction.',
     'For constant mass, zero net force implies zero acceleration.',
     'A book resting on a table has balanced upward support and downward gravitational force.'),
    ('second_law','Newton’s second law','The net force equals the rate of change of momentum.',
     'Use the combined external force rather than one arbitrarily selected force. For constant mass the law links force with acceleration. The acceleration direction follows the net force. Units provide a check on the relationship.',
     'For constant mass, F = ma. F is in N, m in kg, and a in m s⁻².',
     'Given a constant mass of 2 kg and net force 6 N, a = F/m = 6/2 = 3 m s⁻².'),
    ('momentum','Linear momentum','Linear momentum is the product of mass and velocity.',
     'Momentum has the direction of velocity. Comparing moving objects requires their masses as well as their velocities. A larger mass at the same velocity has larger momentum. A reversal of velocity reverses momentum direction.',
     'p = mv; the SI unit is kg m s⁻¹.',
     'A 3 kg object moving east at 2 m s⁻¹ has momentum 6 kg m s⁻¹ east.'),
    ('third_law','Newton’s third law','Interaction forces on two different bodies are equal in magnitude and opposite in direction.',
     'The pair belongs to one interaction. Its members act on different objects. They therefore do not cancel in the force balance of just one object. Identify both interacting objects before drawing the pair.',
     '', 'When a hand pushes a wall, the wall pushes the hand with an equal opposite force.'),
    ('friction','Static and kinetic friction','Friction opposes relative sliding or its tendency between contacting surfaces.',
     'Static friction can adjust while surfaces do not slide. Kinetic friction applies during sliding. A limiting static-friction value is a bound rather than the value in every resting situation. Decide whether sliding occurs before choosing a friction model.',
     'The simple model gives limiting static friction f_max = μ_s N.',
     'A stationary book on a level table needs no horizontal friction when no horizontal force tends to move it.'),
]

MATHEMATICS = [
    ('definition','Quadratic equations','A quadratic equation in one variable has degree two.',
     'Write the equation in standard form before identifying its coefficients. The leading coefficient must be nonzero. Terms may need to be collected onto one side. The highest remaining power determines the degree.',
     'ax² + bx + c = 0, where a ≠ 0.', 'For x² − 5x + 6 = 0, the coefficients are a = 1, b = −5 and c = 6.'),
    ('roots','Roots and verification','A root makes the equation true when substituted for the variable.',
     'Finding a candidate answer is only part of solving an equation. Substitution checks whether it satisfies the original statement. A quadratic can have two distinct real roots, a repeated real root or no real root. Keep the number system clear when describing roots.',
     '', 'For x² − 5x + 6 = 0, substituting x = 2 gives 4 − 10 + 6 = 0.'),
    ('factorisation','Solving by factorisation','Factorisation rewrites an expression as a product.',
     'Set the equation equal to zero before applying the zero-product property. One factor must be zero when a product is zero. Solve each linear factor separately. Verify both candidates in the original equation.',
     '(x − 2)(x − 3) = 0 implies x = 2 or x = 3.',
     'Given x² − 5x + 6 = 0, factor to (x − 2)(x − 3) = 0. The roots are 2 and 3.'),
    ('square','Completing the square','Completing the square rewrites a quadratic using a squared binomial.',
     'Balance every addition on both sides of the equation. A perfect square exposes how the unknown can be isolated. Taking a square root introduces both signs when the right side is positive. Check the original equation after obtaining the candidates.',
     'x² − 6x + 9 = (x − 3)².',
     'For x² − 6x + 5 = 0, obtain (x − 3)² = 4, so x = 3 ± 2, giving 1 and 5.'),
    ('formula','The quadratic formula','The quadratic formula solves ax² + bx + c = 0 when a ≠ 0.',
     'Standard form identifies the coefficients with their signs. Parentheses protect negative values during substitution. The square-root term determines whether real roots exist. The denominator applies to the entire numerator.',
     'x = (−b ± √(b² − 4ac))/(2a).',
     'Given x² − 5x + 6 = 0, x = (5 ± √(25 − 24))/2 = (5 ± 1)/2, giving 2 and 3.'),
    ('discriminant','The discriminant','The discriminant is the expression b² − 4ac.',
     'Its sign classifies real roots before they are calculated. A positive value gives two distinct real roots. Zero gives a repeated real root. A negative value gives no real roots.',
     'D = b² − 4ac.', 'For x² + x + 1 = 0, D = 1 − 4 = −3, so there are no real roots.'),
    ('relations','Sum and product of roots','The coefficients determine the sum and product of the roots.',
     'The coefficient signs matter in both relationships. The leading coefficient is a divisor. These checks can expose an algebra error without resolving the equation. Use the standard form with all terms on the same side.',
     'For roots α and β, α + β = −b/a and αβ = c/a.',
     'For x² − 5x + 6 = 0, roots 2 and 3 have sum 5 and product 6.'),
    ('application','Area problems and valid roots','A mathematical solution must also satisfy the original situation.',
     'Define the unknown and its unit before forming an equation. Translate the area relationship carefully. Algebraic roots are candidates rather than automatically valid lengths. Reject a root that violates a stated physical constraint.',
     'The area of a rectangle equals length × breadth.',
     'A square of area 9 m² has side satisfying x² = 9. The algebraic candidates are ±3, and the valid side length is 3 m.'),
]


@pytest.fixture(params=[('Physics','Class 11','laws_of_motion',PHYSICS),
                        ('Mathematics','Class 10','quadratic_equations',MATHEMATICS)],ids=['physics','mathematics'])
def future(request):
    subject,level,slug,content=request.param
    engine=create_engine('sqlite://',poolclass=StaticPool,connect_args={'check_same_thread':False})
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        chapter=ContentChapter(board='NCERT',class_level=level,subject=subject,chapter_number=4,
            chapter_name=slug.replace('_',' ').title(),slug='future_'+slug,status='published',
            page_count=16,extracted_page_count=16,source_hash='reviewed-fixture-v1',published_source_hash='reviewed-fixture-v1')
        db.add(chapter);db.flush()
        concepts={}
        for i,(key,title,definition,explanation,formula,example) in enumerate(content,1):
            row=ContentConcept(chapter_id=chapter.id,concept_id=slug+'_'+key,title=title,
                definition=definition,core_explanation=explanation,formulas=[formula] if formula else [],
                examples=[example],key_points=[definition],source_pages=[i],difficulty_level=3,
                importance_level='high',typical_exam_weightage='high')
            db.add(row);concepts[row.concept_id]=row
        db.commit()
        curriculum=resolve_notes_curriculum(db,chapter.slug,subject=subject,class_level=level)
        assert len(curriculum['units'])>=2
        app=FastAPI();app.include_router(study_router)
        app.dependency_overrides[get_db]=lambda:db
        app.dependency_overrides[verify_firebase_user]=lambda:{'uid':'future-student'}
        with TestClient(app) as client:
            yield db,client,chapter,curriculum,concepts
    engine.dispose()




def _get(client,chapter,unit,**params):
    response=client.get('/notes/'+chapter.slug,params={'topic_id':unit['id'],'subject':chapter.subject,
        'class_level':chapter.class_level,**params})
    assert response.status_code==200,response.text
    return response.json()


def test_future_subject_every_topic_subtopic_and_all_nine_views_are_scoped(future):
    db,client,chapter,curriculum,concepts=future
    for unit in curriculum['units']:
        allowed={s['id'] for s in unit['ncert_subtopics']}
        for proficiency in PROFICIENCIES:
            for style in ['balanced','concise','detailed']:
                payload,material=compiled_lesson(db,curriculum,unit,proficiency=proficiency,style=style)
                assert not material['missing_subtopics'] and not material['conflicts']
                assert payload['pages'] and not payload.get('review_required')
                assert {s for p in payload['pages'] for s in p['source_section_ids']}==allowed
                assert all(p['blocks'] and p['text'].strip() for p in payload['pages'])
                assert [p['page_number'] for p in payload['pages']]==list(range(1,len(payload['pages'])+1))
                assert payload['proficiency']==proficiency and payload['explanation_style']==style
        notes=_get(client,chapter,unit)
        assert notes['resolution']=='lesson' and notes['chapter']['subject']==chapter.subject
        assert notes['source_types']==['NCERT']
        assert not notes['review_required']
        assert all('Chemistry' not in p['text'] for p in notes['pages'])
        for subtopic in unit['ncert_subtopics']:
            result=_get(client,chapter,unit,topic_id=subtopic['id'])
            assert result['topic_id']==unit['id']
            assert subtopic['id'] in result['pages'][result['saved_page']]['source_section_ids']
    db.commit()
    assert db.query(AdaptiveNoteLesson).count()==9*len(curriculum['units'])



def test_future_planning_notes_progress_exact_ask_ai_and_manual_durations(future):
    db,client,chapter,curriculum,concepts=future
    first=curriculum['units'][0]
    before=deepcopy(curriculum['units'][1]['estimated_minutes'])
    set_topic_duration(db,curriculum=curriculum,unit_id=first['id'],minimum=45,maximum=55,actor_uid='admin')
    db.commit()
    start_route(db,user_id='future-student',class_level=chapter.class_level,subject=chapter.subject,
        selections=[{'chapter_ref':chapter.slug,'chapter_proficiency':'know_a_little'}],study_time_today='15')
    db.add(AICoachProfile(user_id='future-student',coach_id='future-subject',study_preferences={'preferred_style':'detailed'}));db.commit()
    notes=_get(client,chapter,first)
    assert notes['active_unit_id']==first['id'] and notes['can_save_position']
    assert notes['proficiency']=='new_to_it' and notes['explanation_style']=='detailed'
    assert notes['estimated_minutes']=={'min':45,'max':55}
    assert notes['allocated_session_minutes']<=15
    last=notes['pages'][-1];index=len(notes['pages'])-1
    record_heartbeat(db,user_id='future-student',compact=True,last_position={'unit_id':first['id'],'notes_topic':first['id'],
        'notes_page':index,'notes_page_id':last['id'],'notes_total_pages':len(notes['pages']),
        'notes_version':notes['content_version'],'notes_completed_page_ids':[last['id']]})
    db.expire_all()
    reopened=_get(client,chapter,first,content_version=notes['content_version'])
    assert reopened['saved_page_id']==last['id'] and reopened['completed_page_ids']==[last['id']]
    context=retrieve_context(db,chapter.slug,first['id'],subject=chapter.subject,class_level=chapter.class_level,
        page_id=last['id'],content_version=notes['content_version'],subtopic_ids=last['source_section_ids'],
        proficiency=notes['proficiency'],style=notes['explanation_style'])
    assert context['matched_sections']==[last['id']] and context['context']
    assert all((b.get('expanded_text') or b['text']) in context['context'] for b in last['blocks'])
    other=_get(client,chapter,curriculum['units'][1])
    assert other['estimated_minutes']==before
    assert db.query(History).count()==0
    current=_get_state(db,'future-student')
    assert current.current_unit_id==first['id']
    with pytest.raises(ValueError):
        retrieve_context(db,chapter.slug,first['id'],subject=chapter.subject,class_level=chapter.class_level,
            subtopic_ids=[curriculum['units'][1]['ncert_subtopics'][0]['id']])



@pytest.mark.parametrize('source_field',['examples','core_explanation'])
def test_future_calculations_keep_source_worked_example_semantics_and_atomic_solutions(future,source_field):
    db,client,chapter,curriculum,concepts=future
    suffix='_second_law' if chapter.subject=='Physics' else '_formula'
    concept=next(c for key,c in concepts.items() if key.endswith(suffix))
    unit=next(u for u in curriculum['units'] if concept.concept_id in {s['id'] for s in u['ncert_subtopics']})
    original=concept.examples[0]
    if source_field=='core_explanation':
        concept.core_explanation+='\n\n'+original
        concept.examples=[]
        db.commit()
    material=retrieve_material(db,curriculum,unit)
    section=next(s for s in material['sections'] if concept.concept_id in s['subtopic_ids'])
    assert section['teaching_format']=='numerical'
    assert any(b['kind']=='worked_example' and b['text']==original for b in section['blocks'])
    for proficiency in PROFICIENCIES:
        lesson,_=compiled_lesson(db,curriculum,unit,proficiency=proficiency,material=material)
        blocks=[b for p in lesson['pages'] for b in p['blocks']]
        assert any(original in (b.get('expanded_text') or b['text']) for b in blocks)
        assert any(b['kind']=='worked_example' and b.get('presentation')!='source_preview' for b in blocks)
