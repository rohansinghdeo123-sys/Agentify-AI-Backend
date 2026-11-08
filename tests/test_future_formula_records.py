"""Structured Physics/Math formulas accepted by ingestion must reach the reader."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import Base
from models import ContentChapter,ContentPage,ContentConcept
from Logic.content_pipeline import publish_chapter
from services.notes_scope_service import resolve_notes_curriculum
from services.notes_material_service import retrieve_material
from services.adaptive_notes_service import compiled_lesson


@pytest.mark.parametrize('subject,record,expression',[
    ('Physics',{'formula':'F = ma','symbols':{'F':'net force','m':'mass','a':'acceleration'},'units':{'F':'N','m':'kg','a':'m s⁻²'},'conditions':'Use constant mass.'},'F = ma'),
    ('Mathematics',{'expression':'D = b² − 4ac','symbols':{'a':'leading coefficient'},'conditions':'a ≠ 0'},'D = b² − 4ac'),
    ('Mathematics',{'latex':r'x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}','conditions':'a ≠ 0'},r'\frac{-b \pm \sqrt{b^2-4ac}}{2a}'),
])
def test_publish_and_render_structured_formula_without_losing_terms_conditions(subject,record,expression):
    engine=create_engine('sqlite://');Base.metadata.create_all(engine)
    with Session(engine) as db:
        chapter=ContentChapter(slug='future_formula',class_level='11',subject=subject,chapter_number=1,
            chapter_name='Source relationships',source_hash='reviewed',concept_count=1,status='validated')
        db.add(chapter);db.flush()
        db.add(ContentPage(chapter_id=chapter.id,page_number=1,text=expression,char_count=100,extraction_quality=1))
        db.add(ContentConcept(chapter_id=chapter.id,concept_id='source_relationship',title='Source relationship',formulas=[record],source_pages=[1]))
        db.flush();publish_chapter(db,chapter.id);db.commit()
        curriculum=resolve_notes_curriculum(db,chapter.slug,subject=subject,class_level='Class 11')
        unit=curriculum['units'][0];material=retrieve_material(db,curriculum,unit)
        assert not material['missing_subtopics']
        for level in ['new_to_it','know_a_little','know_the_basics','mostly_confident']:
            lesson,_=compiled_lesson(db,curriculum,unit,proficiency=level,material=material)
            text='\n'.join(b['text'] for p in lesson['pages'] for b in p['blocks'])
            assert expression in text and record['conditions'] in text
            assert all(value in text for value in record.get('symbols',{}).values())
            assert all(value in text for value in record.get('units',{}).values())
    engine.dispose()


def test_unknown_formula_shape_blocks_publication_instead_of_silent_omission():
    from Logic.content_pipeline import build_coverage_report
    report=build_coverage_report([{'page_number':1,'char_count':10,'extraction_quality':1}],
        [{'concept_id':'relationship','source_pages':[1],'definition':'A useful relationship.','formulas':[{'unrecognized':'F = ma'}]}],[])
    assert not report['ready_for_approval'] and not report['notes_ready']
    assert report['invalid_notes_formula_ids']==['relationship']
