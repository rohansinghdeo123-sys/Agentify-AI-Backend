"""Future source releases cannot publish a topic with no reader content."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database import Base
from models import ContentChapter, ContentConcept, ContentPage
from Logic.content_pipeline import approve_chapter, publish_chapter, build_coverage_report


@pytest.fixture
def db():
    engine=create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def chapter_with_topic(db,subject,**teaching):
    chapter=ContentChapter(slug='future_'+subject.lower(),subject=subject,class_level='11',
        chapter_number=1,chapter_name='Reviewed future chapter',status='validated',
        source_hash='reviewed-source',concept_count=1)
    db.add(chapter);db.flush()
    db.add(ContentPage(chapter_id=chapter.id,page_number=1,text='Verified source text',char_count=100,extraction_quality=1.0))
    topic=ContentConcept(chapter_id=chapter.id,concept_id='future-topic',title='Source topic',source_pages=[1],**teaching)
    db.add(topic);db.commit()
    return chapter,topic


@pytest.mark.parametrize('subject',['Chemistry','Physics','Mathematics'])
def test_empty_topic_cannot_be_approved_or_published_even_with_full_page_coverage(db,subject):
    chapter,topic=chapter_with_topic(db,subject,definition=' ',core_explanation='\n',key_points=[' '])
    for action in [approve_chapter,publish_chapter]:
        with pytest.raises(ValueError,match='not ready for approval'):
            action(db,chapter.id)
        assert chapter.status=='validated'
        assert chapter.validation_report['coverage_score']==1.0
        assert chapter.validation_report['notes_ready'] is False
        assert chapter.validation_report['missing_teaching_concept_ids']==[topic.concept_id]
        assert any(issue.get('check')=='notes_teaching_coverage' for issue in chapter.validation_report['issues'])


def test_fixing_source_content_clears_only_the_current_notes_blocker(db):
    chapter,topic=chapter_with_topic(db,'Physics')
    with pytest.raises(ValueError):
        approve_chapter(db,chapter.id)
    topic.definition='Acceleration is the rate of change of velocity.'
    db.flush()
    result=publish_chapter(db,chapter.id,published_by='reviewer')
    assert result.status=='published'
    assert result.validation_report['notes_ready'] is True
    assert result.validation_report['missing_teaching_concept_ids']==[]
    assert not any(issue.get('check')=='notes_teaching_coverage' for issue in result.validation_report['issues'])
    assert result.published_source_hash==result.source_hash


@pytest.mark.parametrize('teaching',[
    {'definition':'A source definition.'},
    {'core_explanation':'A verified source explanation.'},
    {'key_points':['A verified key idea.']},
    {'formulas':['F = ma']},
    {'examples':['Given m = 2 kg and a = 3 m s⁻², F = 6 N.']},
    {'properties':['The leading coefficient is nonzero.']},
    {'applications':['Quadratic equations can model rectangular area.']},
    {'common_mistakes':[{'mistake':'Using zero as the leading coefficient.','correction':'The leading coefficient must be nonzero.'}]},
])
def test_every_existing_reader_teaching_field_can_supply_a_future_topic(db,teaching):
    chapter,_=chapter_with_topic(db,'Mathematics',**teaching)
    approved=approve_chapter(db,chapter.id)
    assert approved.validation_report['notes_ready'] is True
    assert approved.validation_report['ready_for_approval'] is True


def test_notes_check_never_clears_unrelated_source_validation_errors():
    report=build_coverage_report([{'page_number':1,'char_count':100,'extraction_quality':1}],
        [{'concept_id':'valid','source_pages':[1],'definition':'An actual definition.'}],[],
        [{'severity':'error','message':'Damaged source notation'}])
    assert report['notes_ready'] is True
    assert report['ready_for_approval'] is False
    assert report['blocking_issue_count']==1


@pytest.mark.parametrize('concept_ids',[[None],[''],['  '],['net-force','net_force']])
def test_missing_or_ambiguous_source_ids_cannot_publish_blank_destinations(concept_ids):
    report=build_coverage_report([{'page_number':1,'char_count':100,'extraction_quality':1}],
        [{'concept_id':key,'source_pages':[1],'definition':'Net force is the vector sum of the forces.'} for key in concept_ids],[])
    assert report['notes_ready'] is False
    assert report['ready_for_approval'] is False
    assert report['invalid_notes_topic_ids']
