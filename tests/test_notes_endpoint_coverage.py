"""Every released Chemistry unit must survive real HTTP serialization and resume."""
from scripts.audit_notes_endpoints import audit


def test_every_released_chemistry_unit_opens_and_resumes_through_actual_endpoints():
    report=audit(proficiencies=('new_to_it',),styles=('balanced',),aliases=False)
    assert report['errors']==[],report['errors']
    assert report['summary']['chapters']==9
    assert report['summary']['units']==72
    assert report['summary']['variants']==72
    assert report['summary']['progress_roundtrips']==72
    page_counts={count for chapter in report['chapters'] for count in chapter['page_counts']}
    assert len(page_counts)>1 and max(page_counts)>10,'Pages were forced into a fixed ten-page layout'
    assert all(chapter['subtopics']>0 and chapter['variants']==chapter['units'] for chapter in report['chapters'])
