"""Real indexed-source persistence and protected admin transport, without models."""
import hashlib
import io
from copy import deepcopy
from datetime import date
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.security import verify_firebase_user
from database import Base, get_db
from models import AdminAuditLog, ContentChapter, ContentChunk, ContentConcept, StudyRouteState
from services.notes_documents_service import (
    MASTER_NOTES, find_master_sections, install_legacy_master_metadata, register_master_document,
)


def test_equivalent_named_formulas_do_not_create_false_source_conflicts():
    from services.notes_documents_service import _formula_conflict
    assert not _formula_conflict('Density = mass / volume.','Density = m / V.')
    assert not _formula_conflict('Weight = mass x gravitational acceleration.','Weight = m × g.')
    assert _formula_conflict('Density = mass / volume.','Density = V / m.')


@pytest.fixture
def db():
    engine = create_engine('sqlite://',poolclass=StaticPool,connect_args={'check_same_thread':False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def curriculum():
    return {'curriculum_key':'published_math10_quadratics','chapter_slug':'quadratic_equations',
            'chapter_title':'Quadratic Equations','chapter_number':4,'class_level':'Class 10','subject':'Mathematics',
            'units':[{'id':'quadratic_unit','ncert_subtopics':[{'id':'quadratic_definition'},{'id':'quadratic_formula'}]},
                     {'id':'roots_unit','ncert_subtopics':[{'id':'root_relationships'}]}]}


def section(**changes):
    return {'id':'definition','title':'A quadratic equation','unit_id':'quadratic_unit',
            'subtopic_ids':['quadratic_definition'],'source_pages':[1],
            'text':'A quadratic equation has a squared unknown. Its leading coefficient must be nonzero.',
            'reviewed':True,'complete':True,'teaching_format':'concept',**changes}


def register(db,c,sections=None,**kwargs):
    return register_master_document(db,curriculum=c,title='Reviewed teacher notes',document_key='teacher-v1',
        sections=sections or [section()],actor_uid='admin',**kwargs)


def test_generic_master_is_persisted_searchable_and_never_queries_embeddings_on_read(db,curriculum):
    doc = register(db,curriculum)
    db.commit()
    assert doc.status == 'indexed_master' and doc.board == 'Master Notes'
    queries=[]
    def capture(conn,cursor,statement,*args):
        queries.append(statement)
    event.listen(db.bind,'before_cursor_execute',capture)
    with patch('Logic.embeddings.embed_query',side_effect=AssertionError('Embedding network call on notes open')):
        result = find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])
    event.remove(db.bind,'before_cursor_execute',capture)
    assert result['covered_subtopic_ids'] == ['quadratic_definition']
    assert result['sections'][0]['text'] == section()['text']
    assert result['sections'][0]['source_type'] == MASTER_NOTES
    assert result['sections'][0]['source_document']['sha256'] == doc.source_hash
    assert not any('content_chunks.embedding' in query for query in queries)
    assert db.query(ContentChunk).one().lexical_terms


def test_idempotent_updates_touch_only_the_selected_master_document(db,curriculum):
    ncert=ContentChapter(slug='original-ncert',status='published',class_level='10',subject='Mathematics',chapter_name='NCERT original')
    db.add(ncert);db.flush()
    db.add(ContentConcept(chapter_id=ncert.id,concept_id='original-concept',core_explanation='Original NCERT teaching'))
    db.add(ContentChunk(chapter_id=ncert.id,chunk_id='original-chunk',text='Original NCERT source'))
    state=StudyRouteState(user_id='student',route_date=date.today(),class_level='Class 10',subject='Mathematics',
                          runtime_json={'bookmark':4},current_unit_id='quadratic_unit')
    db.add(state);db.commit()
    first=register(db,curriculum);db.commit()
    first_id,first_hash=first.id,first.source_hash
    second=register(db,curriculum);db.commit()
    assert second.id == first_id and second.version == 'v1'
    assert db.query(ContentChapter).count() == 2 and db.query(ContentChunk).count() == 2
    updated=register(db,curriculum,[section(text='A quadratic equation has degree two and a nonzero leading coefficient.')]);db.commit()
    assert updated.id==first_id and updated.version=='v2' and updated.source_hash!=first_hash
    assert db.query(ContentConcept).one().core_explanation == 'Original NCERT teaching'
    assert db.query(ContentChunk).filter_by(chunk_id='original-chunk').one().text == 'Original NCERT source'
    assert state.runtime_json == {'bookmark':4}


@pytest.mark.parametrize('change',[{'unit_id':'another-unit'},{'subtopic_ids':['root_relationships']},
    {'subtopic_ids':[]},{'source_pages':[True]},{'source_pages':[-1]},{'reviewed':'yes'}])
def test_invalid_or_cross_unit_mappings_fail_before_source_rows_are_written(db,curriculum,change):
    with pytest.raises(ValueError):
        register(db,curriculum,[section(**change)])
    assert db.query(ContentChapter).count() == 0


def test_partial_unreviewed_missing_and_unselected_sections_do_not_claim_coverage(db,curriculum):
    register(db,curriculum,[section(complete=False),section(id='formula',title='Formula',subtopic_ids=['quadratic_formula'],reviewed=False,text='The quadratic formula follows by completing the square.')])
    db.commit()
    result=find_master_sections(db,curriculum,curriculum['units'][0])
    assert not result['covered_subtopic_ids']
    assert [s['title'] for s in result['sections']] == ['A quadratic equation']
    assert result['conflicts']
    document=register_master_document(db,curriculum=curriculum,title='Combined source',document_key='combined',
        sections=[section(subtopic_ids=['quadratic_definition','quadratic_formula'])],actor_uid='admin')
    db.commit()
    scoped=find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_formula'])
    assert not scoped['sections'] and not scoped['covered_subtopic_ids']
    assert any('unselected' in f['reason'] for f in scoped['conflicts'])
    # A row missing from an otherwise reviewed document cannot count as covered.
    rows=db.query(ContentChunk).filter_by(chapter_id=document.id).all()
    for row in rows:db.delete(row)
    db.commit()
    assert not find_master_sections(db,curriculum,curriculum['units'][0])['covered_subtopic_ids']


def test_source_and_text_tampering_invalidate_coverage_and_cache_fingerprint(db,curriculum):
    doc=register(db,curriculum);db.commit()
    original=find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])
    row=db.query(ContentChunk).one();row.text='Tampered wording unrelated to the registered source.';db.commit()
    changed=find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])
    assert not changed['sections'] and not changed['covered_subtopic_ids'] and changed['conflicts']
    assert changed['source_fingerprint'] != original['source_fingerprint']
    doc.source_hash='0'*64;db.commit()
    assert not find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])['sections']


def test_master_search_never_crosses_class_subject_or_target_chapter(db,curriculum):
    register(db,curriculum);db.commit()
    for changed in [{'class_level':'Class 11'},{'subject':'Physics'},{'chapter_slug':'another_chapter'},{'curriculum_key':'other-map'}]:
        requested={**curriculum,**changed}
        result=find_master_sections(db,requested,requested['units'][0],['quadratic_definition'])
        assert not result['sections'] and not result['covered_subtopic_ids']


def test_conflicting_master_formulas_are_flagged_and_withheld(db,curriculum):
    register(db,curriculum,[section(text='Density = mass / volume.')])
    register_master_document(db,curriculum=curriculum,title='A different source',document_key='other',
        sections=[section(text='Density = volume / mass.')],actor_uid='admin')
    db.commit()
    result=find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])
    assert not result['sections'] and not result['covered_subtopic_ids']
    assert any('Conflicting' in conflict['reason'] for conflict in result['conflicts'])


def test_identical_master_content_is_not_repeated(db,curriculum):
    register(db,curriculum)
    register_master_document(db,curriculum=curriculum,title='Another copy',document_key='copy',
        sections=[section(text=section()['text'].replace(' ','  '))],actor_uid='admin')
    db.commit()
    result=find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])
    assert len(result['sections'])==1 and result['covered_subtopic_ids']==['quadratic_definition']


def test_embedding_is_optional_index_work_and_is_retained_on_unchanged_reindex(db,curriculum):
    with patch('Logic.embeddings.embeddings_enabled',return_value=True),patch('Logic.embeddings.embed_texts',return_value=[[0.0,1.0,0.0]]) as embedding:
        doc=register(db,curriculum,embed=True);db.commit()
        assert embedding.call_count == 1
        assert db.query(ContentChunk).one().embedding == [0.0,1.0,0.0]
        register(db,curriculum,embed=True);db.commit()
        assert embedding.call_count == 1 and doc.version=='v1'


def _pdf(*pages):
    """Small real text PDF fixture, using only the standard library."""
    objects=[b'',b'']
    page_ids=[]
    for text in pages:
        page_id=len(objects)+1;page_ids.append(page_id)
        content_id=page_id+1
        objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id+1} 0 R >>'.encode())
        stream=('BT /F1 12 Tf 40 750 Td ('+text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')+') Tj ET').encode()
        objects.append(b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream')
    # Insert the font object and adjust page/content references to its insertion.
    objects.insert(2,b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    ids=[i+1 for i in page_ids]
    objects[0]=b'<< /Type /Catalog /Pages 2 0 R >>'
    objects[1]=f'<< /Type /Pages /Kids [{" ".join(f"{i} 0 R" for i in ids)}] /Count {len(ids)} >>'.encode()
    data=b'%PDF-1.4\n';offsets=[0]
    for i,obj in enumerate(objects,1):
        offsets.append(len(data));data+=f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
    xref=len(data);data+=f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode()
    data+=b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets[1:])
    return data+f'trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode()


def test_real_pdf_extraction_retains_only_explicit_pages_and_rejects_false_attribution(db,curriculum,tmp_path,monkeypatch):
    from services import notes_documents_service as service
    monkeypatch.setattr(service,'ROOT',tmp_path);monkeypatch.setattr(service,'UPLOAD_ROOT',tmp_path/'uploads')
    source='A quadratic equation has a squared unknown.'
    pdf=_pdf(source,'Trigonometry is a different topic.')
    doc=register(db,curriculum,[section(text=None)],pdf_bytes=pdf);db.commit()
    result=find_master_sections(db,curriculum,curriculum['units'][0],['quadratic_definition'])
    assert source in result['sections'][0]['text'] and 'Trigonometry' not in result['sections'][0]['text']
    assert doc.source_hash==hashlib.sha256(pdf).hexdigest()
    with pytest.raises(ValueError,match='not found'):
        register_master_document(db,curriculum=curriculum,title='Wrong attribution',document_key='wrong',
            sections=[section(text='Trigonometry is a different topic.')],actor_uid='admin',pdf_bytes=pdf)
    with pytest.raises(ValueError,match='could not be extracted'):
        register(db,curriculum,[section(text=None)],pdf_bytes=b'%PDF-this-is-not-a-readable-pdf')


def test_legacy_master_bridge_preserves_original_index_and_vectors(db):
    from services.master_notes_service import artifact,install_master_notes
    from Logic.planning.curriculum_registry import load_planning_curriculum
    document=install_master_notes(db);source=artifact();curriculum=load_planning_curriculum()
    before={r.chunk_id:(r.text,deepcopy(r.metadata_json),deepcopy(r.embedding)) for r in db.query(ContentChunk).filter_by(chapter_id=document.id)}
    install_legacy_master_metadata(db,document,source,curriculum);db.commit()
    after={r.chunk_id:(r.text,r.metadata_json,r.embedding) for r in db.query(ContentChunk).filter_by(chapter_id=document.id)}
    assert before==after
    unit=curriculum['units'][3]
    result=find_master_sections(db,curriculum,unit)
    assert result['sections'] and set(result['covered_subtopic_ids'])=={s['id'] for s in unit['ncert_subtopics']}


def test_admin_transport_checks_authority_scope_audit_and_uses_existing_catalog(db):
    from routers.notes_admin import router
    from services.notes_scope_service import resolve_notes_curriculum
    chapter=ContentChapter(slug='ncert_class_10_mathematics_chapter_4_quadratic_equations',board='NCERT',status='published',
        class_level='10',subject='Mathematics',chapter_number=4,chapter_name='Quadratic Equations',page_count=2,extracted_page_count=2)
    db.add(chapter);db.flush()
    db.add(ContentConcept(chapter_id=chapter.id,concept_id='quadratic_source',title='Quadratic equations',source_pages=[1],key_points=['A quadratic has degree two.']))
    db.commit()
    curriculum=resolve_notes_curriculum(db,chapter.slug,subject='Mathematics',class_level='Class 10')
    unit=curriculum['units'][0];subtopic=unit['ncert_subtopics'][0]['id']
    payload={'title':'Teacher master','document_key':'teacher','class_level':'Class 10','subject':'Mathematics','chapter_ref':chapter.slug,
             'sections':[section(unit_id=unit['id'],subtopic_ids=[subtopic])]}
    app=FastAPI();app.include_router(router)
    app.dependency_overrides[get_db]=lambda:db
    token={'uid':'student','email':'student-only@example.test'}
    app.dependency_overrides[verify_firebase_user]=lambda:token
    with TestClient(app) as client:
        assert client.post('/admin/notes/master-documents',json=payload).status_code==404
        assert db.query(AdminAuditLog).count()==0
        token.update(uid='admin',admin=True)
        response=client.post('/admin/notes/master-documents',json=payload)
        assert response.status_code==200,response.text
        body=response.json();document_id=body['document']['id']
        assert body['document']['source_type']==MASTER_NOTES and 'pdf_path' not in str(body)
        assert db.query(AdminAuditLog).count()==1
        listed=client.get('/admin/notes/master-documents',params={k:payload[k] for k in ['class_level','subject','chapter_ref']})
        assert listed.status_code==200 and len(listed.json()['documents'])==1
        assert client.get(f'/admin/notes/master-documents/{document_id}').json()['document']['sections'][0]['text']==section()['text']
        invalid={**payload,'sections':[section(unit_id=unit['id'],subtopic_ids=['another-subtopic'])]}
        assert client.post('/admin/notes/master-documents',json=invalid).status_code==400
        assert db.query(ContentChapter).filter_by(status='published').count()==1
        assert db.query(ContentConcept).count()==1


def test_admin_pdf_upload_uses_explicit_existing_bindings_and_does_not_expose_storage_paths(db,tmp_path,monkeypatch):
    from routers.notes_admin import router
    from services import notes_documents_service as service
    from Logic.planning.curriculum_registry import load_planning_curriculum
    import json
    monkeypatch.setattr(service,'ROOT',tmp_path);monkeypatch.setattr(service,'UPLOAD_ROOT',tmp_path/'uploads')
    c=load_planning_curriculum();u=c['units'][0];s=u['ncert_subtopics'][0]['id']
    app=FastAPI();app.include_router(router)
    app.dependency_overrides[get_db]=lambda:db
    app.dependency_overrides[verify_firebase_user]=lambda:{'uid':'admin','admin':True}
    text='Chemistry studies matter and its transformations.'
    form={'title':'PDF master','document_key':'pdf-master','class_level':c['class_level'],'subject':c['subject'],
          'chapter_ref':c['chapter_slug'],'sections_json':json.dumps([section(unit_id=u['id'],subtopic_ids=[s],text=None)])}
    with TestClient(app) as client:
        response=client.post('/admin/notes/master-documents/upload',data=form,files={'file':('../../outside.pdf',_pdf(text),'application/pdf')})
        assert response.status_code==200,response.text
        data=response.json()['document']
        assert data['source_type']==MASTER_NOTES and 'pdf_path' not in str(data) and 'outside.pdf' not in str(data)
        assert db.query(AdminAuditLog).count()==1
        row=db.query(ContentChapter).one()
        assert '..' not in row.pdf_path and row.pdf_path.startswith('uploads/')
        broken=client.post('/admin/notes/master-documents/upload',data={**form,'sections_json':'not-json'},files={'file':('bad.pdf',_pdf(text),'application/pdf')})
        assert broken.status_code==400
