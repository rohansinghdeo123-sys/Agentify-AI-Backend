"""Reviewed master documents bound to the existing curriculum, never a new catalog.

Sources use ContentChapter/Page/Chunk with a separate role/status/slug. Their
section manifest references existing unit and subtopic IDs; no topic hierarchy
is generated here. Reading is deterministic and makes no model/embedding call.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import defer

from models import ContentChapter, ContentChunk, ContentPage

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / "data/notes/uploads"
MASTER_NOTES = "MASTER_NOTES"
MAX_PDF_BYTES = 25 * 1024 * 1024
FORMATS = {"concept", "law", "numerical", "classification", "formula", "formula_based"}


def _digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _class_key(value):
    value = re.sub(r"[^a-z0-9]", "", str(value).lower())
    value = re.sub(r"^(?:class|grade)", "", value)
    return {"i":"1","ii":"2","iii":"3","iv":"4","v":"5","vi":"6","vii":"7","viii":"8","ix":"9","x":"10","xi":"11","xii":"12"}.get(value,value)


def _text(value, label, maximum=50000):
    if not isinstance(value,str) or not value.strip() or len(value)>maximum or "\ufffd" in value:
        raise ValueError(f"Invalid {label}")
    return value.strip()


def _scope_sections(curriculum, sections):
    if not isinstance(sections,list) or not 1 <= len(sections) <= 2000:
        raise ValueError("Provide one to 2000 explicitly mapped sections")
    units = {u['id']:u for u in curriculum['units']}
    prepared, identities = [], set()
    for order, item in enumerate(sections):
        if not isinstance(item,dict):
            raise ValueError("A master section must be an object")
        section_id = _text(item.get('id'), 'section ID',160)
        if section_id in identities:
            raise ValueError("Section IDs must be unique within a document")
        identities.add(section_id)
        unit = units.get(item.get('unit_id'))
        if not unit:
            raise ValueError("Master section references an unknown curriculum unit")
        allowed = {s['id'] for s in unit['ncert_subtopics']}
        subtopics = item.get('subtopic_ids')
        if not isinstance(subtopics,list) or not subtopics or any(not isinstance(s,str) for s in subtopics) or not set(subtopics) <= allowed:
            raise ValueError("Master section must reference only existing subtopics in its assigned unit")
        pages = item.get('source_pages') or []
        if not isinstance(pages,list) or any(type(p) is not int or p<1 for p in pages):
            raise ValueError("Source pages must be positive integer page numbers")
        teaching = item.get('teaching_format','concept')
        if teaching not in FORMATS:
            raise ValueError("Unsupported teaching format")
        for flag in ['reviewed','complete','review_required']:
            if type(item.get(flag,False)) is not bool:
                raise ValueError(f"{flag} must be boolean")
        prepared.append({
            'id':section_id,'title':_text(item.get('title'),'section title',240),
            'unit_id':unit['id'],'subtopic_ids':list(dict.fromkeys(subtopics)),
            'source_pages':sorted(set(pages)), 'source_order':order,
            'teaching_format':teaching,'reviewed':item.get('reviewed',False),
            'complete':item.get('complete',False),'review_required':item.get('review_required',False),
            'review_notes':str(item.get('review_notes') or '')[:2000],
            'text':item.get('text'),'start_anchor':item.get('start_anchor'),'end_anchor':item.get('end_anchor'),
        })
    return prepared


def _slice_source(text, start, end):
    def boundary(anchor):
        anchor = _text(anchor,'source anchor',400)
        return re.compile(r"\s+".join(re.escape(w) for w in anchor.split()))
    if start:
        match = boundary(start).search(text)
        if not match:
            raise ValueError("Start anchor does not exist in the explicitly selected PDF pages")
        text = text[match.start():]
    if end:
        match = boundary(end).search(text)
        if not match:
            raise ValueError("End anchor does not exist after the selected start boundary")
        text = text[:match.start()]
    return _text(text,'extracted section')


def register_master_document(db, *, curriculum, title, document_key, sections, actor_uid, pdf_bytes=None, embed=False):
    """Validate everything before replacing only this distinct master document.

    Flushes but never commits. The protected caller commits with its audit row.
    PDF mappings are explicit pages/anchors; supplied text must occur in that
    selected source. Structured sections are themselves the uploaded source.
    """
    title = _text(title,'document title',240)
    document_key = _text(document_key,'document key',160)
    prepared = _scope_sections(curriculum, sections)
    extracted, path = [], ''
    if pdf_bytes is not None:
        if not isinstance(pdf_bytes,bytes) or not pdf_bytes.startswith(b'%PDF-') or not 1 <= len(pdf_bytes) <= MAX_PDF_BYTES:
            raise ValueError("Provide a PDF no larger than 25 MB")
        from Logic.content_pipeline import extract_pdf_pages
        pdf_hash = _digest(pdf_bytes)
        UPLOAD_ROOT.mkdir(parents=True,exist_ok=True)
        destination = UPLOAD_ROOT / f'{pdf_hash}.pdf'
        if destination.exists() and _digest(destination.read_bytes()) != pdf_hash:
            raise ValueError("Stored master source digest mismatch")
        if not destination.exists():
            destination.write_bytes(pdf_bytes)
        try:
            extracted = extract_pdf_pages(destination)
        except RuntimeError:
            raise
        except Exception as exc:
            raise ValueError("PDF text could not be extracted; provide a readable, unlocked source document") from exc
        if not extracted or len(extracted)>1000:
            raise ValueError("The master PDF has no readable pages or exceeds the page limit")
        by_page = {p['page_number']:p['text'] for p in extracted}
        for item in prepared:
            pages = item['source_pages']
            if not pages or not set(pages) <= set(by_page):
                raise ValueError("Every PDF section requires explicit source pages inside the uploaded PDF")
            if any((p.get('text_sanity') or {}).get('suspected_encoding_corruption') or (p.get('text_sanity') or {}).get('suspected_formula_glyph_corruption')
                   for p in extracted if p['page_number'] in pages):
                raise ValueError("Mapped PDF source text has damaged encoding or scientific notation; review extraction first")
            source = _slice_source('\n\n'.join(by_page[p] for p in pages),item.pop('start_anchor'),item.pop('end_anchor'))
            supplied = item.get('text')
            if supplied:
                supplied = _text(supplied,'section text')
                if ' '.join(supplied.split()) not in ' '.join(source.split()):
                    raise ValueError("Section text was not found inside its mapped PDF source")
                item['text'] = supplied
            else:
                item['text'] = source
        path = destination.relative_to(ROOT).as_posix()
        source_hash = pdf_hash
    else:
        for item in prepared:
            if item.pop('start_anchor') or item.pop('end_anchor'):
                raise ValueError("Source anchors require an uploaded PDF")
            item['text'] = _text(item.get('text'),'section text')
            if not item['source_pages']:
                item['source_pages'] = [1]
        source_hash = _digest(prepared)
    if sum(len(s['text']) for s in prepared)>1_000_000:
        raise ValueError("Master sections exceed the one-million-character indexing limit")
    slug = 'master_notes_' + _digest([curriculum['curriculum_key'],document_key])[:32]
    manifest = []
    for item in prepared:
        item['chunk_id'] = slug + '_section_' + _digest(item['id'])[:16]
        item['text_sha256'] = _digest(item['text'].encode())
        manifest.append({k:v for k,v in item.items() if k!='text'})
    manifest_hash = _digest(manifest)
    document = db.query(ContentChapter).filter_by(slug=slug).one_or_none()
    if document and ((document.validation_report or {}).get('source_type') != MASTER_NOTES or document.status != 'indexed_master'):
        raise ValueError("A non-master record already uses this document identity")
    unchanged = bool(document and document.source_hash == source_hash and (document.validation_report or {}).get('section_manifest_sha256') == manifest_hash and document.chapter_name == title)
    if unchanged and not embed:
        return document
    if not document:
        document = ContentChapter(slug=slug,version='v1'); db.add(document); db.flush()
    elif not unchanged:
        current = re.fullmatch(r'v(\d+)',document.version or '')
        document.version = f'v{int(current[1])+1}' if current else 'v2'
    document.board = 'Master Notes'
    document.class_level = _class_key(curriculum['class_level'])
    document.subject = curriculum['subject']
    document.chapter_number = curriculum['chapter_number']
    document.chapter_name = title
    document.pdf_path = path
    document.source_hash = document.published_source_hash = source_hash
    document.status = 'indexed_master'
    document.approved_by = str(actor_uid)
    document.approved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    document.page_count = len(extracted) if extracted else max(p for s in prepared for p in s['source_pages'])
    document.extracted_page_count = len(extracted)
    document.chunk_count = len(prepared)
    document.validation_report = {
        'source_type':MASTER_NOTES,'document_key':document_key,
        'document_format':'pdf' if pdf_bytes is not None else 'structured_sections',
        'target_chapter_slug':curriculum['chapter_slug'],'target_curriculum_key':curriculum['curriculum_key'],
        'class_level':document.class_level,'subject':curriculum['subject'],'source_sha256':source_hash,
        'section_manifest_sha256':manifest_hash,'sections':manifest,
    }
    existing = {row.chunk_id:row for row in db.query(ContentChunk).filter_by(chapter_id=document.id).all()}
    keep = {s['chunk_id'] for s in prepared}
    for old_id, row in existing.items():
        if old_id not in keep:
            db.delete(row)
    vectors = {}
    if embed:
        from Logic import embeddings
        if embeddings.embeddings_enabled():
            pending = [s for s in prepared if s['chunk_id'] not in existing or existing[s['chunk_id']].text != s['text'] or not existing[s['chunk_id']].embedding]
            if pending:
                generated = embeddings.embed_texts([s['text'] for s in pending])
                if len(generated) != len(pending):
                    raise ValueError("Embedding provider returned an incomplete section batch")
                vectors = {s['chunk_id']:v for s,v in zip(pending,generated)}
    for item in prepared:
        row = existing.get(item['chunk_id'])
        if not row:
            row = ContentChunk(chapter_id=document.id,chunk_id=item['chunk_id']); db.add(row)
        if row.text != item['text']:
            row.embedding = None
        row.text = item['text']; row.section_title = item['title']
        row.page_start = min(item['source_pages']); row.page_end = max(item['source_pages'])
        row.lexical_terms = sorted(set(re.findall(r'[a-z0-9]+',item['text'].lower())))
        row.token_estimate = max(1,len(item['text'])//4)
        metadata = {k:v for k,v in item.items() if k not in {'text','chunk_id'}}
        metadata.update(source_type=MASTER_NOTES,source_sha256=source_hash)
        if item['chunk_id'] in vectors:
            from Logic import embeddings
            row.embedding = vectors[item['chunk_id']]
            metadata.update(embedding_model=embeddings.embedding_model(),embedding_endpoint_host=embeddings.embedding_endpoint_host(),embedding_dimensions=len(row.embedding))
        elif row.embedding:
            metadata.update({k:v for k,v in (row.metadata_json or {}).items() if k.startswith('embedding_')})
        row.metadata_json = metadata
    db.query(ContentPage).filter_by(chapter_id=document.id).delete(synchronize_session=False)
    for page in extracted:
        db.add(ContentPage(chapter_id=document.id,page_number=page['page_number'],text=page['text'],char_count=len(page['text']),
                           extraction_quality=page.get('extraction_quality',1),metadata_json={'source_type':MASTER_NOTES,'source_sha256':source_hash}))
    db.flush()
    return document


def install_legacy_master_metadata(db, document, artifact, curriculum):
    """Bind the already audited bundled master index without rewriting its rows."""
    owners = {s['id']:u['id'] for u in curriculum['units'] for s in u['ncert_subtopics']}
    entries, warnings = [], []
    for order, section in enumerate(artifact['sections']):
        units = {owners.get(s) for s in section['subtopic_ids']}
        if None in units or len(units)!=1:
            warnings.append({'section_id':section['id'],'title':section['title'],
                             'reason':'Master section has no unambiguous existing topic/subtopic binding'})
            continue
        entries.append({'id':section['id'],'chunk_id':section['id'],'unit_id':units.pop(),
                        'subtopic_ids':list(section['subtopic_ids']),'source_pages':list(section['source_pages']),
                        'source_order':order,'reviewed':True,'complete':True,'review_required':False,
                        'teaching_format':section['teaching_format'],'text_sha256':section['sha256']})
    report = {**(document.validation_report or {}),'source_type':MASTER_NOTES,'legacy_bridge':True,
              'target_chapter_slug':curriculum['chapter_slug'],'target_curriculum_key':curriculum['curriculum_key'],
              'class_level':_class_key(curriculum['class_level']),'subject':curriculum['subject'],
              'source_sha256':artifact['pdf_sha256'],'document_format':'pdf',
              'section_manifest_sha256':_digest(entries),'sections':entries,'mapping_warnings':warnings}
    if document.source_hash != artifact['pdf_sha256']:
        raise ValueError("Legacy master document source digest mismatch")
    document.validation_report = report
    db.flush()
    return document


def find_master_sections(db, curriculum, unit, requested_subtopic_ids=None):
    allowed = {s['id'] for s in unit['ncert_subtopics']}
    requested = set(allowed if requested_subtopic_ids is None else requested_subtopic_ids)
    if not requested or not requested <= allowed:
        raise ValueError("Requested master sections are outside the assigned topic")
    documents = db.query(ContentChapter).filter(
        ContentChapter.status=='indexed_master',
        func.lower(ContentChapter.subject)==str(curriculum['subject']).lower(),
        ContentChapter.validation_report['source_type'].as_string()==MASTER_NOTES,
        ContentChapter.validation_report['target_curriculum_key'].as_string()==curriculum['curriculum_key'],
    ).order_by(ContentChapter.id).all()
    sections, conflicts, fingerprint, document_coverage = [], [], [], []
    for document in documents:
        report = document.validation_report or {}
        if _class_key(document.class_level)!=_class_key(curriculum['class_level']) or report.get('target_chapter_slug')!=curriculum['chapter_slug']:
            continue
        relevant = [s for s in report.get('sections',[]) if s.get('unit_id')==unit['id'] and set(s.get('subtopic_ids',[])).intersection(requested)]
        if not relevant:
            continue
        fingerprint.append([document.id,document.source_hash,document.version,report.get('section_manifest_sha256')])
        if report.get('source_sha256') != document.source_hash or _digest(report.get('sections',[])) != report.get('section_manifest_sha256'):
            conflicts.append({'reason':'Master document index failed source verification','subtopic_ids':sorted(requested),'section_ids':[],'source_document_id':document.id})
            continue
        ids = [s['chunk_id'] for s in relevant if set(s['subtopic_ids']) <= requested]
        rows = {r.chunk_id:r for r in db.query(ContentChunk).options(defer(ContentChunk.embedding)).filter(ContentChunk.chapter_id==document.id,ContentChunk.chunk_id.in_(ids)).all()} if ids else {}
        verified = set()
        for binding in relevant:
            row = rows.get(binding['chunk_id'])
            scope = set(binding['subtopic_ids'])
            meta = row.metadata_json or {} if row else {}
            reason = None
            if not scope <= requested:
                reason = 'Master section spans an unselected subtopic; an exact binding is required'
            elif not row or not row.text or _digest(row.text.encode()) != binding.get('text_sha256') or (meta.get('text_sha256') or meta.get('sha256')) != binding.get('text_sha256'):
                reason = 'Master section is missing or failed text verification'
            elif not report.get('legacy_bridge') and (meta.get('source_type')!=MASTER_NOTES or meta.get('source_sha256')!=document.source_hash or meta.get('unit_id')!=unit['id'] or set(meta.get('subtopic_ids',[]))!=scope):
                reason = 'Master section source/scope metadata failed verification'
            elif not binding.get('reviewed') or binding.get('review_required') or not meta.get('reviewed',report.get('legacy_bridge',False)) or meta.get('review_required'):
                reason = binding.get('review_notes') or 'Master section requires review before teaching'
            if reason:
                conflicts.append({'reason':reason,'subtopic_ids':sorted(scope & requested),'section_ids':[binding['chunk_id']],'source_document_id':document.id})
                fingerprint.append([binding['chunk_id'],reason])
                continue
            fingerprint.append([row.chunk_id,binding['text_sha256'],bool(binding.get('complete')),bool(meta.get('complete',report.get('legacy_bridge',False)))])
            verified.add(row.chunk_id)
            source_document = {'id':document.id,'slug':document.slug,'title':document.chapter_name,'sha256':document.source_hash,'format':report.get('document_format','pdf')}
            sections.append({'id':row.chunk_id,'title':row.section_title,'text':row.text,'source_type':MASTER_NOTES,
                             'source_document':source_document,'source_pages':list(binding['source_pages']),
                             'subtopic_ids':list(binding['subtopic_ids']),'unit_id':unit['id'],
                             'teaching_format':binding.get('teaching_format','concept'),'reviewed':True,
                             'complete':bool(binding.get('complete') and meta.get('complete',report.get('legacy_bridge',False))),
                             'text_sha256':binding['text_sha256'],'source_order':binding.get('source_order',0),
                             'provenance':{'source_type':MASTER_NOTES,'document_id':document.id,'section_id':row.chunk_id,'source_sha256':document.source_hash}})
        complete = {s for s in requested if (required := [b for b in relevant if s in b['subtopic_ids']]) and all(b['chunk_id'] in verified and b.get('complete') and rows[b['chunk_id']].metadata_json.get('complete',report.get('legacy_bridge',False)) for b in required)}
        document_coverage.append(complete)
    # Named scientific formula conflicts are withheld for review. Similar but
    # nonidentical prose is not, by itself, evidence of a scientific conflict.
    blocked = set()
    for index, item in enumerate(sections):
        for previous in sections[:index]:
            common = set(item['subtopic_ids']) & set(previous['subtopic_ids'])
            if common and _formula_conflict(previous['text'],item['text']):
                blocked.update(common)
                conflicts.append({'reason':'Conflicting master-note formula statements require review','subtopic_ids':sorted(common),'section_ids':[previous['id'],item['id']]})
    sections = [s for s in sections if not blocked.intersection(s['subtopic_ids'])]
    unique = {}
    for section in sections:
        key = (_digest(' '.join(section['text'].split()).encode()),tuple(sorted(section['subtopic_ids'])))
        unique.setdefault(key,section)
    covered = set().union(*document_coverage) if document_coverage else set()
    covered.difference_update(blocked)
    return {'sections':list(unique.values()),'covered_subtopic_ids':sorted(covered),
            'conflicts':conflicts,'source_fingerprint':_digest(fingerprint)}


def _formula_conflict(left, right):
    pattern = r"(?i)\b(density|weight|molarity|molality|mass %|speed|force|pressure|acceleration|resistance)\s*=\s*([^.;\n]+)"
    def normalized(name,value):
        aliases={'density':{'mass':'m','volume':'v'},'weight':{'gravitational acceleration':'g','mass':'m'}}
        result=value.lower()
        for term,symbol in aliases.get(name,{}).items():
            result=re.sub(r'\b'+re.escape(term)+r'\b',symbol,result)
        result=re.sub(r'\s+x\s+','*',result).replace('×','*').replace('−','-')
        result=re.sub(r'\(([a-z_]+)\)',r'\1',result)
        return re.sub(r'\s+','',result)
    def facts(text):
        return {name.lower():normalized(name.lower(),value) for name,value in re.findall(pattern,text)
                if not re.search(r'\d',value)}
    a,b = facts(left),facts(right)
    return any(name in a and a[name] != value for name,value in b.items())


def serialize_master_document(document):
    report = document.validation_report or {}
    return {'id':document.id,'title':document.chapter_name,'class_level':document.class_level,'subject':document.subject,
            'target_chapter_slug':report.get('target_chapter_slug'),'target_curriculum_key':report.get('target_curriculum_key'),
            'source_type':MASTER_NOTES,'source_sha256':document.source_hash,'version':document.version,
            'section_count':document.chunk_count,'sections':report.get('sections',[]),
            'mapping_warnings':report.get('mapping_warnings',[])}
