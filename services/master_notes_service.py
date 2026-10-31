"""Section-level Amit-first retrieval shared by Notes and the grounded tutor."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from models import ContentChapter, ContentChunk
from sqlalchemy.orm import defer
from Logic.planning.curriculum_registry import resolve_planning_curriculum

ROOT = Path(__file__).resolve().parents[1]
SLUG = "amit_master_class11_chemistry_ch1"

@lru_cache(maxsize=1)
def artifact():
    return json.loads((ROOT / "data/notes/amit_chapter1_sections.json").read_text(encoding="utf-8"))

def install_master_notes(db):
    """Idempotently install verified sections; never overwrite NCERT/learner rows."""
    data = artifact()
    if hashlib.sha256((ROOT / "data/notes/amit_chapter1.pdf").read_bytes()).hexdigest() != data["pdf_sha256"]:
        raise ValueError("Master PDF digest mismatch")
    chapter = db.query(ContentChapter).filter_by(slug=SLUG).first()
    if not chapter:
        chapter = ContentChapter(slug=SLUG, board="Amit Master Notes", class_level="11", subject="Chemistry", chapter_number=1,
            chapter_name="Some Basic Concepts of Chemistry", status="indexed_master", approved_by="Amit",
            pdf_path="data/notes/amit_chapter1.pdf", source_hash=data["pdf_sha256"], published_source_hash=data["pdf_sha256"], page_count=7)
        db.add(chapter); db.flush()
    chapter.source_hash=data["pdf_sha256"]; chapter.published_source_hash=data["pdf_sha256"]
    existing = {c.chunk_id:c for c in db.query(ContentChunk).filter_by(chapter_id=chapter.id).all()}
    for section in data["sections"]:
        if hashlib.sha256(section["text"].encode()).hexdigest() != section["sha256"]:
            raise ValueError("Master section digest mismatch")
        row = existing.get(section["id"])
        if row and row.text == section["text"] and row.metadata_json == {k:v for k,v in section.items() if k != "embedding"}:
            continue
        if not row:
            row = ContentChunk(chapter_id=chapter.id, chunk_id=section["id"]); db.add(row)
        row.text=section["text"]; row.section_title=section["title"]
        row.page_start=min(section["source_pages"]); row.page_end=max(section["source_pages"])
        row.lexical_terms=sorted(set(re.findall(r"[a-z]+", row.text.lower())))
        row.token_estimate=len(row.text)//4
        row.embedding=section.get("embedding")
        row.metadata_json={k:v for k,v in section.items() if k != "embedding"}
    chapter.chunk_count=len(data["sections"])
    db.commit()
    return chapter

def resolve_unit(chapter, topic, preferred_unit=None):
    c=resolve_planning_curriculum(chapter_ref=chapter, subject="Chemistry", class_level="Class 11")
    if not c or c["chapter_number"] != 1: return None
    units=c["units"]
    unit=next((u for u in units if topic in {u["id"],u["primary_topic_id"],*u.get("legacy_topic_ids",[])}),None)
    if unit:return c,unit
    # Previously published concept URLs used merged, hashed concept IDs.
    # Resolve them via the audited excerpt map, never the old paragraph reader.
    from services.topic_notes_service import chapter_one_mapping
    candidates=[]
    for mapped in chapter_one_mapping()["units"]:
        count=sum(p["source_concept_id"]==topic for p in mapped["pages"])
        if count:candidates.append((count,mapped["unit_id"]))
    seed_groups={"atoms_and_molecules":0,"nature_and_classification_of_matter":0,
        "properties_of_matter_and_measurement":1,"si_units_and_prefixes":1,"mass_weight_volume_density":1,
        "uncertainty_and_scientific_notation":1,"significant_figures":1,"dimensional_analysis":1,
        "laws_of_chemical_combinations":2,"daltons_atomic_theory":2,"atomic_molecular_formula_masses":3,
        "mole_concept_and_molar_mass":4,"percentage_composition":4,"empirical_and_molecular_formulae":4,
        "stoichiometry_and_limiting_reagent":5,"concentration_of_solutions":5}
    if topic in seed_groups:return c,units[seed_groups[topic]]
    if not topic:return c,next((u for u in units if u["id"]==preferred_unit),units[0])
    if candidates:
        selected=next((uid for _,uid in candidates if uid==preferred_unit),None)
        if not selected:selected=max(candidates,key=lambda item:item[0])[1]
        return c,next(u for u in units if u["id"]==selected)
    return None

def ncert_gap(db, curriculum, unit, subtopic):
    """Retrieve exact, known source boundaries; never fetch an entire unit for a gap."""
    from Logic.tools.knowledge_search import _cached_planning_pdf_pages, _planning_source_segment
    # All other authored subtopics are covered by Amit's PDF. Unknown/missing
    # boundaries fail closed rather than pulling a neighbouring NCERT section.
    segments = {
        "intro.development_of_chemistry": [
            {"page":1,"start_anchor":"The branch of science that studies the preparation","end_anchor":"After studying this unit"},
            {"page":2}, {"page":3,"end_anchor":"From the above discussion"}],
        "1.1": [s for s in unit["source_segments"] if s.get("start_anchor", "").startswith("1.1 ")],
    }.get(subtopic, [])
    if not segments:
        # Exact numbered subtopic boundaries, including headings spanning lines.
        source=curriculum["source"]
        source_pages=_cached_planning_pdf_pages(str(ROOT / source["path"]),source.get("sha256", ""))
        joined="\n".join(text for _,text in source_pages)
        marker=re.search(r"(?im)^\s*"+re.escape(subtopic)+r"\s+[A-Za-z]",joined)
        if not marker:return []
        rest=joined[marker.end():]
        end=re.search(r"(?m)^\s*1\.\d+(?:\.\d+)?\s+[A-Za-z]",rest)
        text=joined[marker.start():marker.end()+(end.start() if end else len(rest))]
        # Keep only registered unit pages; fail closed for ambiguous/box headings.
        allowed_pages={int(s["page"]) for s in unit["source_segments"]}
        if not any(text[:40] in body for page,body in source_pages if page in allowed_pages):return []
        from Logic.tools.knowledge_search import _planning_pdf_chunks
        return [{"id":f"ncert-{subtopic}-{i}","title":subtopic,"text":chunk,"source":"Supplementary NCERT",
            "source_pages":sorted(allowed_pages),"subtopic_ids":[subtopic],"teaching_format":"concept"}
            for i,chunk in enumerate(_planning_pdf_chunks(text))]
    source=curriculum["source"]
    pages=dict(_cached_planning_pdf_pages(str(ROOT / source["path"]),source.get("sha256", "")))
    output=[]
    for i,segment in enumerate(segments):
        text,_=_planning_source_segment(pages[segment["page"]],segment)
        # The source PDF typesets the carbonate subscript on a separate text
        # baseline. Restore that verified notation, not a guessed number.
        text=re.sub(r"CaCO\s*\n\s*3", "CaCO₃", text)
        # Reuse the existing paragraph chunker; each chunk has a precise boundary.
        from Logic.tools.knowledge_search import _planning_pdf_chunks
        for j,chunk in enumerate(_planning_pdf_chunks(text)):
            output.append({"id":f"ncert-{subtopic}-{i}-{j}","title":next(s["title"] for s in unit["ncert_subtopics"] if s["id"]==subtopic),
                "text":chunk,"source":"Supplementary NCERT", "source_pages":[segment["page"]],"subtopic_ids":[subtopic],"teaching_format":"concept"})
    return output

def retrieve_notes(db, chapter, topic, requested_subtopics=None, question=""):
    resolved=resolve_unit(chapter,topic)
    if not resolved: return None
    curriculum,unit=resolved
    allowed={s["id"] for s in unit["ncert_subtopics"]}
    requested=set(requested_subtopics or allowed)
    if not requested <= allowed: raise ValueError("Subtopics outside selected Planning unit")
    # Indexed DB sections are the primary source, not a prompt-level preference.
    chapter_row=db.query(ContentChapter).filter_by(slug=SLUG,status="indexed_master").first()
    query=db.query(ContentChunk).filter_by(chapter_id=chapter_row.id).order_by(ContentChunk.id) if chapter_row else None
    rows=(query.options(defer(ContentChunk.embedding)) if not question else query).all() if query is not None else []
    verified={s["id"]:s["sha256"] for s in artifact()["sections"]}
    sections=[];covered=set()
    from Logic import embeddings
    query_vector=embeddings.embed_query(question) if question else None
    terms=set(re.findall(r"[a-z]{3,}",question.lower()))
    ranks={}
    for row in rows:
        meta=dict(row.metadata_json or {}); ids=set(meta.get("subtopic_ids",[]))
        if not ids.intersection(requested): continue
        if verified.get(row.chunk_id) != meta.get("sha256"): continue
        if not row.text or not meta.get("sha256") or hashlib.sha256(row.text.encode()).hexdigest()!=meta["sha256"]: continue
        lexical=len(terms.intersection(row.lexical_terms or []))
        semantic=embeddings.similarity(query_vector,row.embedding) if query_vector and embeddings.compatible_with_stored(query_vector,row.embedding,
            stored_model=meta.get("embedding_model", ""),stored_endpoint_host=meta.get("embedding_endpoint_host", "")) else 0
        ranks[meta["id"]]=lexical+semantic
        sections.append({**meta,"text":row.text,"source":"Amit Master Notes"});covered.update(ids)
    expected={topic:{s["id"] for s in artifact()["sections"] if topic in s["subtopic_ids"]} for topic in requested}
    found={s["id"] for s in sections}
    covered={topic for topic in covered if expected.get(topic) and expected[topic] <= found}
    if question:sections.sort(key=lambda s:-ranks.get(s["id"],0))
    missing=requested-covered; conflicts=[]
    for gap in sorted(missing):
        for section in ncert_gap(db,curriculum,unit,gap):
            if formula_conflicts(sections, section) or set(section["subtopic_ids"]).intersection(covered):
                conflicts.append({"subtopic":gap,"reason":"Potential source conflict or overlapping coverage; supplement withheld for review"});continue
            # Keep only new source sentences when NCERT repeats already-known
            # wording inside a genuinely incomplete subtopic.
            known_sentences=[part for item in sections if item["source"]=="Amit Master Notes"
                for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])",item["text"])]
            def duplicate(sentence):
                words=set(re.findall(r"[a-z0-9]+",sentence.lower()))
                return any(words and len(words & set(re.findall(r"[a-z0-9]+",known.lower()))) / max(len(words | set(re.findall(r"[a-z0-9]+",known.lower()))),1) >= 0.85 for known in known_sentences)
            parts=re.split(r"(?<=[.!?])\s+(?=[A-Z])",section["text"])
            section={**section,"text":" ".join(part for part in parts if not duplicate(part))}
            if section["text"] and section["text"] not in {s["text"] for s in sections}:sections.append(section)
    return {"curriculum":curriculum,"unit":unit,"sections":sections,"missing_subtopics":sorted(missing),"conflicts":conflicts}

def formula_conflicts(master_sections, supplement):
    """Conservatively flag differing statements of the same named formula.

    Worked examples and rounding differences are not treated as definitions.
    Disagreements are withheld for human review, never silently corrected.
    """
    pattern=r"(?i)\b(density|weight|molarity|molality|mass %)\s*=\s*([^.;\n]+)"
    def facts(text):
        return {name.lower():re.sub(r"\s+"," ",value.strip().lower()) for name,value in re.findall(pattern,text)}
    known={}
    for section in master_sections:
        if section["source"]=="Amit Master Notes":known.update(facts(section["text"]))
    return any(name in known and known[name]!=value for name,value in facts(supplement["text"]).items())

def teaching_page(section):
    """Choose a teaching structure; retain every sourced sentence and notation."""
    kind=section.get("teaching_format","concept")
    children=section.get("children",[section])
    points=[]
    for child in children:
        sentences=re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])",child["text"])
        for i,sentence in enumerate(sentences):
            is_example="example" in sentence.lower() or child["title"].lower().startswith("example")
            label=("Worked example" if kind in {"numerical","formula"} else "Example / application") if is_example else (
                "Statement" if kind=="law" and i==0 else
                "Categories and characteristics" if kind=="classification" and i>0 else
                "Formula / application" if kind in {"numerical","formula"} and any(x in sentence for x in ["=", " / "]) else
                "Terms and units" if kind in {"numerical","formula"} else
                "Definition" if i==0 else "Simple explanation")
            if section.get("source")=="Supplementary NCERT":label="Supplementary NCERT"
            points.append({"label":label,"text":sentence})
    connections={
        "Atom":"A copper wire is visible because it contains a huge number of copper atoms; the example connects the tiny atom to an everyday object.",
        "Matter":"A book, water, air and a stone are examples of matter because each has mass and occupies space, even though air has no fixed shape.",
        "Molecule":"In H₂O, the subscript 2 counts hydrogen atoms, while no subscript after O means one oxygen atom. This is why the formula represents two hydrogen atoms bonded with one oxygen atom.",
        "Pure substances and mixtures":"Salt solution is uniform throughout, which is why it is homogeneous. Oil and water are not uniform, which is why that mixture is heterogeneous.",
    }
    if kind=="concept":
        for child in children:
            if child["title"] in connections:points.append({"label":"Why this example works", "text":connections[child["title"]]})
    if points:
        points.append({"label":"Practice" if kind in {"numerical","formula"} else "Key takeaway",
            "text":"Rework the source example above. Explain each conversion and include the units at every step." if kind in {"numerical","formula"} else children[0]["text"].split(". ")[0]+"."})
    return {k:v for k,v in {**section,"text":"\n\n".join(p["label"]+": "+p["text"] for p in points),"blocks":points}.items() if k!="children"}

def lesson_pages(retrieval):
    pages=[]
    for subtopic in retrieval["unit"]["ncert_subtopics"]:
        children=[s for s in retrieval["sections"] if subtopic["id"] in s["subtopic_ids"]]
        # A master subsection can cover parent and child NCERT headings; show it once.
        seen={sid for p in pages for sid in p.get("source_section_ids",[])}
        children=[s for s in children if s["id"] not in seen]
        if not children:continue
        for source in ["Amit Master Notes","Supplementary NCERT"]:
            group=[s for s in children if s["source"]==source]
            if not group:continue
            kind=next((s["teaching_format"] for s in group if s["teaching_format"] in {"numerical","law","classification","formula"}),"concept")
            pages.append(teaching_page({"id":group[0]["id"],"title":subtopic["title"],"source":source,
                "subtopic_ids":[subtopic["id"]],"source_pages":sorted({p for s in group for p in s["source_pages"]}),
                "source_section_ids":[s["id"] for s in group],"teaching_format":kind,"children":group}))
    return sorted(pages,key=lambda p:p["source"]=="Supplementary NCERT")
