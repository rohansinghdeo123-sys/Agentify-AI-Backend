"""Teaching adaptation is driven by scoped source content, not fixed pages."""
from copy import deepcopy
from decimal import Decimal

import pytest

from services.adaptive_content_compiler import PROFICIENCIES, compile_adaptive_lesson


def block(kind, text, level="essential", **extra):
    return {"kind":kind,"text":text,"level":level,**extra}


@pytest.fixture
def physics():
    return {
        "curriculum":{"class_level":"Class 9","subject":"Physics","chapter_slug":"force_and_motion"},
        "unit":{"id":"physics_force","title":"Force and acceleration","estimated_minutes":{"min":45,"max":55},
                "ncert_subtopics":[{"id":"force","title":"Force and acceleration"},{"id":"balance","title":"Balanced forces"}]},
        "source_fingerprint":"physics-source-1",
        "sections":[
            {"id":"master-force","title":"Why pushes change motion","source_type":"NCERT","subtopic_ids":["force"],
             "source_document":{"id":"teacher-notes","title":"NCERT Mechanics"},"source_pages":[1],"teaching_format":"concept",
             "blocks":[
                 block("example","Push an empty trolley and then a loaded trolley. The trolleys respond differently to a similar push. This observation motivates a relationship between force, mass and change in motion."),
                 block("explanation","A change in velocity is acceleration. The relationship connects the net force acting on an object with its acceleration. The mass describes how strongly the object resists this change."),
                 block("definition","A force is a push or pull due to an interaction. Net force is the vector sum of the forces acting on the object."),
                 block("explanation","Velocity includes both speed and direction. Acceleration describes change in velocity with time, rather than simply how fast an object is moving.","foundation"),
                 block("formula","For constant mass, F = ma. F is net force in newtons (N), m is mass in kilograms (kg), and a is acceleration in m s⁻²."),
                 block("conditions","Use the net external force and a consistent direction convention. The expression F = ma here assumes constant mass."),
                 block("worked_example","Given m = 2 kg and net force F = 6 N, find acceleration. Use a = F/m. Substitute: a = (6 N)/(2 kg) = 3 m s⁻². The object changes velocity at 3 m s⁻² in the force direction."),
                 block("misconception","A moving object need not have a nonzero net force. A zero net force gives zero acceleration, not necessarily zero velocity."),
                 block("takeaway","Force, mass and acceleration answer different questions. Link them through the net force relationship. Keep the physical condition in mind before applying it."),
             ],
             "questions":[
                 {"question":"What does net force mean?","answer":"The vector sum of all the forces acting on the object.","difficulty":1,"source_section_ids":["force"]},
                 {"question":"Why use net force rather than one selected force?","answer":"Acceleration depends on the combined vector effect of the forces.","difficulty":2,"source_section_ids":["force"]},
                 {"question":"Calculate the acceleration when a net force of 6 N acts on a mass of 2 kg.","answer":"a = F/m = 6/2 = 3 m s⁻².","difficulty":3,"source_section_ids":["force"]},
                 {"question":"Which assumption would invalidate using this constant-mass expression without further analysis?","answer":"A changing mass needs further analysis; the stated form assumes constant mass.","difficulty":4,"source_section_ids":["force"]},
             ]},
            {"id":"master-force-application","title":"Keep the physical meanings separate","source_type":"NCERT","subtopic_ids":["force"],
             "source_pages":[2],"teaching_format":"concept",
             "blocks":[
                 block("example","A trolley can keep moving after your push ends. Motion alone does not identify the net force. The important observation is whether its velocity changes."),
                 block("explanation","Check which quantity the question asks for before choosing a calculation. Distinguish a force from a change in motion. The relationship connects them through mass."),
                 block("connection","The same relationship can be read in different directions: known mass and force determine acceleration, while known mass and acceleration determine net force.","foundation"),
             ]},
            {"id":"ncert-balance","title":"Balanced and unbalanced forces","source_type":"NCERT","subtopic_ids":["balance"],
             "source_document":{"title":"NCERT Physics"},"source_pages":[7],"teaching_format":"classification",
             "blocks":[
                 block("definition","Balanced forces have a zero vector sum. They produce no acceleration."),
                 block("definition","Unbalanced forces have a nonzero vector sum. The net force determines the acceleration of the constant-mass object."),
                 block("example","Equal opposite forces on the trolley balance each other. Neither force needs to vanish individually; their vector sum is zero."),
                 block("comparison","Individual forces and net force are different quantities. Balanced forces can act on an object that is already moving at constant velocity."),
             ]},
        ],"conflicts":[],"missing_subtopics":[],
    }


def visible_words(lesson):
    return sum(len(b["text"].split()) for p in lesson["pages"] for b in p["blocks"])


def all_blocks(lesson):
    return [b for p in lesson["pages"] for b in p["blocks"]]


def test_three_views_are_different_and_decrease_depth_without_losing_core_science(physics):
    views = [compile_adaptive_lesson(physics,proficiency=p) for p in PROFICIENCIES]
    assert len({tuple(page["text"] for page in lesson["pages"]) for lesson in views}) == 3
    assert [visible_words(v) for v in views] == sorted([visible_words(v) for v in views],reverse=True)
    assert [len(v["pages"]) for v in views] == sorted([len(v["pages"]) for v in views],reverse=True)
    assert len({len(v["pages"]) for v in views}) >= 3
    for lesson in views:
        assert lesson["estimated_minutes"] == {"min":45,"max":55}
        blocks = all_blocks(lesson)
        for source in physics["sections"]:
            for original in source["blocks"]:
                if original["kind"] in {"definition","formula","conditions","worked_example","misconception","comparison"}:
                    assert any(b["text"] == original["text"] for b in blocks)
        assert {s for p in lesson["pages"] for s in p["source_section_ids"]} == {"force","balance"}
        assert lesson["essential_block_ids"]


def test_concise_capsules_keep_complete_original_explanations_available(physics):
    beginner = compile_adaptive_lesson(physics,proficiency="new_to_it")
    confident = compile_adaptive_lesson(physics,proficiency="mostly_confident")
    assert all("expanded_text" not in b for b in all_blocks(beginner))
    capsules = [b for b in all_blocks(confident) if b.get("expanded_text")]
    assert capsules
    original = {b["text"] for s in physics["sections"] for b in s["blocks"]}
    assert all(all(part in original for part in b["expanded_text"].split("\n\n")) for b in capsules)
    assert all(b["text"] in b["expanded_text"] or all(fragment.rstrip(".").strip() in b["expanded_text"] for fragment in b["text"].split(". ")) for b in capsules)
    for b in all_blocks(confident):
        if b["kind"] in {"definition","formula","conditions"}:
            assert b["text"] in original
            if b.get("expanded_text"):
                assert b["presentation"]=="source_preview" and b["expanded_text"].startswith(b["text"])


def test_preference_and_configured_workload_control_support_not_source_outcomes(physics):
    balanced = compile_adaptive_lesson(physics,proficiency="mostly_confident")
    detailed = compile_adaptive_lesson(physics,proficiency="mostly_confident",style="detailed")
    assert visible_words(detailed) > visible_words(balanced)
    with_extension = deepcopy(physics)
    with_extension["sections"][0]["blocks"].append(block("explanation","Reading the relationship in reverse can connect an acceleration measurement with the net force that caused it.","extension"))
    short = compile_adaptive_lesson(with_extension,duration={"min":20,"max":25})
    long = compile_adaptive_lesson(with_extension,duration={"min":60,"max":75})
    assert any(b["level"] == "extension" for b in all_blocks(long))
    assert not any(b["level"] == "extension" for b in all_blocks(short))
    assert long["adaptation"]["depth"] == "extended"
    assert short["adaptation"]["depth"] == "focused"


def test_history_uses_the_same_engine_without_science_specific_or_fixed_page_logic():
    material = {"unit":{"id":"civics-democracy","title":"Democratic government","estimated_minutes":{"min":15,"max":20},
                        "ncert_subtopics":[{"id":"participation","title":"Participation"}]},
                "sections":[{"id":"history-master","title":"Who participates?","source_type":"NCERT","subtopic_ids":["participation"],
                             "text":"Democracy is a form of government in which the rulers are elected by the people.\n\nPolitical equality gives each adult citizen an equal voting right. This principle concerns participation in choosing representatives.","source_pages":[3]}]}
    for proficiency in PROFICIENCIES:
        lesson = compile_adaptive_lesson(material,proficiency=proficiency)
        assert 0 < len(lesson["pages"]) < 10
        assert lesson["source_types"] == ["NCERT"]
        assert all(p["source"] == "NCERT" for p in lesson["pages"])
        text = " ".join(p["text"] for p in lesson["pages"])
        assert "elected by the people" in text
        assert "Chemistry" not in text and "NCERT" not in text


def test_numerical_solution_is_an_atomic_visible_step_with_exact_values_and_units():
    solution = "Given m = 2.00 kg and F = 6.00 N, find a.\nUse a = F/m.\na = 6.00/2.00 = 3.00 m s⁻².\nThe acceleration is directed along the net force."
    material = {"unit":{"id":"numerical","title":"A constant-mass calculation","estimated_minutes":{"min":20,"max":25},
                        "ncert_subtopics":[{"id":"calculation","title":"Calculation"}]},
                "sections":[{"id":"worked-source","title":"Calculate the acceleration","source_type":"NCERT", "subtopic_ids":["calculation"],
                             "text":solution,"source_pages":[4],"teaching_format":"numerical"}]}
    assert Decimal("6.00")/Decimal("2.00") == Decimal("3.00")
    for proficiency in PROFICIENCIES:
        lesson = compile_adaptive_lesson(material,proficiency=proficiency)
        assert len(lesson["pages"]) == 1
        assert lesson["pages"][0]["blocks"][0]["text"] == solution
        assert "expanded_text" not in lesson["pages"][0]["blocks"][0]


def test_chemical_formulas_charges_brackets_and_reaction_coefficients_are_unchanged():
    formulas = "Ca(OH)₂ + H₂SO₄ → CaSO₄ + 2H₂O. Fe³⁺ and SO₄²⁻ retain their stated ionic charges. C₆H₁₂O₆ contains the stated atom counts."
    material = {"unit":{"id":"chemistry","title":"Chemical notation","estimated_minutes":{"min":15,"max":20},"ncert_subtopics":[{"id":"notation"}]},
                "sections":[{"id":"formula-source","title":"Balance and identify charges","source_type":"NCERT", "subtopic_ids":["notation"],
                             "source_pages":[2],"blocks":[block("formula",formulas)]}]}
    for proficiency in PROFICIENCIES:
        assert compile_adaptive_lesson(material,proficiency=proficiency)["pages"][0]["blocks"][0]["text"] == formulas


def test_provenance_preserves_mapped_ncert_sources(physics):
    lesson = compile_adaptive_lesson(physics)
    assert lesson["source_types"] == ["NCERT"]
    sections = {s["id"]:s for s in physics["sections"]}
    for page in lesson["pages"]:
        assert page["source_refs"] and page["source_pages"]
        for b in page["blocks"]:
            assert all(sections[ref["id"]]["source_type"] == b["source_type"] for ref in b["source_refs"])
            assert set(b["subtopic_ids"]) <= set(sections[b["source_section_ids"][0]]["subtopic_ids"])
            assert all(ref["id"] in sections for ref in b["source_refs"])
    assert any(p["source"] == "NCERT" for p in lesson["pages"])


def test_pattern_interrupts_are_source_scoped_answered_and_adapted(physics):
    views = {p:compile_adaptive_lesson(physics,proficiency=p) for p in PROFICIENCIES}
    signatures = set()
    for proficiency, lesson in views.items():
        interrupts = [i for p in lesson["pages"] for i in p["interrupts"]]
        assert interrupts
        assert all(i["type"] and i["heading"] and i["question"] and i["answer"] for i in interrupts)
        assert all(set(i["source_section_ids"]) <= {"force","balance"} for i in interrupts)
        assert all(set(i["source_refs"]) <= {s["id"] for s in physics["sections"]} for i in interrupts)
        signatures.add(tuple((i["type"],i["question"]) for i in interrupts))
    assert len(signatures) == 3


def test_page_ids_are_stable_but_source_or_view_changes_cannot_fake_completion(physics):
    original = deepcopy(physics)
    first = compile_adaptive_lesson(physics)
    second = compile_adaptive_lesson(physics)
    assert physics == original
    assert first == second
    assert len({p["id"] for p in first["pages"]}) == len(first["pages"])
    assert all(p["content_anchor"] and p["source_refs"] for p in first["pages"])
    changed = deepcopy(physics)
    changed["source_fingerprint"] = "new-source"
    assert not {p["id"] for p in first["pages"]} & {p["id"] for p in compile_adaptive_lesson(changed)["pages"]}
    assert not {p["id"] for p in first["pages"]} & {p["id"] for p in compile_adaptive_lesson(physics,proficiency="mostly_confident")["pages"]}
    question_changed = deepcopy(physics)
    question_changed["sections"][0]["questions"][0]["answer"] += " Retain the vector direction."
    assert first != compile_adaptive_lesson(question_changed)


@pytest.mark.parametrize("mutation",[
    lambda m:m["sections"][0].update(subtopic_ids=["unrelated"]),
    lambda m:m["sections"][0].update(source_type="guessed_source"),
    lambda m:m["sections"][0]["blocks"][0].update(source_type="guessed_source"),
    lambda m:m["sections"][0]["questions"][0].update(source_section_ids=["another-topic"]),
])
def test_unrelated_or_misattributed_content_is_rejected(physics,mutation):
    mutation(physics)
    with pytest.raises(ValueError):
        compile_adaptive_lesson(physics)


def test_optional_generation_can_only_add_anchored_support_and_never_remove_core():
    material = {"unit":{"id":"density","title":"Density","estimated_minutes":{"min":20,"max":25},"ncert_subtopics":[{"id":"density"}]},
                "sections":[{"id":"density-master","title":"Density","source_type":"NCERT", "subtopic_ids":["density"],
                             "text":"Density = mass / volume. Density connects mass to the volume occupied.","source_pages":[1]}]}
    def generated(*args,**kwargs):
        return {"support_blocks":[{"kind":"explanation","level":"foundation","text":"Mass and occupied volume both matter when discussing density.",
                                  "source_section_ids":["density-master"],"evidence_text":"Density connects mass to the volume occupied."}]}
    lesson = compile_adaptive_lesson(material,generator=generated)
    assert lesson["adaptation"]["enrichment_status"] == "accepted"
    assert any(b["teacher_explanation"] for b in all_blocks(lesson) if b.get("teacher_explanation"))
    assert all(sentence in " ".join(b["text"] for b in all_blocks(lesson)) for sentence in material["sections"][0]["text"].split(". "))
    for bad in [
        {"kind":"explanation","level":"foundation","text":"Density = volume / mass.","source_section_ids":["density-master"],"evidence_text":"Density = mass / volume."},
        {"kind":"explanation","level":"foundation","text":"Density is 999 kg m⁻³.","source_section_ids":["density-master"],"evidence_text":"Density = mass / volume."},
        {"kind":"explanation","level":"foundation","text":"A new unrelated explanation.","source_section_ids":["not-retrieved"],"evidence_text":"Density = mass / volume."},
    ]:
        result = compile_adaptive_lesson(material,generator=lambda *a,**k:{"support_blocks":[bad]})
        assert result["adaptation"]["enrichment_status"] == "rejected"
        assert " ".join(b["text"] for b in all_blocks(result)) == material["sections"][0]["text"]
    result = compile_adaptive_lesson(material,generator=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("Provider unavailable")))
    assert result["adaptation"]["enrichment_status"] == "unavailable"
    assert result["pages"]


def test_no_empty_forced_revision_page_or_artificial_ten_page_target(physics):
    for proficiency in PROFICIENCIES:
        lesson = compile_adaptive_lesson(physics,proficiency=proficiency)
        pages = lesson["pages"]
        assert [p["page_number"] for p in pages] == list(range(1,len(pages)+1))
        assert len(pages) != 10
        assert all(p["title"].strip() and p["blocks"] and p["text"].strip() for p in pages)
        assert all("completed 30%" not in p["text"] and "progress bar" not in p["text"] for p in pages)


def test_missing_material_is_not_replaced_with_invented_content(physics):
    physics["sections"] = []
    with pytest.raises(ValueError,match="No verified"):
        compile_adaptive_lesson(physics)


def test_advanced_example_selection_keeps_core_and_a_full_solution_with_atomic_reveals():
    text="Density = mass / volume. Units are g cm⁻³. Example: mass is 60 g and volume is 20 cm³. Density = 60/20 = 3 g cm⁻³. Example: mass is 80 g and volume is 20 cm³. Density = 80/20 = 4 g cm⁻³."
    material={"unit":{"id":"density","title":"Density","estimated_minutes":{"min":30,"max":35},"ncert_subtopics":[{"id":"density"}]},
        "sections":[{"id":"master-density","title":"Density","source_type":"NCERT","subtopic_ids":["density"],
            "source_pages":[1],"teaching_format":"numerical","text":text}]}
    for proficiency in PROFICIENCIES:
        blocks=all_blocks(compile_adaptive_lesson(material,proficiency=proficiency))
        assert any(b['text']=='Density = mass / volume.' for b in blocks)
        assert any('Units are g cm⁻³.' in b['text'] for b in blocks)
        worked=[b for b in blocks if b['kind']=='worked_example']
        assert any('Density = ' in b['text'] and not b.get('presentation') for b in worked)
        assert all('Density = ' in (b.get('expanded_text') or b['text']) for b in worked)
        for b in worked:
            if b.get('presentation')=='source_preview':
                assert b['expansion_label']=='Reveal the full worked solution'
                assert b['expanded_text'] in text and b['text'] in b['expanded_text']
                assert b['expanded_text'].endswith('g cm⁻³.')


def test_molecular_mass_calculations_are_separate_atomic_examples_not_a_generic_formula():
    material={"unit":{"id":"molecular-mass","title":"Molecular mass","estimated_minutes":{"min":20,"max":25},"ncert_subtopics":[{"id":"masses"}]},
        "sections":[{"id":"masses-master","title":"Molecular mass","source_type":"NCERT","subtopic_ids":["masses"],
            "source_pages":[2],"teaching_format":"numerical","text":"Add the atomic masses of all atoms in one molecule. H₂O = 2 x 1 + 16 = 18 u. CO₂ = 12 + 2 x 16 = 44 u."}]}
    first=compile_adaptive_lesson(material)
    worked=[b for b in all_blocks(first) if b['kind']=='worked_example']
    assert [b['text'] for b in worked]==['H₂O = 2 x 1 + 16 = 18 u.','CO₂ = 12 + 2 x 16 = 44 u.']
    confident=compile_adaptive_lesson(material,proficiency='mostly_confident')
    assert any(b.get('presentation')=='source_preview' for b in all_blocks(confident))
    assert any(b['kind']=='worked_example' and not b.get('presentation') for b in all_blocks(confident))


def test_default_body_adaptation_is_not_earned_by_interrupts_or_labels(physics):
    views=[compile_adaptive_lesson(physics,proficiency=p) for p in PROFICIENCIES]
    body=lambda lesson:' '.join(b['text'] for page in lesson['pages'] for b in page['blocks'])
    assert len({body(lesson) for lesson in views})==3
    for lesson in views:
        for page in lesson['pages']:
            for activity in page['interrupts']:
                if activity['source_refs'] and activity['question'] not in {q['question'] for s in physics['sections'] for q in s.get('questions',[])}:
                    assert page['title'] in activity['question']
