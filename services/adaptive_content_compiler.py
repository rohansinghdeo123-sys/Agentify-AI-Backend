"""Source-faithful teaching views, semantic pagination and active-reading cards.

This module deliberately has no database, provider or curriculum lookup. The
retriever supplies the selected unit's verified material; this compiler only
changes how that material is taught. A formula or worked solution is never
truncated to meet a word/page quota.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

COMPILER_VERSION = "adaptive_content_v3"
PROFICIENCIES = ("new_to_it", "know_a_little", "know_the_basics", "mostly_confident")
SOURCE_LABELS = {"MASTER_NOTES": "Master Notes", "NCERT_FALLBACK": "Supplementary NCERT"}
PROFICIENCY_LABELS = {
    "new_to_it": "Learning for the First Time",
    "know_a_little": "Know a Little",
    "know_the_basics": "Know the Basics",
    "mostly_confident": "Mostly Confident",
}
PROTECTED_KINDS = {
    "definition", "law", "principle", "statement", "formula", "terms", "units",
    "condition", "conditions", "limitation", "classification", "worked_example",
    "derivation", "misconception", "comparison",
}
STRICT_CORE_KINDS = PROTECTED_KINDS - {"worked_example"}
KNOWN_KINDS = PROTECTED_KINDS | {
    "explanation", "example", "connection", "takeaway", "practice", "clarification",
}
TYPE_ALIASES = {
    "conceptual": "concept", "explanatory": "concept", "principle": "law",
    "formula_based": "formula", "formula-based": "formula", "calculation": "numerical",
}
_LABEL_PRIORITY = ("worked_example", "derivation", "formula", "definition", "classification", "conditions", "condition", "misconception", "comparison", "law", "principle", "statement", "terms", "units", "limitation")
_SCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−", "01234567890123456789-+-")


def _scientific_tokens(text):
    normalized = text.translate(_SCRIPT_DIGITS)
    return set(re.findall(r"(?<![A-Za-z])(?:[A-Z][a-z]?\d*){2,}(?:\^[+-]?\d*[+-]?)?(?![A-Za-z])", normalized))


def _digest(value: Any, size=20) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:size]


def _ordered(values):
    return list(dict.fromkeys(values))


def _sentences(text):
    # A sentence boundary needs whitespace and a following word. Decimal
    # points, scientific powers, symbols and equation lines remain intact.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z][a-z]|[A-Z][A-Z][a-z]|[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉0-9]|[A-Z][A-Z][₀₁₂₃₄₅₆₇₈₉0-9])", text) if part.strip()]


def _critical_sentence(text):
    return bool(re.search(r"\d|[=≠≤≥⇌→]|\b(?:only|unless|except|provided|must|cannot|not|if|when|condition|limitation)\b", text, re.I))


def _source_blocks(section):
    blocks = section.get("blocks")
    if blocks:
        result = []
        for block in blocks:
            if not isinstance(block, dict) or not str(block.get("text") or "").strip():
                continue
            kind = str(block.get("kind") or "").lower()
            if kind not in KNOWN_KINDS:
                label = str(block.get("label") or "").lower()
                kind = next((name for name in _LABEL_PRIORITY if name.replace("_", " ") in label), "explanation")
            result.append({**deepcopy(block), "kind": kind, "text": str(block["text"]).strip()})
        return result
    text = str(section.get("text") or "").strip()
    if not text:
        return []
    teaching_type = TYPE_ALIASES.get(section.get("teaching_format"), section.get("teaching_format", "concept"))
    titled_example = bool(re.match(r"(?i)^\s*example\b",str(section.get("title") or "")) and re.search(r"\d|[=→]",text))
    worked = bool((titled_example or re.search(r"\b(?:given|solution|calculate|find|example)\b", text, re.I)) and re.search(r"\d|[=→]", text))
    if teaching_type == "formula" or (teaching_type == "numerical" and worked):
        # A source may teach a relationship and then solve several examples in
        # the same paragraph. Separate at explicit source example boundaries,
        # preserving every original solution as a complete, atomic block.
        markers=list(re.finditer(r"(?i)\bexample\s*:",text))
        if markers and markers[0].start():
            prefix=text[:markers[0].start()].strip()
            result=_source_blocks({"text":prefix,"teaching_format":"concept"})
            for index,marker in enumerate(markers):
                solution=text[marker.start():markers[index+1].start() if index+1<len(markers) else len(text)].strip()
                result.append({"kind":"worked_example","text":solution,"level":"essential"})
                # General restrictions/rearrangements that follow an example
                # remain visible even when the worked solution is expandable.
                for sentence in _sentences(solution):
                    if re.search(r"(?i)\b(?:rearranging|use solution|use solvent|not solvent|not solution|no unit|has no unit|can change with temperature|does not change with temperature|check that|must|only|unless)\b",sentence):
                        result.append({"kind":"conditions","text":sentence,"level":"essential"})
            return result
        # PDF equation/solution line breaks are not independent page breaks.
        kind = "worked_example" if worked else "formula"
        return [{"kind": kind, "text": text, "level": "essential"}]
    if teaching_type=="numerical" and re.search(r"=\s*(?:about\s+)?[+−-]?\d",text):
        sentences=_sentences(text)
        calculations=[s for s in sentences if re.search(r"=\s*(?:about\s+)?[+−-]?\d",s)]
        if calculations:
            result=[]
            for sentence in sentences:
                if sentence in calculations:
                    result.append({"kind":"worked_example","text":sentence,"level":"essential"})
                else:
                    result.extend(_source_blocks({"text":sentence,"teaching_format":"concept"}))
            return result
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result = []
    for index, paragraph in enumerate(paragraphs):
        sentences = _sentences(paragraph)
        # A PDF paragraph is a container, not a teaching role. In particular,
        # do not label the definition, explanation AND examples as one giant
        # protected definition: that defeats all proficiency adaptation.
        if teaching_type == "classification":
            result.extend({"kind":"classification", "text":s, "level":"essential"} for s in sentences)
            continue
        pending = []
        def flush():
            if pending:
                result.append({"kind":"explanation", "text":" ".join(pending), "level":"essential"})
                pending.clear()
        for sentence_index, sentence in enumerate(sentences):
            if re.search(r"\b(?:misconception|do not confuse|common mistake)\b", sentence, re.I):
                kind = "misconception"
            elif re.search(r"[=→⇌]|\b(?:SI unit|units? (?:is|are)|must|only|unless|except|provided)\b", sentence, re.I):
                kind = "formula" if re.search(r"[=→⇌]",sentence) else "conditions"
            elif re.search(r"\b(?:example|for instance|e\.g\.)", sentence, re.I):
                kind = "example"
            elif index == 0 and sentence_index == 0:
                kind = "law" if teaching_type == "law" else "definition"
            else:
                pending.append(sentence)
                continue
            flush()
            result.append({"kind":kind, "text":sentence, "level":"essential"})
        flush()
    return result


def _normalise_material(material):
    unit = material.get("unit") or {}
    allowed = {str(s["id"]) for s in unit.get("ncert_subtopics") or unit.get("subtopics") or []}
    if not unit.get("id") or not allowed:
        raise ValueError("Adaptive notes need the existing selected unit and mapped subtopics")
    sections, seen = [], set()
    for original in material.get("sections") or []:
        section = deepcopy(original)
        identity = str(section.get("id") or "")
        subtopics = _ordered(str(s) for s in section.get("subtopic_ids") or [])
        if not identity or identity in seen:
            raise ValueError("Retrieved sections need unique source identities")
        if not subtopics or not set(subtopics) <= allowed:
            raise ValueError("Retrieved section crosses the selected topic mapping")
        if section.get("source_type") not in SOURCE_LABELS:
            raise ValueError("Retrieved section needs an explicit authoritative source type")
        section.update(id=identity, subtopic_ids=subtopics, blocks=_source_blocks(section), _native_blocks=bool(original.get("blocks")))
        if not section["blocks"]:
            continue
        for block in section["blocks"]:
            if block.get("source_type", section["source_type"]) != section["source_type"]:
                raise ValueError("A teaching block cannot silently change its source")
            if block.get("subtopic_ids") and not set(block["subtopic_ids"]) <= set(subtopics):
                raise ValueError("A teaching block crosses its retrieved section")
            if "\ufffd" in block["text"]:
                raise ValueError("Source text has damaged scientific encoding")
        section["title"] = str(section.get("title") or unit.get("title") or "Learning topic").strip()
        section["source_pages"] = sorted({int(p) for p in section.get("source_pages") or [] if isinstance(p, int) and not isinstance(p, bool) and p > 0})
        seen.add(identity)
        sections.append(section)
    if not sections:
        raise ValueError("No verified teaching content was retrieved for this topic")
    return unit, allowed, sections


def _generator_support(generator, material, sections, proficiency, style, duration):
    """Optional, anchored enrichment cannot replace or delete source science."""
    if generator is None or all(s.get("_native_blocks") and len(s["blocks"]) > 1 for s in sections):
        return {}, "not_requested"
    try:
        result = generator(deepcopy(material), proficiency=proficiency, style=style, duration=deepcopy(duration))
        if not isinstance(result, dict):
            return {}, "rejected"
        source_by_id = {s["id"]: s for s in sections}
        additions = {}
        for block in result.get("support_blocks") or []:
            ids = list(block.get("source_section_ids") or [])
            if not ids or not set(ids) <= set(source_by_id) or len(ids) != 1:
                return {}, "rejected"
            source = source_by_id[ids[0]]
            evidence = str(block.get("evidence_text") or "").strip()
            source_text = "\n\n".join(b["text"] for b in source["blocks"])
            text = str(block.get("text") or "").strip()
            if not text or not evidence or evidence not in source_text:
                return {}, "rejected"
            if block.get("kind") not in {"explanation", "connection", "clarification"} or block.get("level") not in {"foundation", "extension"}:
                return {}, "rejected"
            # In particular, an enhancement may not invent numerical data,
            # reaction coefficients, dates or worked-example answers.
            if not set(re.findall(r"\d+(?:\.\d+)?", text)) <= set(re.findall(r"\d+(?:\.\d+)?", source_text)):
                return {}, "rejected"
            if not _scientific_tokens(text) <= _scientific_tokens(source_text):
                return {}, "rejected"
            # Formula-bearing additions must retain the actual retrieved
            # expression. Merely reusing its numbers could invert a ratio.
            normalized_source = re.sub(r"\s+", " ", source_text).casefold()
            for sentence in _sentences(text):
                if re.search(r"[=≠⇌→]|\b(?:equals|equal to)\b", sentence, re.I) and re.sub(r"\s+", " ", sentence).casefold() not in normalized_source:
                    return {}, "rejected"
            source_terms = set(re.findall(r"[a-z]{3,}", source_text.lower()))
            terms = set(re.findall(r"[a-z]{3,}", text.lower()))
            if terms and len(terms & source_terms) < min(3, len(terms)):
                return {}, "rejected"
            additions.setdefault(ids[0], []).append({"kind": block["kind"], "level": block["level"], "text": text,
                                                   "teacher_explanation": True, "evidence_text": evidence})
        return additions, "accepted" if additions else "not_available"
    except Exception:
        # Source-backed notes remain available when an optional provider is
        # unavailable, rejects its schema, or exceeds its budget.
        return {}, "unavailable"


def _include_block(block, proficiency, style, duration):
    kind, level = block["kind"], block.get("level", "essential")
    if kind in PROTECTED_KINDS or level == "essential":
        return True
    if level == "foundation":
        if proficiency == "new_to_it":
            return style != "concise" or kind not in {"connection", "example"}
        if proficiency == "know_a_little":
            return style == "detailed" or bool(block.get("necessary_prerequisite"))
        return style == "detailed"
    if level == "extension":
        return (style == "detailed" or duration["max"] >= 60) and style != "concise" and proficiency in {"new_to_it", "know_a_little"}
    return False


def _present_block(block, section, ordinal, proficiency, style):
    result = deepcopy(block)
    text = result["text"]
    protected = result["kind"] in PROTECTED_KINDS
    if proficiency != "new_to_it" and style != "detailed" and not protected:
        sentences = _sentences(text)
        lead = {"know_a_little":2,"know_the_basics":1,"mostly_confident":1}[proficiency]
        critical=[sentence for sentence in sentences if _critical_sentence(sentence)]
        # An authored teaching step keeps its explicit main claim beside its
        # qualifications. A compact extracted paragraph instead prioritises
        # the precise conditions for a basics-level recap; its complete source
        # explanation remains available on the same block.
        selected = [sentence for index, sentence in enumerate(sentences)
                    if (index < lead and (proficiency=="know_a_little" or
                        (proficiency=="know_the_basics" and (not critical or section.get("_native_blocks"))))) or _critical_sentence(sentence)]
        if not selected:
            selected=sentences[:lead]
        capsule = " ".join(selected)
        if capsule and capsule != text:
            result["text"] = capsule
            result["expanded_text"] = text
    source_ref = {
        "id": section["id"], "source_type": section["source_type"],
        "source_document": deepcopy(section.get("source_document") or {}),
        "source_pages": list(section["source_pages"]), "subtopic_ids": list(section["subtopic_ids"]),
    }
    result.update(id="block_" + _digest([section["id"], ordinal, block["text"]]),
                  source_type=section["source_type"], source_section_ids=[section["id"]],
                  source_pages=list(section["source_pages"]), subtopic_ids=list(section["subtopic_ids"]),
                  source_refs=[source_ref], content_anchor=[section["id"], ordinal])
    return result


def _worked_preview(text, proficiency):
    """An exact source problem/given, never a shortened partial solution."""
    first=_sentences(text)[0]
    sentences=_sentences(text)
    if len(sentences)>1:
        # A learner who knows the basics can keep an explicit source method
        # hint. Confident revision starts from the givens alone. Both reveal
        # the same complete solution, rather than printing incomplete steps.
        method=next((s for s in sentences[1:] if re.match(r"(?i)^(?:use|apply|start|begin|count|choose|place|balance|convert|divide|multiply)\b",s)),None)
        return " ".join([first,method]) if method and proficiency!="mostly_confident" else first
    # A single-line solved calculation can be attempted from its left-hand
    # quantity. The presentation explicitly calls this an example to attempt;
    # all equalities, substitutions and answers stay together on reveal.
    lhs=re.split(r"\s*=\s*",first,maxsplit=1)[0].strip()
    if lhs != first and (re.search(r"\b(?:of|in|for|mass|moles|number|percentage|density|molarity|molality)\b",lhs,re.I) or _scientific_tokens(lhs)):
        return lhs
    return first


def _adaptation_plan(sections, proficiency, style):
    if proficiency=="new_to_it" or style=="detailed":
        return set(),set()
    collapsed,previews=set(),set()
    seen_examples=set()
    worked=[]
    for section in sections:
        for ordinal,block in enumerate(section["blocks"]):
            key=(section["id"],ordinal)
            if block["kind"] in {"example","connection"}:
                scope=tuple(section["subtopic_ids"])
                if scope in seen_examples:
                    collapsed.add(key)
                else:
                    seen_examples.add(scope)
            if block["kind"]=="worked_example":
                worked.append((key,block))
    if len(worked)>1:
        # Keep a complete representative solution, favouring explicit method,
        # checks and exceptions. Other source problems invite retrieval before
        # revealing their full original solution. No solution is cut in half.
        representative=max(worked,key=lambda item:(bool(re.search(r"(?i)\b(?:exception|check|limiting|different|balance|condition)\b",item[1]["text"])),
            len(re.findall(r"[=→⇌]|\b(?:use|substitute|giving|because|therefore|so)\b",item[1]["text"],re.I))))[0]
        retained={representative}
        if proficiency=="know_a_little":
            # One fully worked example per mapped outcome refreshes a partly
            # familiar learner's method without repeating every routine case.
            seen_outcomes=set()
            for section in sections:
                for ordinal,block in enumerate(section["blocks"]):
                    scope=tuple(section["subtopic_ids"])
                    if block["kind"]=="worked_example" and scope not in seen_outcomes:
                        retained.add((section["id"],ordinal));seen_outcomes.add(scope)
        previews={key for key,_ in worked if key not in retained}
    if proficiency=="mostly_confident":
        for section in sections:
            for ordinal,block in enumerate(section["blocks"]):
                if block["kind"] in {"explanation","clarification","connection","takeaway"} and not any(_critical_sentence(s) for s in _sentences(block["text"])):
                    collapsed.add((section["id"],ordinal))
    return collapsed,previews


def _attach_source_details(presented,hidden):
    """Keep optional explanations available beside their source's core claim."""
    if not hidden:
        return presented
    if not presented:
        # A section consisting only of explanations still needs its actual
        # central claim visible; never replace science with a generic label.
        presented=[hidden.pop(0)]
    target=next((b for b in presented if b["kind"] in STRICT_CORE_KINDS),presented[0])
    if hidden:
        source=[target.get("expanded_text") or target["text"],*(b.get("expanded_text") or b["text"] for b in hidden)]
        target.update(expanded_text="\n\n".join(source),presentation="source_preview",
            expansion_label="Read the full source explanation")
    return presented


def _phase(block, has_core):
    if block["kind"] in {"worked_example", "derivation"}:
        return "worked"
    if block["kind"] in {"misconception", "comparison", "takeaway", "practice"}:
        return "consolidation"
    if block["kind"] in {"example", "connection"}:
        return "application" if has_core else "orientation"
    return "core"


def _section_steps(section, blocks, proficiency):
    if proficiency == "new_to_it" and section.get("teaching_format", "concept") == "concept":
        # Source-backed intuition precedes the formal definition for a first
        # reading; source order remains unchanged inside each teaching role.
        simple = [b for b in blocks if b["kind"] in {"explanation", "connection"}]
        if simple and blocks and blocks[0]["kind"] == "definition":
            blocks = simple + [b for b in blocks if b not in simple]
    runs, has_core = [], False
    for block in blocks:
        phase = _phase(block, has_core)
        has_core = has_core or phase == "core"
        if runs and runs[-1]["phase"] == phase and phase != "worked":
            runs[-1]["blocks"].append(block)
        else:
            runs.append({"title": section["title"], "phase": phase, "blocks": [block],
                         "teaching_format": section.get("teaching_format", "concept")})
    # A lone short context/recap block is not a useful stand-alone page.
    index = 0
    while index < len(runs):
        run = runs[index]
        if len(run["blocks"]) == 1 and run["phase"] != "worked" and len(runs) > 1:
            if index + 1 < len(runs) and runs[index+1]["phase"] != "worked":
                runs[index+1]["blocks"] = run["blocks"] + runs[index+1]["blocks"]
                runs.pop(index)
                continue
            if index and runs[index-1]["phase"] != "worked":
                runs[index-1]["blocks"].extend(run["blocks"])
                runs.pop(index)
                continue
        index += 1
    if proficiency == "new_to_it":
        return runs
    if proficiency == "know_a_little":
        result = []
        for run in runs:
            if result and ((result[-1]["phase"], run["phase"]) in {("orientation", "core"), ("application", "consolidation")}):
                result[-1]["blocks"].extend(run["blocks"])
                result[-1]["phase"] = run["phase"]
            else:
                result.append(run)
        return result
    # A concise conceptual recap can connect its examples/conditions on the
    # same page. Keep each worked solution/derivation intact as its own step.
    result = []
    for run in runs:
        if result and run["phase"] != "worked" and result[-1]["phase"] != "worked":
            result[-1]["blocks"].extend(run["blocks"])
        else:
            result.append(run)
    return result


def _subtopics(step):
    return _ordered(s for b in step["blocks"] for s in b["subtopic_ids"])


def _paginate_steps(steps, proficiency):
    pages = []
    for step in steps:
        if not step["blocks"]:
            continue
        merge = False
        if pages and _subtopics(pages[-1]) == _subtopics(step):
            previous = pages[-1]
            if proficiency == "mostly_confident":
                # A revision page follows one connected subtopic. Starting a
                # second worked solution is a genuine new learning step.
                worked_before = any(b["kind"] in {"worked_example", "derivation"} for b in previous["blocks"])
                worked_next = any(b["kind"] in {"worked_example", "derivation"} for b in step["blocks"])
                merge = not (worked_before and worked_next)
            elif len(previous["blocks"]) == 1 and previous["phase"] != "worked" and step["phase"] != "worked":
                merge = True
        if merge:
            pages[-1]["blocks"].extend(step["blocks"])
        else:
            pages.append(deepcopy(step))
    return pages


def _question_difficulty(question):
    declared = question.get("difficulty") or question.get("difficulty_level")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared in {1, 2, 3, 4}:
        return int(declared)
    labels = {"foundation":1, "foundational":1, "conceptual":2, "application":3, "challenge":4, "exam":4}
    if str(declared).lower() in labels:
        return labels[str(declared).lower()]
    text = question["question"].lower()
    if re.search(r"\b(?:trap|exception|justify|evaluate|predict|would happen|cannot|invalid)\b", text):
        return 4
    if re.search(r"\b(?:calculate|compare|distinguish|difference|determine|find|arrange)\b", text):
        return 3
    return 1 if re.search(r"^(?:what|name|identify|define)\b", text) else 2


def _questions(sections, allowed):
    result, seen = [], set()
    for section in sections:
        for supplied in section.get("questions") or []:
            question = deepcopy(supplied)
            if not isinstance(question.get("question"), str) or not isinstance(question.get("answer"), str):
                continue
            if not question["question"].strip() or not question["answer"].strip():
                continue
            ids = list(question.get("subtopic_ids") or question.get("source_section_ids") or section["subtopic_ids"])
            # Existing teaching questions use mapped subtopic IDs. A source
            # section ID supplied instead is translated only to its own map.
            if ids == [section["id"]]:
                ids = list(section["subtopic_ids"])
            if not set(ids) <= allowed or not set(ids) <= set(section["subtopic_ids"]):
                raise ValueError("A practice question crosses its retrieved source scope")
            identity = question["question"].strip().casefold()
            if identity in seen:
                continue
            seen.add(identity)
            question.update(question=question["question"].strip(), answer=question["answer"].strip(),
                            source_section_ids=ids, subtopic_ids=ids, source_refs=[section["id"]],
                            source_type=section["source_type"])
            question["difficulty"] = _question_difficulty(question)
            result.append(question)
    return result


def _derived_activity(page, proficiency):
    blocks = page["blocks"]
    worked = next((b for b in blocks if b["kind"] in {"worked_example", "derivation"}), None)
    misconception = next((b for b in blocks if b["kind"] == "misconception"), None)
    example = next((b for b in blocks if b["kind"] in {"example", "connection"}), None)
    condition = next((b for b in blocks if b["kind"] in {"condition", "conditions", "limitation"}), None)
    law = next((b for b in blocks if b["kind"] in {"law","principle","statement"}), None)
    classification = next((b for b in blocks if b["kind"]=="classification"), None)
    formula = next((b for b in blocks if b["kind"]=="formula"), None)
    topic='“'+page['title']+'”'
    definitions = [b for b in blocks if b["kind"] == "definition"]
    if worked:
        numerical_prompts = {
            "new_to_it": ("quick_knowledge_check", "Read the givens", "Which values are given in this worked example, and which quantity must be found?"),
            "know_a_little": ("numerical_challenge", "Choose the relationship", "Which formula connects the givens with the required quantity in this solution? Explain that choice."),
            "know_the_basics": ("numerical_challenge", "Check the method", "Check the substitution and units in this solution. How do they confirm what the result represents?"),
            "mostly_confident": ("numerical_challenge", "Reconstruct the reasoning", "Reconstruct this solution without looking. Which stated condition and unit check would expose an invalid application?"),
        }
        kind, heading, question = numerical_prompts[proficiency]
        question=f"For {topic}: {question[0].lower()+question[1:]}"
        activity = kind, heading, question, worked
    elif misconception and proficiency in {"know_a_little", "mostly_confident"}:
        prompt = f"In {topic}, which mistaken assumption is corrected? Explain the correction before revealing it." if proficiency == "know_a_little" else f"For {topic}, how would you refute the mistaken assumption using the precise source condition? Give the reasoning before checking it."
        activity = ("common_misconception", "Spot the trap", prompt, misconception)
    elif len(definitions) >= 2 and proficiency in {"know_the_basics", "mostly_confident"}:
        prompt = f"Compare the two definitions in {topic}. Which defining condition distinguishes them?" if proficiency == "know_the_basics" else f"In {topic}, would exchanging the defining conditions preserve both meanings? Justify the distinction from the source definitions."
        activity = ("spot_the_difference", "Keep the ideas distinct", prompt, definitions[0])
        answer = "\n\n".join(b["text"] for b in definitions[:2])
    elif condition and proficiency == "mostly_confident":
        activity = ("apply_your_knowledge", "Check the condition", f"Before applying the idea in {topic}, which stated condition must hold, and why does it matter?", condition)
    elif law:
        question=f"State the law in {topic} in your own words, retaining its qualifications." if proficiency in {"new_to_it","know_a_little"} else f"Recall the precise law in {topic}. Which qualification would be lost in an overgeneralised statement?"
        activity=("quick_knowledge_check","State the law precisely",question,law)
    elif classification:
        question=f"In {topic}, what property separates the categories? Explain how a stated characteristic identifies its category." if proficiency in {"new_to_it","know_a_little"} else f"For the classification in {topic}, distinguish the categories using their defining properties rather than memorising names."
        activity=("spot_the_difference","Classify by the defining property",question,classification)
        answer="\n\n".join(b.get("expanded_text") or b["text"] for b in blocks if b["kind"]=="classification")
    elif formula:
        question=f"For the relationship in {topic}, what do the symbols and stated units represent?" if proficiency in {"new_to_it","know_a_little"} else f"Before using the relationship in {topic}, which units and stated conditions need checking? Explain what each quantity represents."
        activity=("apply_your_knowledge","Read the relationship",question,formula)
    elif example and proficiency == "new_to_it":
        activity = ("everyday_connection", "Connect the example", f"How does the example in {topic} demonstrate the idea? Connect the observation to its source explanation.", example)
    else:
        prompts = {
            "new_to_it": ("think_about_it", "Explain it simply", "How would you explain the central idea on this page in your own words?"),
            "know_a_little": ("quick_knowledge_check", "Check the reasoning", "What is the central claim on this page, and which detail makes it precise?"),
            "know_the_basics": ("apply_your_knowledge", "Use the defining idea", "Identify the defining condition on this page and explain how you would recognise a correct application."),
            "mostly_confident": ("quick_knowledge_check", "Precision check", "Recall the precise statement on this page. Which condition or qualification must you retain?"),
        }
        kind, heading, prompt = prompts[proficiency]
        chosen = next((b for b in blocks if b["kind"] in PROTECTED_KINDS), blocks[0])
        activity = kind, heading, f"For {topic}: {prompt[0].lower()+prompt[1:]}", chosen
    kind, heading, question, chosen = activity
    return {"type":kind, "heading":heading, "question":question,
            "hint":"Compare your reasoning with the source explanation on this page." if proficiency in {"new_to_it", "know_a_little"} else None,
            "answer":locals().get("answer", chosen.get("expanded_text") or chosen["text"]),
            "source_section_ids":list(page["source_section_ids"]),
            "source_refs":_ordered(ref["id"] for b in blocks for ref in b["source_refs"])}


def _attach_practice(pages, questions, proficiency):
    target = PROFICIENCIES.index(proficiency) + 1
    used = set()
    last_interrupt = -2
    for index, page in enumerate(pages):
        boundary = index == len(pages)-1 or set(page["source_section_ids"]) != set(pages[index+1]["source_section_ids"])
        meaningful_break = boundary or any(b["kind"] in {"worked_example", "misconception", "comparison", "takeaway"} for b in page["blocks"])
        if not meaningful_break or index-last_interrupt < 2:
            continue
        candidates = [(i,q) for i,q in enumerate(questions) if i not in used and q['difficulty'] == target and set(q["subtopic_ids"]) <= set(page["source_section_ids"])]
        if candidates:
            chosen_index, chosen = min(candidates, key=lambda item:(abs(item[1]["difficulty"]-target), item[0]))
            used.add(chosen_index)
            kinds = {1:("quick_knowledge_check","Pause and recall"),2:("think_about_it","Check your understanding"),
                     3:("apply_your_knowledge","Apply the idea"),4:("numerical_challenge" if any(b['kind']=='worked_example' for b in page['blocks']) else "apply_your_knowledge","Test the precise reasoning")}
            kind, heading = kinds[target]
            page["interrupts"] = [{"type":kind,"heading":heading,"question":chosen["question"],"answer":chosen["answer"],
                                   "hint":chosen.get("hint"),"source_section_ids":chosen["subtopic_ids"],"source_refs":chosen["source_refs"]}]
        else:
            page["interrupts"] = [_derived_activity(page,proficiency)]
        last_interrupt = index
    if not any(p.get("interrupts") for p in pages):
        pages[-1]["interrupts"] = [_derived_activity(pages[-1],proficiency)]
    # Revision stays on an existing meaningful final page. Select distinct
    # outcome/type combinations, not an arbitrary final-page question quota.
    revision, covered = [], set()
    for i, question in sorted(enumerate(questions), key=lambda item:(abs(item[1]["difficulty"]-target),item[0])):
        if i in used:
            continue
        outcome = (tuple(question["subtopic_ids"]), question["difficulty"])
        if outcome in covered:
            continue
        if question["difficulty"] > target+1 and target < 3:
            continue
        if target >= 3 and question['difficulty'] < target-1:
            continue
        covered.add(outcome)
        revision.append(deepcopy(question))
    if revision:
        pages[-1]["questions"] = revision


def compile_adaptive_lesson(material, *, proficiency="new_to_it", style="balanced", duration=None, generator=None):
    """Compile one source-verified Planning topic into a proficiency-aware view.

    ``duration`` is authoritative workload, never re-estimated here. Student
    session time is intentionally not used to delete source learning outcomes;
    the caller selects a portion from the resulting page array.
    """
    if proficiency not in PROFICIENCIES:
        raise ValueError("Unknown Planning proficiency")
    if style not in {"balanced", "concise", "detailed"}:
        raise ValueError("Unknown explanation preference")
    unit, allowed, sections = _normalise_material(material)
    duration = deepcopy(duration or unit.get("estimated_minutes") or {})
    if (not isinstance(duration.get("min"), int) or isinstance(duration.get("min"), bool)
            or not isinstance(duration.get("max"), int) or isinstance(duration.get("max"), bool)
            or duration["min"] <= 0 or duration["max"] < duration["min"]):
        raise ValueError("A configured minimum/maximum topic duration is required")
    support, enrichment_status = _generator_support(generator, material, sections, proficiency, style, duration)
    collapsed,previews=_adaptation_plan(sections,proficiency,style)
    steps, essential_ids, seen = [], [], {}
    for section in sections:
        presented,hidden = [],[]
        for ordinal, block in enumerate([*section["blocks"], *support.get(section["id"], [])]):
            if not _include_block(block,proficiency,style,duration):
                continue
            candidate = _present_block(block,section,ordinal,proficiency,style)
            key=(section["id"],ordinal)
            if key in collapsed:
                hidden.append(candidate)
                continue
            if key in previews:
                preview=_worked_preview(block["text"],proficiency)
                if preview != block["text"]:
                    candidate.update(text=preview,expanded_text=block["text"],presentation="source_preview",
                        expansion_label="Reveal the full worked solution",label="Try the source example")
            if block.get("level","essential") == "essential" or block["kind"] in PROTECTED_KINDS:
                essential_ids.append(candidate["id"])
            key = (candidate["source_type"],tuple(candidate["subtopic_ids"]),re.sub(r"\s+"," ",block["text"]).strip())
            if key in seen:
                old = seen[key]
                old["source_section_ids"] = _ordered([*old["source_section_ids"],section["id"]])
                old["source_refs"] += [ref for ref in candidate["source_refs"] if ref not in old["source_refs"]]
                # Completion coverage uses the retained block identity.
                if candidate["id"] in essential_ids:
                    essential_ids.remove(candidate["id"])
                    essential_ids.append(old["id"])
                continue
            seen[key] = candidate
            presented.append(candidate)
        steps.extend(_section_steps(section,_attach_source_details(presented,hidden),proficiency))
    grouped = _paginate_steps(steps,proficiency)
    pages = []
    for index, group in enumerate(grouped):
        blocks = group["blocks"]
        refs = []
        for block in blocks:
            refs.extend(ref for ref in block["source_refs"] if ref not in refs)
        subtopics = _ordered(s for block in blocks for s in block["subtopic_ids"])
        source_types = _ordered(ref["source_type"] for ref in refs)
        anchor = [block["content_anchor"] for block in blocks]
        page = {"page_number":index+1,"title":group["title"],"blocks":blocks,"interrupts":[],
                "source_section_ids":subtopics,"subtopic_ids":subtopics,"source_refs":refs,
                "source_pages":sorted({p for ref in refs for p in ref["source_pages"]}),
                "source_types":source_types,"source":" + ".join(SOURCE_LABELS[s] for s in source_types),
                "teaching_format":group["teaching_format"],"content_anchor":"anchor_"+_digest(anchor),
                "text":"\n\n".join(block["text"] for block in blocks)}
        # Hash the actual taught content and immutable source evidence, not an
        # array index: regeneration/proficiency changes cannot fake completion.
        page["id"] = "adaptive_" + _digest([COMPILER_VERSION,unit["id"],proficiency,style,material.get("source_fingerprint"),anchor,blocks])
        pages.append(page)
    if not pages:
        raise ValueError("The selected teaching view has no meaningful source content")
    questions = _questions(sections,allowed)
    _attach_practice(pages,questions,proficiency)
    for page in pages:
        if page.get("questions"):
            page["text"] += "\n\nCheck your understanding\n"+"\n".join(q["question"] for q in page["questions"])
        if page.get("interrupts"):
            page["text"] += "\n\n"+"\n".join(i["question"] for i in page["interrupts"])
        page["id"] = "adaptive_" + _digest([COMPILER_VERSION,unit["id"],proficiency,style,
            material.get("source_fingerprint"),page["content_anchor"],page["blocks"],page.get("questions",[]),page["interrupts"]])
    return {
        "pages":pages,"compiler_version":COMPILER_VERSION,"proficiency":proficiency,
        "proficiency_label":PROFICIENCY_LABELS[proficiency],"explanation_style":style,
        "estimated_minutes":duration,"source_fingerprint":material.get("source_fingerprint"),
        "source_types":_ordered(s["source_type"] for s in sections),
        "covered_subtopic_ids":_ordered(s for section in sections for s in section["subtopic_ids"]),
        "essential_block_ids":_ordered(essential_ids),
        "adaptation":{"level":PROFICIENCIES.index(proficiency)+1,
                      "depth":"extended" if duration["max"] >= 60 else "developed" if duration["max"] >= 40 else "focused",
                      "full_explanations":proficiency in {"new_to_it","know_a_little"} or style=="detailed",
                      "enrichment_status":enrichment_status},
        "conflicts":deepcopy(material.get("conflicts") or []),
        "missing_subtopics":deepcopy(material.get("missing_subtopics") or []),
    }
