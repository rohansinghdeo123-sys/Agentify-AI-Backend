import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from Logic.knowledge_graph import knowledge_graph
from Logic.tools.knowledge_search import SECTION_FILE_MAP, search_knowledge_base

ARTIFACT_DATA_NOT_AVAILABLE = "Artifact data not available for this section yet"

HYDROCARBON_ARTIFACT_METADATA = {
    "alkanes": {
        "title": "Alkanes",
        "definition": "Alkanes are saturated hydrocarbons containing only single covalent bonds between carbon atoms.",
        "core_explanation": "Alkanes contain the maximum possible number of hydrogen atoms. Their structure explains why they are relatively unreactive and why substitution is their characteristic reaction.",
        "properties": ["Saturated hydrocarbons", "Carbon-carbon single bonds", "General formula CₙH₂ₙ₊₂", "Substitution reactions", "Boiling point rises with molecular mass"],
        "key_points": ["Open-chain alkanes follow the general formula CₙH₂ₙ₊₂.", "Methane is the simplest alkane.", "Alkanes are also called paraffins because they are relatively unreactive.", "Chain isomerism starts from butane.", "They mainly undergo combustion and substitution reactions."],
        "examples": ["Methane, ethane, propane, and butane are common examples of alkanes."],
    },
    "alkenes": {
        "title": "Alkenes",
        "definition": "Alkenes are unsaturated hydrocarbons containing at least one carbon-carbon double bond.",
        "core_explanation": "The carbon-carbon double bond contains one sigma bond and one pi bond. The weaker pi bond makes alkenes more reactive than alkanes.",
        "properties": ["Unsaturated hydrocarbons", "Carbon-carbon double bond", "General formula CₙH₂ₙ", "Addition reactions", "Geometrical isomerism"],
        "key_points": ["Open-chain alkenes with one double bond follow the general formula CₙH₂ₙ.", "The double bond contains one sigma bond and one pi bond.", "Alkenes are more reactive than alkanes.", "They mainly undergo addition reactions.", "Some alkenes show cis-trans isomerism."],
        "examples": ["Ethene, propene, and but-2-ene are common examples of alkenes."],
    },
    "alkynes": {
        "title": "Alkynes",
        "definition": "Alkynes are unsaturated hydrocarbons containing at least one carbon-carbon triple bond.",
        "core_explanation": "The carbon-carbon triple bond contains one sigma bond and two pi bonds. Terminal alkynes also show weak acidic character.",
        "properties": ["Unsaturated hydrocarbons", "Carbon-carbon triple bond", "General formula CₙH₂ₙ₋₂", "Addition reactions", "Terminal alkynes are weakly acidic"],
        "key_points": ["Open-chain alkynes with one triple bond follow the general formula CₙH₂ₙ₋₂.", "The triple bond contains one sigma bond and two pi bonds.", "Ethyne is the simplest alkyne.", "Alkynes undergo addition reactions.", "Only terminal alkynes show acidic character."],
        "examples": ["Ethyne, propyne, and but-2-yne are common examples of alkynes."],
    },
    "aromatics": {
        "title": "Aromatic Hydrocarbons",
        "definition": "Aromatic hydrocarbons contain a stable ring system such as the benzene ring.",
        "core_explanation": "Benzene is the central aromatic compound. Its delocalized electrons make the ring unusually stable, so substitution reactions are more common than addition reactions.",
        "properties": ["Benzene ring", "Delocalized electrons", "Benzene formula C₆H₆", "Resonance stability", "Electrophilic substitution"],
        "key_points": ["Benzene has the molecular formula C₆H₆.", "Its six carbon atoms form a planar ring.", "Delocalized electrons give benzene extra stability.", "Aromatic hydrocarbons usually undergo substitution reactions.", "Toluene is methylbenzene."],
        "examples": ["Benzene, toluene, and naphthalene are common aromatic compounds."],
    },
}

def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value:
        return [value]
    return []


def _short_label(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", _as_text(text)).strip(". ")
    if not cleaned:
        return fallback

    if " is " in cleaned.lower():
        return cleaned.split(" is ", 1)[0][:34].strip() or fallback
    if " are " in cleaned.lower():
        return cleaned.split(" are ", 1)[0][:34].strip() or fallback

    words = cleaned.split()
    return " ".join(words[:4])[:34].strip() or fallback


def _find_exact_concept(section_id: str, topic: Optional[str]) -> Optional[Dict[str, Any]]:
    concept = knowledge_graph.get_concept(section_id)
    if concept:
        return concept

    normalized_topic = re.sub(r"[^a-z0-9]+", "_", (topic or "").strip().lower()).strip("_")
    for candidate in knowledge_graph.concepts.values():
        candidate_title = re.sub(
            r"[^a-z0-9]+",
            "_",
            _as_text(candidate.get("title")).lower(),
        ).strip("_")
        if normalized_topic and candidate_title == normalized_topic:
            return candidate

    return None


def _strip_markdown(value: str) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", value)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^[#>\-\d\.\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _first_sentence(value: str, fallback: str) -> str:
    cleaned = _strip_markdown(value)
    if not cleaned:
        return fallback
    match = re.search(r"(.+?[.!?])\s", cleaned)
    return (match.group(1) if match else cleaned[:220]).strip()


def _extract_formulas(context: str) -> List[str]:
    formula_tokens = re.compile(
        r"(?:[A-Z][a-z]?(?:[\u2080-\u2089\u20990-9\u208a\u208b+\-]*)){2,}"
    )
    general_hydrocarbon_tokens = re.compile(
        r"C(?:n|\u2099)?H(?:n|[\u2080-\u2089\u20990-9\u208a\u208b+\-])+"
    )
    bond_tokens = re.compile(r"C\s*(?:=|\u2261)\s*C")
    reaction_marker = re.compile(r"(?:\u2192|->)")
    general_formulas: List[str] = []
    standalone_formulas: List[str] = []
    reactions: List[str] = []
    bond_formulas: List[str] = []

    def add(bucket: List[str], value: str) -> None:
        cleaned = _strip_markdown(value).replace("\\", "").strip(" -:;,.")
        if cleaned and cleaned not in bucket:
            bucket.append(cleaned)

    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        plain_line = _strip_markdown(line)
        chemical_tokens = formula_tokens.findall(plain_line) + general_hydrocarbon_tokens.findall(plain_line)
        meaningful_tokens = [
            token
            for token in chemical_tokens
            if re.search(r"[\u2080-\u2089\u2099n0-9]", token)
        ]
        for match in general_hydrocarbon_tokens.findall(plain_line):
            if "n" in match or "\u2099" in match:
                add(general_formulas, match)

        if reaction_marker.search(plain_line) and len(plain_line) <= 120:
            left, right = reaction_marker.split(plain_line, maxsplit=1)
            left_has_formula = bool(formula_tokens.search(left) or general_hydrocarbon_tokens.search(left))
            right_has_formula = bool(formula_tokens.search(right) or general_hydrocarbon_tokens.search(right))
            if left_has_formula and right_has_formula:
                add(reactions, plain_line)
                continue

        for match in bond_tokens.findall(plain_line):
            add(bond_formulas, match.replace(" ", ""))

        if "formula" in plain_line.lower():
            for match in meaningful_tokens:
                add(standalone_formulas, match)
        else:
            compact_line = plain_line.replace(" ", "")
            if general_hydrocarbon_tokens.fullmatch(compact_line) or formula_tokens.fullmatch(compact_line):
                if re.search(r"[\u2080-\u2089\u2099n0-9]", compact_line):
                    add(standalone_formulas, compact_line)

        if len(general_formulas) + len(standalone_formulas) + len(reactions) + len(bond_formulas) >= 16:
            break

    formulas: List[str] = []
    for formula in general_formulas[:1] + standalone_formulas + reactions + bond_formulas:
        if formula not in formulas:
            formulas.append(formula)
    return formulas[:5]


def _markdown_to_concept(
    section_id: str,
    topic: Optional[str],
    content_scope: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    result = search_knowledge_base(
        section_id=section_id,
        question=topic or section_id.replace("_", " "),
        max_paragraphs=10,
        max_chars=9000,
        scope=content_scope,
    )
    context = _as_text(result.get("context"))
    if not context or result.get("error"):
        return None

    source_context = context
    source_path = SECTION_FILE_MAP.get(section_id)
    if source_path:
        try:
            with open(source_path, "r", encoding="utf-8") as source:
                source_context = source.read()
        except FileNotFoundError:
            pass

    metadata = HYDROCARBON_ARTIFACT_METADATA.get(section_id)
    if metadata:
        related_lookup = {
            "alkanes": ["alkenes", "alkynes", "aromatics"],
            "alkenes": ["alkanes", "alkynes", "aromatics"],
            "alkynes": ["alkanes", "alkenes", "aromatics"],
            "aromatics": ["alkanes", "alkenes", "alkynes"],
        }
        return {
            "concept_id": section_id,
            **metadata,
            "formulas": _extract_formulas(source_context),
            "common_mistakes": [
                {
                    "mistake": f"Treating {metadata['title']} as a memorization topic only.",
                    "correction": "Connect each formula, property, and reaction to the bonding pattern first.",
                    "frequency": "medium",
                },
                {
                    "mistake": "Confusing similar hydrocarbon families.",
                    "correction": "Check the bond type first, then apply formula, naming, and reaction rules.",
                    "frequency": "high",
                },
            ],
            "related_concepts": [
                {"concept_id": item, "relationship": "related_hydrocarbon"}
                for item in related_lookup[section_id]
            ],
            "prerequisites": [],
            "_artifact_source": result.get("source", "markdown"),
        }

    title = topic or section_id.replace("_", " ").title()
    heading = next(
        (
            _strip_markdown(line)
            for line in context.splitlines()
            if line.strip().startswith("#") and _strip_markdown(line)
        ),
        "",
    )
    if heading:
        title = heading

    bullet_points = []
    for line in context.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            point = _strip_markdown(re.sub(r"^([-*]|\d+\.)\s+", "", stripped))
            if point and len(point) > 8:
                bullet_points.append(point)

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", _strip_markdown(context))
        if len(sentence.strip()) > 24
    ]
    key_points = []
    for item in bullet_points + sentences:
        if item not in key_points:
            key_points.append(item)
        if len(key_points) >= 8:
            break

    formulas = _extract_formulas(source_context)

    examples = [
        sentence
        for sentence in sentences
        if any(marker in sentence.lower() for marker in ["example", "e.g", "for example", "such as"])
    ][:4]

    related_lookup = {
        "alkanes": ["alkenes", "alkynes", "aromatics"],
        "alkenes": ["alkanes", "alkynes", "aromatics"],
        "alkynes": ["alkanes", "alkenes", "aromatics"],
        "aromatics": ["alkanes", "alkenes", "alkynes"],
    }
    related_concepts = [
        {"concept_id": item, "relationship": "related_hydrocarbon"}
        for item in related_lookup.get(section_id, [])
    ]

    common_mistakes = [
        {
            "mistake": f"Reading {title} as a memorization topic only.",
            "correction": "Connect each formula, property, and reaction to the structural feature first.",
            "frequency": "medium",
        },
        {
            "mistake": "Confusing similar hydrocarbon families.",
            "correction": "Check the bond type first, then apply formula, naming, and reaction rules.",
            "frequency": "high",
        },
    ] if section_id in related_lookup else []

    return {
        "concept_id": section_id,
        "title": title,
        "definition": _first_sentence(context, f"{title} is a selected study topic."),
        "core_explanation": _strip_markdown(context)[:900],
        "key_points": key_points,
        "examples": examples,
        "formulas": formulas,
        "properties": [_short_label(point, f"Point {index + 1}") for index, point in enumerate(key_points[:6])],
        "common_mistakes": common_mistakes,
        "related_concepts": related_concepts,
        "prerequisites": [],
        "_artifact_source": result.get("source", "markdown"),
    }


def _relationship_label(value: Any) -> str:
    if isinstance(value, dict):
        return _as_text(value.get("relationship") or "connects")
    return "connects"


def _related_id(value: Any) -> str:
    if isinstance(value, dict):
        return _as_text(value.get("concept_id") or value.get("title") or "related")
    return _as_text(value)


def _title_from_concept_id(concept_id: str) -> str:
    concept = knowledge_graph.get_concept(concept_id)
    if concept:
        return _as_text(concept.get("title")) or concept_id.replace("_", " ").title()
    return concept_id.replace("_", " ").title()


def _build_concept_map(concept: Dict[str, Any]) -> Dict[str, Any]:
    concept_id = _as_text(concept.get("concept_id")) or "concept"
    title = _as_text(concept.get("title")) or _title_from_concept_id(concept_id)
    properties = [_as_text(item) for item in _as_list(concept.get("properties")) if _as_text(item)]
    key_points = [_as_text(item) for item in _as_list(concept.get("key_points")) if _as_text(item)]
    related = _as_list(concept.get("related_concepts"))
    prerequisites = [_as_text(item) for item in _as_list(concept.get("prerequisites")) if _as_text(item)]

    nodes = [
        {
            "id": concept_id,
            "label": title,
            "description": _as_text(concept.get("definition")) or _as_text(concept.get("core_explanation")),
            "kind": "core",
        }
    ]
    edges = []

    source_items = properties[:7] or [_short_label(point, f"Point {index + 1}") for index, point in enumerate(key_points[:5])]
    for index, item in enumerate(source_items):
        node_id = f"{concept_id}-property-{index + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": item,
                "description": key_points[index] if index < len(key_points) else f"Important part of {title}.",
                "kind": "property",
            }
        )
        edges.append({"from": concept_id, "to": node_id, "label": "includes"})

    for index, item in enumerate(related[:3]):
        related_id = _related_id(item)
        node_id = f"{concept_id}-related-{index + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": _title_from_concept_id(related_id),
                "description": f"Connected idea: {_relationship_label(item).replace('_', ' ')}.",
                "kind": "related",
            }
        )
        edges.append({"from": concept_id, "to": node_id, "label": _relationship_label(item).replace("_", " ")})

    for index, item in enumerate(prerequisites[:2]):
        node_id = f"{concept_id}-prerequisite-{index + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": _title_from_concept_id(item),
                "description": "Review this first if the main idea feels confusing.",
                "kind": "prerequisite",
            }
        )
        edges.append({"from": node_id, "to": concept_id, "label": "supports"})

    return {
        "type": "concept_map",
        "title": f"{title} concept map",
        "subtitle": "A quick visual route through the idea.",
        "nodes": nodes,
        "edges": edges,
    }


def _build_flip_cards(concept: Dict[str, Any]) -> Dict[str, Any]:
    title = _as_text(concept.get("title")) or "Topic"
    cards = []
    definition = _as_text(concept.get("definition"))
    if definition:
        cards.append(
            {
                "front": f"What is {title}?",
                "back": definition,
                "tag": "definition",
            }
        )

    for index, point in enumerate(_as_list(concept.get("key_points"))[:6]):
        text = _as_text(point)
        if text:
            cards.append(
                {
                    "front": _short_label(text, f"Key idea {index + 1}"),
                    "back": text,
                    "tag": "key point",
                }
            )

    for example in _as_list(concept.get("examples"))[:2]:
        text = _as_text(example)
        if text:
            cards.append(
                {
                    "front": "Can you connect this to real life?",
                    "back": text,
                    "tag": "example",
                }
            )

    if not cards:
        cards.append(
            {
                "front": f"Core idea of {title}",
                "back": _as_text(concept.get("core_explanation")) or "Review the teacher notes for this topic.",
                "tag": "core",
            }
        )

    return {
        "type": "flip_cards",
        "title": "Tap-to-reveal cards",
        "subtitle": "Fast recall without rereading long notes.",
        "cards": cards[:8],
    }


def _extract_variables(formula: str) -> List[str]:
    if "=" not in formula:
        return []

    left, right = formula.split("=", 1)
    left_terms = {item.lower() for item in re.findall(r"[A-Za-z]+", left)}
    variables = []
    for token in re.findall(r"[A-Za-z]+", right):
        normalized = token.lower()
        if normalized not in left_terms and normalized not in {"and", "or", "by", "per"} and normalized not in variables:
            variables.append(normalized)
    return variables[:4]


def _build_formula_lab(concept: Dict[str, Any]) -> Dict[str, Any]:
    formulas = []
    for item in _as_list(concept.get("formulas"))[:5]:
        expression = _as_text(item)
        if not expression:
            continue
        label = _short_label(expression.split("=", 1)[0] if "=" in expression else expression, "Formula")
        formulas.append(
            {
                "label": label,
                "formula": expression,
                "variables": _extract_variables(expression),
                "hint": "Write units first, substitute values second, calculate last.",
            }
        )

    return {
        "type": "formula_lab",
        "title": "Formula lab",
        "subtitle": "Use formulas actively, not just as text.",
        "formulas": formulas,
        "empty_note": "This topic is concept-heavy, so focus on definitions, examples, and mistakes first.",
    }


def _build_mistake_cards(concept: Dict[str, Any]) -> Dict[str, Any]:
    mistakes = []
    for item in _as_list(concept.get("common_mistakes"))[:5]:
        if isinstance(item, dict):
            mistake = _as_text(item.get("mistake"))
            correction = _as_text(item.get("correction"))
            frequency = _as_text(item.get("frequency")) or "medium"
        else:
            mistake = _as_text(item)
            correction = "Compare it with the key points before answering."
            frequency = "medium"

        if mistake:
            mistakes.append(
                {
                    "mistake": mistake,
                    "correction": correction,
                    "frequency": frequency,
                }
            )

    return {
        "type": "mistake_cards",
        "title": "Mistake shield",
        "subtitle": "Common traps students should avoid.",
        "mistakes": mistakes,
        "empty_note": "No common mistakes are listed for this topic yet.",
    }


def available_artifact_sections() -> List[str]:
    return sorted(set(SECTION_FILE_MAP) | set(knowledge_graph.concepts))


def generate_study_artifacts(
    section_id: str,
    topic: Optional[str] = None,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    content_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_id = re.sub(r"[^a-z0-9]+", "_", (section_id or "").strip().lower()).strip("_")
    if not normalized_id:
        raise ValueError("section_id is required.")

    concept = _find_exact_concept(normalized_id, topic)
    if not concept and (normalized_id in SECTION_FILE_MAP or content_scope):
        concept = _markdown_to_concept(normalized_id, topic, content_scope)
    if not concept:
        raise LookupError(ARTIFACT_DATA_NOT_AVAILABLE)

    title = _as_text(concept.get("title")) or _title_from_concept_id(normalized_id)
    key_points = [_as_text(item) for item in _as_list(concept.get("key_points")) if _as_text(item)]
    formulas = [_as_text(item) for item in _as_list(concept.get("formulas")) if _as_text(item)]
    mistakes = _as_list(concept.get("common_mistakes"))

    return {
        "available": True,
        "source": _as_text(concept.get("_artifact_source")) or "knowledge_graph",
        "section_id": _as_text(concept.get("concept_id")) or normalized_id,
        "subject": _as_text(subject) or "Chemistry",
        "chapter": _as_text(chapter),
        "topic": _as_text(topic) or title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "subtitle": _as_text(concept.get("definition")) or _as_text(concept.get("core_explanation")),
        "student_goal": f"Understand {title}, recall the key points, and avoid common exam mistakes.",
        "quality": {
            "key_points": len(key_points),
            "formulas": len(formulas),
            "mistakes": len(mistakes),
        },
        "artifacts": [
            _build_concept_map(concept),
            _build_flip_cards(concept),
            _build_formula_lab(concept),
            _build_mistake_cards(concept),
        ],
    }
