"""Build the source-bound Chemistry notes alias index without a DB/model call.

Run from the backend root:
    python scripts/build_notes_topic_index.py
    python scripts/build_notes_topic_index.py --check

All alias ownership is a set, ordered by the existing NCERT teaching units.
An old source concept/group can span several units; this index preserves that
ambiguity for the reader instead of selecting an arbitrary teaching topic.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Logic.planning.curriculum_registry import load_planning_curriculum
from services.catalog_service import _learning_units_for_chapter
from services.content_release_bundle import _verified_payload
from services.master_notes_service import CHAPTER_ONE_SEED_GROUPS

BUNDLE = ROOT / "data/releases/ncert_class_11_chemistry_v1.json.gz"
OUTPUT = ROOT / "data/notes/teacher_chemistry/topic_aliases.json"


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _curricula():
    return [load_planning_curriculum(), *json.loads(
        (ROOT / "data/notes/teacher_chemistry/curricula.json").read_text(encoding="utf-8"))]


def _source_owners(number, concepts, groups, curriculum):
    """Reviewed bridges for pre-existing published source concepts.

    Chapter 1's excerpt map records real concept/page ownership. Its final
    mixed exercise concept was intentionally excluded from that excerpt map;
    the reviewed questions concern measurement, masses, composition and
    stoichiometry, so they retain those four candidate units.

    Chapter 2's first eight source concepts follow the canonical topic
    sequence. Its two final reviews deliberately span several topics: the
    first covers charge, radiation, Bohr, wave mechanics and orbitals/filling;
    the last covers photons/waves and quantum-number/filling applications.
    The full candidate sets below prevent either review being misrepresented
    as one narrow teaching unit. Chapters 3–9 already share published grouping.
    """
    units = curriculum["units"]
    if number == 1:
        excerpt = json.loads((ROOT / "data/notes/chapter_1_notes_mapping.json").read_text(encoding="utf-8"))
        owners = {}
        for unit in excerpt["units"]:
            for page in unit["pages"]:
                owners.setdefault(page["source_concept_id"], set()).add(unit["unit_id"])
        review_id = "unit_ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry_fc4070d4c3"
        owners[review_id] = {units[i]["id"] for i in [1, 3, 4, 5]}
        return owners
    if number == 2:
        expected = [
            "9ee5550ee4", "ca2ddd7084", "eda6b97af2", "5c8b06c211", "c89a1af440",
            "5806864edd", "2be9ace09d", "54591cc7e1", "313c48a951", "6a2e831e11",
        ]
        if [c["concept_id"].rsplit("_", 1)[-1] for c in concepts] != expected:
            raise ValueError("Chapter 2 source identities/order changed; review the alias ownership bridge")
        indexes = [[0], [1], [2], [2], [3], [4], [5], [6], [0, 2, 3, 4, 5, 6], [2, 4, 5, 6]]
        return {concept["concept_id"]: {units[i]["id"] for i in positions}
                for concept, positions in zip(concepts, indexes)}
    owners = {}
    for group in groups:
        for concept in group["concepts"]:
            owners.setdefault(concept.concept_id, set()).add(group["id"])
    return owners


def build_index(bundle_path=BUNDLE):
    payload = _verified_payload(Path(bundle_path))
    curricula = _curricula()
    by_number = {c["chapter_number"]: c for c in curricula}
    if sorted(by_number) != list(range(1, 10)):
        raise ValueError("Expected the nine reviewed Class XI Chemistry curricula")
    output = {"schema_version": 1, "release_digest": payload["digest"], "chapters": {}}
    totals = {"chapters": 0, "units": 0, "source_concepts": 0, "catalog_topics": 0,
              "aliases": 0, "ambiguous_aliases": 0}
    seen_chapters = set()
    for raw in payload["chapters"]:
        chapter = raw["chapter"]
        number = chapter["chapter_number"]
        if number not in by_number or number in seen_chapters:
            raise ValueError("Release chapter is missing, duplicated or outside the reviewed curricula")
        if normalize(chapter["subject"]) != "chemistry" or normalize(chapter["class_level"]) not in {"11", "class_11"}:
            raise ValueError("Release chapter has an unexpected syllabus scope")
        seen_chapters.add(number)
        curriculum = by_number[number]
        units = curriculum["units"]
        unit_order = [u["id"] for u in units]
        valid_units = set(unit_order)
        if len(unit_order) != len(valid_units):
            raise ValueError("Duplicate canonical unit identity")
        aliases = {}

        def add(alias, owners):
            key = normalize(alias)
            if not key:
                return
            candidates = set(owners)
            if not candidates or not candidates <= valid_units:
                raise ValueError(f"Invalid/cross-chapter ownership for alias {key}")
            aliases.setdefault(key, set()).update(candidates)

        for unit in units:
            own = [unit["id"]]
            for identity in [unit["id"], unit["primary_topic_id"], unit["title"], *unit.get("legacy_topic_ids", [])]:
                add(identity, own)
            for field in ["concepts", "ncert_subtopics"]:
                for item in unit[field]:
                    add(item["id"], own)
                    add(item["title"], own)

        if number == 1:
            for alias, position in CHAPTER_ONE_SEED_GROUPS.items():
                add(alias, [units[position]['id']])

        concepts = raw["concepts"]
        namespace = SimpleNamespace(**chapter)
        namespace.id = number
        groups = _learning_units_for_chapter(namespace, [SimpleNamespace(**c) for c in concepts])
        owners = _source_owners(number, concepts, groups, curriculum)
        for concept in concepts:
            concept_id = concept["concept_id"]
            if concept_id not in owners:
                raise ValueError(f"A source concept has no reviewed ownership: {concept_id}")
            add(concept_id, owners[concept_id])
            add(concept["title"], owners[concept_id])
            for alias in (concept.get("raw_json") or {}).get("source_concept_ids") or []:
                add(alias, owners[concept_id])
        for group in groups:
            candidates = set().union(*(owners[c.concept_id] for c in group["concepts"]))
            add(group["id"], candidates)
            add(group["label"], candidates)

        indexed = {key: [unit_id for unit_id in unit_order if unit_id in values]
                   for key, values in sorted(aliases.items())}
        # Canonical IDs are guaranteed discoverable. Runtime direct ownership
        # takes precedence over any broad source alias with the same spelling.
        for unit_id in unit_order:
            if unit_id not in indexed[normalize(unit_id)]:
                raise ValueError("A canonical teaching unit is absent from the alias index")
        for concept in concepts:
            if not indexed.get(normalize(concept["concept_id"])):
                raise ValueError("A published source concept is absent from the alias index")
        for group in groups:
            if not indexed.get(normalize(group["id"])):
                raise ValueError("A published catalog topic is absent from the alias index")
        output["chapters"][curriculum["curriculum_key"]] = {"aliases": indexed}
        totals["chapters"] += 1
        totals["units"] += len(units)
        totals["source_concepts"] += len(concepts)
        totals["catalog_topics"] += len(groups)
        totals["aliases"] += len(indexed)
        totals["ambiguous_aliases"] += sum(len(values) > 1 for values in indexed.values())
    if seen_chapters != set(by_number) or (totals["units"], totals["source_concepts"], totals["catalog_topics"]) != (72, 83, 76):
        raise ValueError("Reviewed source/catalog coverage changed; audit the new corpus before rebuilding")
    return output, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the checked-in index is stale")
    args = parser.parse_args()
    payload, totals = build_index()
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit("Notes topic index is stale; run scripts/build_notes_topic_index.py")
    else:
        OUTPUT.write_text(content, encoding="utf-8")
    print(json.dumps({**totals, "bytes": len(content.encode("utf-8")), "checked": args.check}, sort_keys=True))


if __name__ == "__main__":
    main()
