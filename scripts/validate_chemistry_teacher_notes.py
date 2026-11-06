"""Validate deployable NCERT teaching content without a DB or model service.

Run from the backend root: python scripts/validate_chemistry_teacher_notes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.teacher_notes_service import ROOT, teacher_curricula, _load_lesson, compile_lesson, _topic_aliases, resolve_teacher_topics


def validate():
    curricula = teacher_curricula()
    page_ids, unit_ids = set(), set()
    report = []
    for curriculum in curricula:
        directory = ROOT / curriculum.get("lesson_directory", "data/notes/teacher_chapter1")
        expected = {f"unit_{i}.json" for i in range(len(curriculum["units"]))}
        if {p.name for p in directory.glob("unit_*.json")} != expected:
            raise ValueError(f"Missing or unexpected lessons in {directory}")
        words = 0
        for unit in curriculum["units"]:
            if unit["id"] in unit_ids:
                raise ValueError("Duplicate teaching unit identity")
            unit_ids.add(unit["id"])
            original = _load_lesson(unit["id"])
            for page in original["pages"]:
                if page["id"] in page_ids:
                    raise ValueError("Duplicate teaching page identity")
                page_ids.add(page["id"])
                words += sum(len(block["text"].split()) for block in page["blocks"])
            for proficiency in ["new_to_it", "know_a_little", "know_the_basics", "mostly_confident"]:
                for style in ["balanced", "concise", "detailed"]:
                    pages = compile_lesson(curriculum,unit,proficiency=proficiency,style=style)
                    if any(not page["source_pages"] for page in pages):
                        raise ValueError("A teaching page has no verifiable source-page mapping")
        report.append({"chapter":curriculum["chapter_number"],"units":len(curriculum["units"]),"pages":len(curriculum["units"])*10,"teaching_words":words})
    aliases = _topic_aliases()
    alias_count = 0
    for curriculum in curricula:
        for alias, candidates in aliases[curriculum['curriculum_key']]['aliases'].items():
            resolved = resolve_teacher_topics(curriculum['chapter_slug'],alias)
            if not resolved or not resolved[1] or not {u['id'] for u in resolved[1]} <= set(candidates):
                raise ValueError('A notes link has no scoped teaching destination')
            alias_count += 1
    return {"chapters":report,"total_units":len(unit_ids),"total_pages":len(page_ids),"adaptations_per_unit":12,"verified_topic_aliases":alias_count}


if __name__ == "__main__":
    print(json.dumps(validate(),indent=2))
