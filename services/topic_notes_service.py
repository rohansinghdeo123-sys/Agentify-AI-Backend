"""NCERT teaching-notes adapter retaining canonical and archived Planning links."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def chapter_one_mapping():
    return json.loads((ROOT / 'data/notes/chapter_1_notes_mapping.json').read_text(encoding='utf-8'))


def mapped_notes(chapter_slug, topic_id, db, user_id, *, subject=None, class_level=None, content_version=None):
    """One universal notes entry point; archived source mappings stay read-only."""
    from services.adaptive_notes_service import notes_response
    return notes_response(db,chapter_slug,topic_id,user_id,subject=subject,class_level=class_level,content_version=content_version)
