"""Persist/index Amit's verified PDF; optionally generate real provider embeddings."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()
from Logic import embeddings
from services.master_notes_service import ROOT, artifact, install_master_notes
from database import SessionLocal

def main():
    data=artifact()
    if "--embed" in sys.argv:
        vectors=embeddings.embed_texts([s["text"] for s in data["sections"]])
        for section,vector in zip(data["sections"],vectors):
            section.update(embedding=vector,embedding_model=embeddings.embedding_model(),
                embedding_endpoint_host=embeddings.embedding_endpoint_host(),embedding_dimensions=len(vector))
        (ROOT / "data/notes/amit_chapter1_sections.json").write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    with SessionLocal() as db:
        chapter=install_master_notes(db)
        print("Persisted master sections:",chapter.chunk_count)
if __name__ == "__main__":main()
