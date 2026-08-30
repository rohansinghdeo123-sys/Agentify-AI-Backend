import os
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base
from Logic import embeddings
from Logic.content_pipeline import embed_missing_chunks, search_approved_content
from models import ContentChapter, ContentChunk


class FakeEmbeddingResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class EmbeddingClientTests(unittest.TestCase):
    def setUp(self):
        embeddings.clear_query_cache()

    def tearDown(self):
        embeddings.clear_query_cache()

    def test_similarity_math(self):
        self.assertAlmostEqual(embeddings.similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(embeddings.similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(embeddings.similarity(None, [1.0]), 0.0)
        self.assertEqual(embeddings.similarity([1.0], [1.0, 0.0]), 0.0)

    def test_embedding_contract_rejects_cross_model_vectors(self):
        with patch("Logic.embeddings.embedding_model", return_value="model-a"), patch(
            "Logic.embeddings.embedding_endpoint_host", return_value="provider-a.example"
        ):
            self.assertTrue(
                embeddings.compatible_with_stored(
                    [1.0, 0.0],
                    [1.0, 0.0],
                    stored_model="model-a",
                    stored_endpoint_host="provider-a.example",
                )
            )
            self.assertFalse(
                embeddings.compatible_with_stored(
                    [1.0, 0.0],
                    [1.0, 0.0],
                    stored_model="model-b",
                )
            )
            self.assertFalse(
                embeddings.compatible_with_stored(
                    [1.0, 0.0],
                    [1.0, 0.0],
                    stored_model="model-a",
                    stored_endpoint_host="provider-b.example",
                )
            )

    def test_legacy_google_model_setting_uses_release_vector_space(self):
        with patch.dict(
            os.environ,
            {
                "EMBEDDINGS_MODEL": "text-embedding-004",
                "EMBEDDINGS_BASE_URL": (
                    "https://generativelanguage.googleapis.com/v1beta/openai"
                ),
            },
            clear=False,
        ):
            self.assertEqual(embeddings.embedding_model(), "gemini-embedding-001")

        with patch.dict(
            os.environ,
            {
                "EMBEDDINGS_MODEL": "text-embedding-004",
                "EMBEDDINGS_BASE_URL": "https://embeddings.example/v1",
            },
            clear=False,
        ):
            self.assertEqual(embeddings.embedding_model(), "text-embedding-004")

    def test_normalize_returns_unit_vector(self):
        normalized = embeddings.normalize([3.0, 4.0])
        self.assertAlmostEqual(sum(value * value for value in normalized), 1.0)

    def test_disabled_without_api_key(self):
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "", "OPENAI_API_KEY": ""}):
            self.assertFalse(embeddings.embeddings_enabled())
            self.assertIsNone(embeddings.embed_query("what is matter"))
            with self.assertRaises(RuntimeError):
                embeddings.embed_texts(["text"])

    def test_embed_texts_parses_and_normalizes_in_order(self):
        payload = {
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ]
        }

        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "test-key"}), patch(
            "Logic.embeddings.requests.post", return_value=FakeEmbeddingResponse(payload)
        ) as post:
            vectors = embeddings.embed_texts(["first", "second"])

        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(vectors[0][0], 0.6)  # [3,4] normalized, index 0 first
        self.assertAlmostEqual(vectors[1][1], 1.0)  # [0,2] normalized
        request_json = post.call_args.kwargs["json"]
        self.assertEqual(request_json["input"], ["first", "second"])

    def test_embed_query_caches_repeated_questions(self):
        payload = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "test-key"}), patch(
            "Logic.embeddings.requests.post", return_value=FakeEmbeddingResponse(payload)
        ) as post:
            first = embeddings.embed_query("What is an alkane?")
            second = embeddings.embed_query("  what is an ALKANE? ")

        self.assertEqual(first, second)
        self.assertEqual(post.call_count, 1)

    def test_embed_query_swallows_api_failure(self):
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "test-key"}), patch(
            "Logic.embeddings.requests.post", side_effect=RuntimeError("api down")
        ):
            self.assertIsNone(embeddings.embed_query("question"))


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        embeddings.clear_query_cache()
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        self.SessionTesting = sessionmaker(bind=engine)
        self.db = self.SessionTesting()

        chapter = ContentChapter(
            board="NCERT",
            class_level="11",
            subject="Chemistry",
            chapter_name="Hydrocarbons",
            slug="ncert_class_11_chemistry_hydrocarbons",
            status="approved",
        )
        self.db.add(chapter)
        self.db.flush()
        self.chapter_id = chapter.id

        # Semantically relevant to "how do plants store energy" but shares no
        # query keywords; embedding points along the first axis.
        self.db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="semantic_chunk",
                text="Photosynthesis converts light into chemical fuel inside chloroplasts.",
                page_start=3,
                page_end=3,
                lexical_terms=["photosynthesis", "chloroplasts", "fuel"],
                embedding=[1.0, 0.0, 0.0],
            )
        )
        # Lexically matching but semantically orthogonal chunk.
        self.db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="lexical_chunk",
                text="Plants need water. Plants need soil. Plants grow slowly over months.",
                page_start=7,
                page_end=7,
                lexical_terms=["plants", "water", "soil"],
                embedding=[0.0, 1.0, 0.0],
            )
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()
        embeddings.clear_query_cache()

    def _search(self, question, query_vector):
        with patch("Logic.content_pipeline.SessionLocal", self.SessionTesting), patch(
            "Logic.embeddings.embed_query", return_value=query_vector
        ):
            return search_approved_content("biology_basics", question)

    def test_semantic_match_without_keyword_overlap(self):
        result = self._search("how does a leaf turn sunshine into usable power", [1.0, 0.0, 0.0])

        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertGreaterEqual(result["semantic_matches"], 1)
        self.assertIn("Photosynthesis converts light", result["context"])

    def test_lexical_fallback_when_embeddings_unavailable(self):
        result = self._search("plants water soil", None)

        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertEqual(result["semantic_matches"], 0)
        self.assertIn("Plants need water", result["context"])
        self.assertNotIn("Photosynthesis converts light", result["context"])

    def test_fusion_prefers_candidate_with_both_signals(self):
        # Query lexically matches the semantic chunk too ("photosynthesis"),
        # so that chunk holds both signals and must outrank the lexical-only one.
        result = self._search("photosynthesis plants", [1.0, 0.0, 0.0])

        self.assertEqual(result["retrieval_mode"], "hybrid")
        first_block = result["context"].split("\n\n")[0]
        self.assertIn("Photosynthesis converts light", result["context"])
        self.assertIn("type: chunk", first_block)
        self.assertEqual(result["matched_sections"][0], "semantic_chunk")

    def test_low_similarity_is_ignored(self):
        # Query vector nearly orthogonal to both chunks: no semantic matches.
        result = self._search("plants water soil", [0.05, 0.05, 0.99])

        self.assertEqual(result["semantic_matches"], 0)
        self.assertIn("Plants need water", result["context"])

    def test_model_mismatch_safely_falls_back_to_lexical(self):
        for chunk in self.db.query(ContentChunk).all():
            chunk.metadata_json = {"embedding_model": "different-model"}
        self.db.commit()
        with patch("Logic.embeddings.embedding_model", return_value="configured-model"):
            result = self._search("plants water soil", [1.0, 0.0, 0.0])

        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertEqual(result["semantic_matches"], 0)
        self.assertIn("Plants need water", result["context"])

    def test_endpoint_mismatch_safely_falls_back_to_lexical(self):
        for chunk in self.db.query(ContentChunk).all():
            chunk.metadata_json = {
                "embedding_model": "shared-model-name",
                "embedding_endpoint_host": "stored-provider.example",
            }
        self.db.commit()
        with patch("Logic.embeddings.embedding_model", return_value="shared-model-name"), patch(
            "Logic.embeddings.embedding_endpoint_host",
            return_value="configured-provider.example",
        ):
            result = self._search("plants water soil", [1.0, 0.0, 0.0])

        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertEqual(result["semantic_matches"], 0)
        self.assertIn("Plants need water", result["context"])


class EmbedBackfillTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        self.db = sessionmaker(bind=engine)()

        chapter = ContentChapter(slug="backfill_chapter", status="approved")
        self.db.add(chapter)
        self.db.flush()
        self.chapter_id = chapter.id
        self.db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="missing_embedding",
                text="Alkanes are saturated hydrocarbons.",
                metadata_json={"source": "pdf"},
            )
        )
        self.db.add(
            ContentChunk(
                chapter_id=chapter.id,
                chunk_id="already_embedded",
                text="Alkenes contain a double bond.",
                embedding=[0.0, 1.0],
            )
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_backfill_skips_when_disabled(self):
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "", "OPENAI_API_KEY": ""}):
            result = embed_missing_chunks(self.db, chapter_id=self.chapter_id)

        self.assertFalse(result["enabled"])
        self.assertEqual(result["embedded"], 0)
        self.assertEqual(result["missing"], 1)

    def test_backfill_embeds_only_missing_chunks(self):
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "test-key"}), patch(
            "Logic.embeddings.embed_texts", return_value=[[1.0, 0.0]]
        ) as embed:
            result = embed_missing_chunks(self.db, chapter_id=self.chapter_id)

        self.assertTrue(result["enabled"])
        self.assertEqual(result["embedded"], 1)
        embed.assert_called_once_with(["Alkanes are saturated hydrocarbons."])

        backfilled = (
            self.db.query(ContentChunk).filter(ContentChunk.chunk_id == "missing_embedding").one()
        )
        self.assertEqual(backfilled.embedding, [1.0, 0.0])
        self.assertIn("embedding_model", backfilled.metadata_json)
        untouched = (
            self.db.query(ContentChunk).filter(ContentChunk.chunk_id == "already_embedded").one()
        )
        self.assertEqual(untouched.embedding, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
