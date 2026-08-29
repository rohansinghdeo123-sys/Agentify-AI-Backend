import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CONTENT_GENERATION_BATCH_DELAY_SECONDS", "0")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from pypdf.generic import DecodedStreamObject

from database import Base
from Logic.content_pipeline import (
    CONTENT_GENERATION_MIN_OUTPUT_TOKENS,
    CONTENT_GENERATION_REQUEST_TOKEN_BUDGET,
    _extract_json_array,
    _default_generation_batch_delay,
    _estimate_generation_input_tokens,
    _extracted_text_sanity,
    _generation_batch_is_reference_only,
    _has_defective_legacy_bookman_cmap,
    _is_full_legacy_bookman_font,
    _legacy_bookman_cmap,
    _normalize_pdf_formula_glyphs,
    _normalize_generated_batch_items,
    _prefer_formula_fallback,
    build_coverage_report,
    chunk_pages,
    embed_missing_chunks,
    generate_concepts_for_chapter,
    infer_metadata_from_pdf_path,
    ingest_pdf_file,
    search_approved_content,
    validate_concept_payloads,
    approve_chapter,
)
from models import ContentChapter, ContentChunk, ContentConcept, ContentPage


class ContentPipelineTests(unittest.TestCase):
    def test_generation_pacing_defaults_to_safe_groq_window(self):
        self.assertEqual(
            _default_generation_batch_delay(
                {"provider_order": "groq", "routes": [{"provider": "groq"}]}
            ),
            61.0,
        )
        self.assertEqual(
            _default_generation_batch_delay(
                {"provider_order": "openai", "routes": [{"provider": "openai"}]}
            ),
            0.0,
        )

    def _session_factory(self):
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        return sessionmaker(bind=engine)

    def test_infer_metadata_from_expected_ncert_path(self):
        root = Path("C:/repo/backend/data/raw/ncert")
        pdf = root / "class_11" / "chemistry" / "chapter_01_some_basic_concepts_of_chemistry.pdf"

        metadata = infer_metadata_from_pdf_path(pdf, root)

        self.assertEqual(metadata["board"], "NCERT")
        self.assertEqual(metadata["class_level"], "11")
        self.assertEqual(metadata["subject"], "Chemistry")
        self.assertEqual(metadata["chapter_number"], 1)
        self.assertEqual(metadata["chapter_name"], "Some Basic Concepts Of Chemistry")
        self.assertIn("class_11", metadata["slug"])

    def test_structural_gate_accepts_only_full_identity_h_bookman_font(self):
        full_file = DecodedStreamObject()
        full_file.set_data(b"f" * 16_000)
        sparse_file = DecodedStreamObject()
        sparse_file.set_data(b"f" * 4_200)

        def font(font_file, *, base="/ABCDEF+Bookman-Light", subtype="/Type0"):
            return {
                "/BaseFont": base,
                "/Subtype": subtype,
                "/Encoding": "/Identity-H",
                "/DescendantFonts": [{
                    "/Subtype": "/CIDFontType2",
                    "/CIDToGIDMap": "/Identity",
                    "/FontDescriptor": {"/FontFile2": font_file},
                }],
            }

        self.assertTrue(_is_full_legacy_bookman_font(font(full_file)))
        self.assertFalse(_is_full_legacy_bookman_font(font(sparse_file)))
        self.assertFalse(_is_full_legacy_bookman_font(
            font(full_file, base="/ABCDEF+Bookman-Light-SC700")
        ))
        self.assertFalse(_is_full_legacy_bookman_font(font(full_file, subtype="/Type1")))

    def test_legacy_bookman_cmap_contains_ascii_dashes_and_bullet(self):
        cmap = _legacy_bookman_cmap().get_data().decode("ascii")
        self.assertIn("<0003> <0061> <0020>", cmap)
        self.assertIn("<0103> <2013>", cmap)
        self.assertIn("<0106> <2022>", cmap)

    def test_bookman_repair_gate_preserves_valid_unicode_map(self):
        full_file = DecodedStreamObject()
        full_file.set_data(b"f" * 16_000)

        def font(cmap):
            return {
                "/BaseFont": "/ABCDEF+Bookman-Light",
                "/Subtype": "/Type0",
                "/Encoding": "/Identity-H",
                "/ToUnicode": cmap,
                "/DescendantFonts": [{
                    "/Subtype": "/CIDFontType2",
                    "/CIDToGIDMap": "/Identity",
                    "/FontDescriptor": {"/FontFile2": full_file},
                }],
            }

        defective = DecodedStreamObject()
        defective.set_data(b"1 beginbfchar\n<0024> <0061>\nendbfchar\n")

        self.assertFalse(_has_defective_legacy_bookman_cmap(font(_legacy_bookman_cmap())))
        self.assertTrue(_has_defective_legacy_bookman_cmap(font(defective)))

    def test_text_sanity_flags_cipher_but_not_normal_chemistry_prose(self):
        encoded_line = (
            "aIWHU VWXG\\LQJ WKLV XQLW\x0f \\RX ZLOO EH DEOH WR "
            "XQGHUVWDQG WKH RUJDQLF FKHPLVWU\\ FRQFHSWV\x11"
        )
        encoded_corpus = "\n".join([encoded_line] * 12)
        normal = (
            "After studying this unit, students understand organic compounds, "
            "their structures, reactions, and applications. CH3-CH2-OH is ethanol. "
        ) * 12
        self.assertTrue(_extracted_text_sanity(encoded_corpus)["suspected_encoding_corruption"])
        self.assertFalse(_extracted_text_sanity(normal)["suspected_encoding_corruption"])
        self.assertTrue(
            _extracted_text_sanity("Enthalpy is /unif0448 H for a process.")[
                "suspected_formula_glyph_corruption"
            ]
        )
        cid_sanity = _extracted_text_sanity(
            "(cid:83)(cid:72)(cid:85)(cid:70)(cid:72)(cid:81)(cid:87)"
        )
        self.assertEqual(cid_sanity["cid_placeholder_count"], 7)
        self.assertTrue(cid_sanity["suspected_encoding_corruption"])

    def test_formula_fallback_requires_fewer_aliases_and_meaningful_text(self):
        primary = "Thermodynamics /unif0448 H equation " * 20
        readable = "Thermodynamics delta H equation and explanation. " * 10
        too_short = "delta H"

        self.assertTrue(_prefer_formula_fallback(primary, readable))
        self.assertFalse(_prefer_formula_fallback(primary, too_short))
        self.assertFalse(_prefer_formula_fallback(readable, primary))
        cid_fallback = " ".join("(cid:83)" for _ in range(200))
        self.assertFalse(_prefer_formula_fallback(primary, cid_fallback))

    def test_private_symbol_formula_glyphs_are_decoded_or_blocked(self):
        raw = "\uf044H \uf03d q \uf02b w; 5\uf0b4T; \uf103 and \uf106; \uf0e9x\uf0f9"

        normalized, changed = _normalize_pdf_formula_glyphs(raw)

        self.assertEqual(normalized, "ΔH = q + w; 5×T; α and σ; [x]")
        self.assertEqual(changed, 8)
        self.assertFalse(
            _extracted_text_sanity(normalized)["suspected_formula_glyph_corruption"]
        )
        self.assertTrue(
            _extracted_text_sanity("unresolved \uf123 glyph")[
                "suspected_formula_glyph_corruption"
            ]
        )

        arrow, arrow_changes = _normalize_pdf_formula_glyphs(
            "Compound heat/unif0be/unif0ae/unif0be/unif0be O2"
        )
        self.assertEqual(" ".join(arrow.split()), "Compound heat → O2")
        self.assertEqual(arrow_changes, 4)
        self.assertFalse(_extracted_text_sanity(arrow)["suspected_encoding_corruption"])

    def test_generation_json_requires_complete_array_and_explicit_citations(self):
        with self.assertRaises(Exception):
            _extract_json_array('[{"concept_id":"valid"}] trailing text')
        with self.assertRaisesRegex(ValueError, "cite at least one"):
            _normalize_generated_batch_items(
                [{"concept_id": "missing_source", "title": "Missing source"}],
                [1, 2],
            )

    def test_only_logarithm_appendix_batches_are_skipped(self):
        logarithms = ContentPage(
            page_number=48,
            text="Logarithms\nN 0 1 2 3 4 5 6 7 8 9\n" + "1234 " * 500,
        )
        continuation = ContentPage(
            page_number=49,
            text="Table II (Continued)\n" + "1234567890 " * 500,
        )
        notes = ContentPage(
            page_number=50,
            text="Notes\nIndex.indd 239\nReprint 2026-27",
        )
        equilibrium = ContentPage(
            page_number=20,
            text="Chemical equilibrium is dynamic and follows the equilibrium law.",
        )

        self.assertTrue(
            _generation_batch_is_reference_only([logarithms, continuation, notes])
        )
        self.assertFalse(_generation_batch_is_reference_only([equilibrium]))
        self.assertFalse(
            _generation_batch_is_reference_only([logarithms, equilibrium])
        )

    def test_reingest_reuses_curriculum_identity_when_filename_slug_changes(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            existing = ContentChapter(
                board="NCERT",
                class_level="11",
                subject="Chemistry",
                chapter_number=3,
                chapter_name="Chapter 3",
                slug="ncert_class_11_chemistry_chapter_3_chapter_3",
                status="indexed",
            )
            db.add(existing)
            db.commit()
            existing_id = existing.id

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "ncert"
                chapter_dir = root / "class_11" / "chemistry"
                chapter_dir.mkdir(parents=True)
                pdf = chapter_dir / "chapter_03_classification_of_elements.pdf"
                pdf.write_bytes(b"%PDF-1.4\nidentity-test")
                page = {
                    "page_number": 1,
                    "text": "Periodic classification organises elements by recurring properties.",
                    "char_count": 67,
                    "extraction_quality": 0.8,
                    "text_sanity": {"suspected_encoding_corruption": False},
                    "decoded_fonts": [],
                }
                with patch("Logic.content_pipeline.extract_pdf_pages", return_value=[page]), \
                     patch("Logic.content_pipeline.embeddings_service.embeddings_enabled", return_value=False):
                    ingested = ingest_pdf_file(db, pdf, root_path=root)
                    db.commit()

            self.assertEqual(ingested.id, existing_id)
            self.assertEqual(
                db.query(ContentChapter).filter(ContentChapter.chapter_number == 3).count(),
                1,
            )
            self.assertIn("classification_of_elements", ingested.slug)
        finally:
            db.close()

    def test_incomplete_generation_never_replaces_existing_concepts(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(
                board="NCERT", class_level="11", subject="Chemistry",
                chapter_number=3, chapter_name="Classification", slug="classification",
                status="indexed",
            )
            db.add(chapter)
            db.flush()
            db.add_all([
                ContentPage(chapter_id=chapter.id, page_number=1, text="A" * 90, char_count=90),
                ContentPage(chapter_id=chapter.id, page_number=2, text="B" * 90, char_count=90),
                ContentConcept(
                    chapter_id=chapter.id, concept_id="existing", title="Existing",
                    definition="Previously validated content", source_pages=[1],
                ),
            ])
            db.commit()
            responses = [
                '[{"concept_id":"periodicity","title":"Periodicity","definition":"d","source_pages":[1]}]',
                "not-json",
            ]
            with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
                os.environ,
                {"CONTENT_GENERATION_CACHE_DIR": cache_dir},
            ), patch(
                "Logic.coach.model_gateway.model_gateway.complete",
                side_effect=responses,
            ):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    generate_concepts_for_chapter(db, chapter.id, max_batch_chars=100)

            self.assertEqual(
                [row.concept_id for row in db.query(ContentConcept).filter_by(chapter_id=chapter.id).all()],
                ["existing"],
            )
        finally:
            db.close()

    def test_generation_retry_reuses_only_exact_successful_batch_checkpoints(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(
                board="NCERT", class_level="11", subject="Chemistry",
                chapter_number=5, chapter_name="Thermodynamics", slug="thermodynamics",
                source_hash="source-hash-v1", status="indexed",
            )
            db.add(chapter)
            db.flush()
            db.add_all([
                ContentPage(chapter_id=chapter.id, page_number=1, text="A" * 90, char_count=90),
                ContentPage(chapter_id=chapter.id, page_number=2, text="B" * 90, char_count=90),
            ])
            db.commit()
            first_batch = (
                '[{"concept_id":"system","title":"System",'
                '"definition":"First batch","source_pages":[1]}]'
            )
            second_batch = (
                '[{"concept_id":"enthalpy","title":"Enthalpy",'
                '"definition":"Second batch","source_pages":[2]}]'
            )

            with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
                os.environ,
                {"CONTENT_GENERATION_CACHE_DIR": cache_dir},
            ):
                with patch(
                    "Logic.coach.model_gateway.model_gateway.complete",
                    side_effect=[first_batch, RuntimeError("provider quota")],
                ) as interrupted, patch(
                    "Logic.coach.model_gateway.model_gateway.records",
                    return_value=[{
                        "status": "success",
                        "provider": "groq",
                        "model": "openai/gpt-oss-120b",
                    }],
                ):
                    with self.assertRaisesRegex(RuntimeError, "provider quota"):
                        generate_concepts_for_chapter(db, chapter.id, max_batch_chars=100)
                self.assertEqual(interrupted.call_count, 2)
                self.assertEqual(db.query(ContentConcept).filter_by(chapter_id=chapter.id).count(), 0)
                self.assertEqual(len(list(Path(cache_dir).rglob("*.json"))), 1)

                with patch(
                    "Logic.coach.model_gateway.model_gateway.complete",
                    return_value=second_batch,
                ) as resumed, patch(
                    "Logic.coach.model_gateway.model_gateway.records",
                    return_value=[{
                        "status": "success",
                        "provider": "groq",
                        "model": "openai/gpt-oss-120b",
                    }],
                ):
                    generated = generate_concepts_for_chapter(
                        db,
                        chapter.id,
                        max_batch_chars=100,
                    )

                self.assertEqual(resumed.call_count, 1)
                self.assertEqual(generated.concept_count, 2)
                self.assertEqual(len(list(Path(cache_dir).rglob("*.json"))), 2)
        finally:
            db.close()

    def test_dense_generation_requests_stay_below_budget_and_resume_by_part(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(
                board="NCERT", class_level="11", subject="Chemistry",
                chapter_number=6, chapter_name="Equilibrium", slug="equilibrium_dense",
                source_hash="source-hash", status="indexed",
            )
            db.add(chapter)
            db.flush()
            # Dense short numeric tokens reproduce the logarithm-table pattern
            # that Groq tokenizes far more heavily than ordinary prose.
            first_text = " ".join(str(1000 + index % 9000) for index in range(1000))
            second_text = " ".join(str(2000 + index % 8000) for index in range(1000))
            db.add_all([
                ContentPage(
                    chapter_id=chapter.id,
                    page_number=1,
                    text=first_text,
                    char_count=len(first_text),
                ),
                ContentPage(
                    chapter_id=chapter.id,
                    page_number=2,
                    text=second_text,
                    char_count=len(second_text),
                ),
            ])
            db.commit()
            first_response = (
                '[{"concept_id":"table_one","title":"Table one",'
                '"definition":"First dense source page","source_pages":[1]}]'
            )
            second_response = (
                '[{"concept_id":"table_two","title":"Table two",'
                '"definition":"Second dense source page","source_pages":[2]}]'
            )
            model_record = [{
                "status": "success",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
            }]

            with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
                os.environ,
                {"CONTENT_GENERATION_CACHE_DIR": cache_dir},
            ):
                with patch(
                    "Logic.coach.model_gateway.model_gateway.complete",
                    side_effect=[first_response, RuntimeError("provider interrupted")],
                ) as interrupted, patch(
                    "Logic.coach.model_gateway.model_gateway.records",
                    return_value=model_record,
                ):
                    with self.assertRaisesRegex(RuntimeError, "provider interrupted"):
                        generate_concepts_for_chapter(
                            db,
                            chapter.id,
                            max_batch_chars=20_000,
                        )

                self.assertEqual(interrupted.call_count, 2)
                self.assertEqual(len(list(Path(cache_dir).rglob("*.json"))), 1)
                self.assertEqual(
                    db.query(ContentConcept).filter_by(chapter_id=chapter.id).count(),
                    0,
                )
                for call in interrupted.call_args_list:
                    messages = call.kwargs["messages"]
                    max_tokens = call.kwargs["max_tokens"]
                    self.assertGreaterEqual(
                        max_tokens,
                        CONTENT_GENERATION_MIN_OUTPUT_TOKENS,
                    )
                    self.assertLessEqual(
                        _estimate_generation_input_tokens(messages) + max_tokens,
                        CONTENT_GENERATION_REQUEST_TOKEN_BUDGET,
                    )

                with patch(
                    "Logic.coach.model_gateway.model_gateway.complete",
                    return_value=second_response,
                ) as resumed, patch(
                    "Logic.coach.model_gateway.model_gateway.records",
                    return_value=model_record,
                ):
                    generated = generate_concepts_for_chapter(
                        db,
                        chapter.id,
                        max_batch_chars=20_000,
                    )

                # The successful first part is reused; only the interrupted
                # dense part is sent again.
                self.assertEqual(resumed.call_count, 1)
                self.assertEqual(generated.concept_count, 2)
                resumed_messages = resumed.call_args.kwargs["messages"]
                resumed_max_tokens = resumed.call_args.kwargs["max_tokens"]
                self.assertLessEqual(
                    _estimate_generation_input_tokens(resumed_messages)
                    + resumed_max_tokens,
                    CONTENT_GENERATION_REQUEST_TOKEN_BUDGET,
                )
        finally:
            db.close()

    def test_truncated_generation_response_is_never_checkpointed_or_imported(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(
                board="NCERT", class_level="11", subject="Chemistry",
                chapter_number=6, chapter_name="Equilibrium", slug="equilibrium",
                source_hash="source-hash", status="indexed",
            )
            db.add(chapter)
            db.flush()
            db.add(ContentPage(
                chapter_id=chapter.id,
                page_number=1,
                text="Chemical equilibrium is dynamic." * 4,
                char_count=124,
            ))
            db.commit()

            response = (
                '[{"concept_id":"equilibrium","title":"Equilibrium",'
                '"definition":"A valid-looking salvaged prefix","source_pages":[1]}]'
            )
            with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
                os.environ,
                {"CONTENT_GENERATION_CACHE_DIR": cache_dir},
            ), patch(
                "Logic.coach.model_gateway.model_gateway.complete",
                return_value=response,
            ), patch(
                "Logic.coach.model_gateway.model_gateway.records",
                return_value=[{"truncated": True, "provider": "test", "model": "test"}],
            ):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    generate_concepts_for_chapter(db, chapter.id, max_batch_chars=500)

                self.assertEqual(list(Path(cache_dir).rglob("*.json")), [])
                self.assertEqual(db.query(ContentConcept).filter_by(chapter_id=chapter.id).count(), 0)
        finally:
            db.close()

    def test_malformed_tail_and_missing_citations_are_never_checkpointed(self):
        SessionTesting = self._session_factory()
        for response in (
            '[{"concept_id":"partial","title":"Partial","definition":"d",'
            '"source_pages":[1]}] trailing',
            '[{"concept_id":"uncited","title":"Uncited","definition":"d"}]',
        ):
            db = SessionTesting()
            try:
                chapter = ContentChapter(
                    board="NCERT", class_level="11", subject="Chemistry",
                    chapter_number=7, chapter_name="Redox", slug=f"redox_{len(response)}",
                    source_hash="source-hash", status="indexed",
                )
                db.add(chapter)
                db.flush()
                db.add(ContentPage(
                    chapter_id=chapter.id,
                    page_number=1,
                    text="Oxidation and reduction involve electron transfer." * 3,
                    char_count=150,
                ))
                db.commit()

                with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
                    os.environ,
                    {"CONTENT_GENERATION_CACHE_DIR": cache_dir},
                ), patch(
                    "Logic.coach.model_gateway.model_gateway.complete",
                    return_value=response,
                ), patch(
                    "Logic.coach.model_gateway.model_gateway.records",
                    return_value=[{
                        "status": "success",
                        "provider": "groq",
                        "model": "openai/gpt-oss-120b",
                    }],
                ):
                    with self.assertRaisesRegex(ValueError, "incomplete"):
                        generate_concepts_for_chapter(db, chapter.id, max_batch_chars=500)

                    self.assertEqual(list(Path(cache_dir).rglob("*.json")), [])
                    self.assertEqual(
                        db.query(ContentConcept).filter_by(chapter_id=chapter.id).count(),
                        0,
                    )
            finally:
                db.close()

    def test_embedding_backfill_detects_json_null_and_keeps_existing_vectors(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(slug="embedding-backfill", status="indexed")
            db.add(chapter)
            db.flush()
            missing = ContentChunk(
                chapter_id=chapter.id, chunk_id="missing", text="needs embedding",
                embedding=None,
            )
            existing = ContentChunk(
                chapter_id=chapter.id, chunk_id="existing", text="already embedded",
                embedding=[1.0, 0.0],
            )
            db.add_all([missing, existing])
            db.commit()

            with patch("Logic.content_pipeline.embeddings_service.embeddings_enabled", return_value=True), \
                 patch("Logic.content_pipeline.embeddings_service.embed_texts", return_value=[[0.0, 1.0]]) as embed, \
                 patch("Logic.content_pipeline.embeddings_service.embedding_model", return_value="test-model"):
                result = embed_missing_chunks(db, chapter_id=chapter.id)
                db.commit()

            self.assertEqual(result["embedded"], 1)
            embed.assert_called_once_with(["needs embedding"])
            self.assertEqual(missing.embedding, [0.0, 1.0])
            self.assertEqual(existing.embedding, [1.0, 0.0])
        finally:
            db.close()

    def test_chunk_pages_and_coverage_report(self):
        pages = [
            {
                "page_number": 1,
                "text": "Photosynthesis is the process used by green plants. It needs sunlight and chlorophyll.",
                "char_count": 84,
                "extraction_quality": 0.8,
            },
            {
                "page_number": 2,
                "text": "The process produces glucose and oxygen. Chlorophyll captures light energy.",
                "char_count": 74,
                "extraction_quality": 0.7,
            },
        ]
        chunks = chunk_pages(pages, max_chars=120, min_chars=10)

        report = build_coverage_report(
            pages,
            [
                {
                    "source_pages": [1, 2],
                }
            ],
            chunks,
        )

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(report["coverage_score"], 1.0)
        self.assertTrue(report["ready_for_approval"])

    def test_validate_concepts_flags_missing_and_out_of_range_sources(self):
        payload = [
            {
                "concept_id": "photosynthesis",
                "title": "Photosynthesis",
                "definition": "Green plants make food using sunlight.",
                "source_pages": [1, 9],
            },
            {
                "concept_id": "empty-source",
                "title": "Empty Source",
                "definition": "A concept without page evidence.",
                "source_pages": [],
            },
        ]

        concepts, issues = validate_concept_payloads(payload, available_pages=[1, 2])

        self.assertEqual(len(concepts), 2)
        issue_names = {name for issue in issues for name in issue.get("issues", [])}
        self.assertIn("source_page_out_of_range", issue_names)
        self.assertIn("missing_source_pages", issue_names)

    def test_approval_requires_validated_concepts(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(
                slug="ncert_class_11_chemistry_chapter_1",
                status="indexed",
                validation_report={"ready_for_approval": True},
                concept_count=0,
            )
            db.add(chapter)
            db.commit()

            with self.assertRaisesRegex(ValueError, "no validated concepts"):
                approve_chapter(db, chapter.id, approved_by="tester")
        finally:
            db.close()

    def test_approval_bumps_version_only_when_source_pdf_changed(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            chapter = ContentChapter(
                slug="ncert_class_11_chemistry_chapter_1",
                status="validated",
                source_hash="hash_a",
                validation_report={"ready_for_approval": True},
                concept_count=2,
            )
            db.add(chapter)
            db.commit()
            # Approval now re-evaluates coverage from real rows, so give the
            # chapter pages and concepts that fully cover them.
            for page_number in (1, 2):
                db.add(ContentPage(chapter_id=chapter.id, page_number=page_number,
                                   text="x", char_count=100, extraction_quality=1.0))
            for page_number in (1, 2):
                db.add(ContentConcept(chapter_id=chapter.id, concept_id=f"concept_{page_number}",
                                      title=f"Concept {page_number}", definition="d",
                                      source_pages=[page_number]))
            db.commit()

            # First approval: goes live as v1 and records the live hash.
            approve_chapter(db, chapter.id, approved_by="founder")
            db.commit()
            self.assertEqual(chapter.version, "v1")
            self.assertEqual(chapter.published_source_hash, "hash_a")

            # Re-approval without a PDF change must NOT bump the version.
            chapter.status = "validated"
            db.commit()
            approve_chapter(db, chapter.id, approved_by="founder")
            db.commit()
            self.assertEqual(chapter.version, "v1")

            # Re-ingest with a new PDF (new source hash), then re-approve:
            # the version must bump so traces can name the source revision.
            chapter.status = "validated"
            chapter.source_hash = "hash_b"
            db.commit()
            approve_chapter(db, chapter.id, approved_by="founder")
            db.commit()
            self.assertEqual(chapter.version, "v2")
            self.assertEqual(chapter.published_source_hash, "hash_b")
        finally:
            db.close()

    def test_search_approved_content_ignores_unapproved_chapters(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            approved = ContentChapter(
                board="NCERT",
                class_level="10",
                subject="Science",
                chapter_name="Life Processes",
                slug="ncert_class_10_science_life_processes",
                status="approved",
            )
            draft = ContentChapter(
                board="NCERT",
                class_level="10",
                subject="Science",
                chapter_name="Draft Chapter",
                slug="draft_chapter",
                status="validated",
            )
            db.add_all([approved, draft])
            db.flush()
            db.add(
                ContentConcept(
                    chapter_id=approved.id,
                    concept_id="photosynthesis",
                    title="Photosynthesis",
                    definition="Photosynthesis lets green plants prepare food using light.",
                    core_explanation="Plants convert light energy into chemical energy.",
                    source_pages=[3],
                )
            )
            db.add(
                ContentChunk(
                    chapter_id=draft.id,
                    chunk_id="draft_chunk",
                    text="Draft-only photosynthesis text should not be returned.",
                    page_start=1,
                    page_end=1,
                    lexical_terms=["photosynthesis"],
                )
            )
            db.commit()

            with patch("Logic.content_pipeline.SessionLocal", SessionTesting):
                result = search_approved_content(
                    "photosynthesis",
                    "photosynthesis means",
                    scope={"subject": "Science", "chapter": "Life Processes"},
                )

            self.assertEqual(result["source"], "approved_content_pipeline")
            self.assertIn("green plants prepare food", result["context"])
            self.assertNotIn("Draft-only", result["context"])
        finally:
            db.close()

    def test_search_scope_topic_narrows_to_matching_chapter(self):
        SessionTesting = self._session_factory()
        db = SessionTesting()
        try:
            hydrocarbons = ContentChapter(
                board="NCERT",
                class_level="11",
                subject="Chemistry",
                chapter_name="Hydrocarbons",
                slug="ncert_class_11_chemistry_hydrocarbons",
                status="approved",
            )
            thermodynamics = ContentChapter(
                board="NCERT",
                class_level="11",
                subject="Chemistry",
                chapter_name="Thermodynamics",
                slug="ncert_class_11_chemistry_thermodynamics",
                status="approved",
            )
            db.add_all([hydrocarbons, thermodynamics])
            db.flush()
            db.add(
                ContentChunk(
                    chapter_id=hydrocarbons.id,
                    chunk_id="hydro_chunk",
                    text="Combustion of hydrocarbons releases energy as heat.",
                    page_start=4,
                    page_end=4,
                    lexical_terms=["combustion", "hydrocarbons", "energy"],
                )
            )
            db.add(
                ContentChunk(
                    chapter_id=thermodynamics.id,
                    chunk_id="thermo_chunk",
                    text="Thermodynamics studies combustion energy transfer in systems.",
                    page_start=9,
                    page_end=9,
                    lexical_terms=["combustion", "energy", "thermodynamics"],
                )
            )
            db.commit()

            with patch("Logic.content_pipeline.SessionLocal", SessionTesting):
                # Topic names a chapter: results must come only from it.
                narrowed = search_approved_content(
                    "combustion",
                    "what happens during combustion",
                    scope={"subject": "Chemistry", "topic": "Thermodynamics"},
                )
                # Topic finer-grained than any chapter name: keep all chapters.
                fallback = search_approved_content(
                    "combustion",
                    "what happens during combustion",
                    scope={"subject": "Chemistry", "topic": "alkanes"},
                )

            self.assertIn("Thermodynamics studies combustion", narrowed["context"])
            self.assertNotIn("Combustion of hydrocarbons", narrowed["context"])
            self.assertIn("Combustion of hydrocarbons", fallback["context"])
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
