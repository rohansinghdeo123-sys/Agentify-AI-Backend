# Adaptive, source-bound notes (November 8)

See the [chapter-by-chapter recheck and future-subject verification](notes-coverage-verification.md) for the expanded endpoint audit and publication safeguards.

This is the current reader contract. It supersedes the historical fixed ten-page, NCERT-primary presentation. Existing reviewed lessons remain a verified NCERT source adapter, rather than a second active reader or a new topic hierarchy.

## Retrieval and scope

`notes_scope_service` resolves the existing registered curriculum or approved published catalog, retaining its unit/subtopic IDs and order. The reviewed Chemistry release retains its stable IDs; old grouped links resolve to those IDs or explicit topic choices. An explicit class/subject mismatch fails closed instead of entering an unscoped legacy lookup.

`notes_documents_service` stores master documents separately in existing ContentChapter/Page/Chunk records (`indexed_master`, `MASTER_NOTES`). Each reviewed section binds to existing unit/subtopic IDs and retains source/text hashes, page references and source order. It never replaces an NCERT chapter or publishes a duplicate catalog chapter. The bundled Amit installer preserves all 64 original sections and their 3072-dimensional embeddings; 62 have exact bindings. The two cross-topic summary sections remain indexed and explicitly flagged for mapping review.

`notes_material_service.retrieve_material` searches that master index first. Complete reviewed coverage skips the NCERT adapter entirely. Only missing subtopics enter the verified reviewed-lesson/PDF/approved-concept adapters; unrelated sections and repeated sentences are excluded. Formula discrepancies are withheld and returned for review. Source labels are actual `MASTER_NOTES` or `NCERT_FALLBACK` provenance, never a prompt-only attribution. No model or remote embedding call is needed to open or paginate notes.

## Teaching and pagination

`adaptive_content_compiler` uses retrieved structured content or source paragraphs, separates definitions/explanations/examples, preserves scientific expressions, and keeps worked solutions atomic. The four Planning proficiencies vary scaffolding, visible explanation detail, conceptual grouping and difficulty of source-backed activities. Concise views retain complete source reasoning in expandable blocks. Explicit detailed preferences restore support. Definitions, laws, formulas, units, conditions and worked solutions are retained. The compiler does not invent examples or numerical answers to fill a page or time quota.

Pages follow meaningful concept/application/solution boundaries. They may be fewer or more than ten. Pattern interrupts are separate typed cards with source-scoped questions, optional hints and revealed explanations. Revision questions use existing source outcomes and proficiency, attached to a meaningful final page.

Topic min/max ranges determine workload; current session time selects a weighted portion without removing learning outcomes. Chapter 1 remains exactly 30–35, 45–50, 45–50, 40–45, 60–75 and 75–90 minutes. Overrides affect only their exact configured curriculum/unit. Future topics can be configured through the protected API below.

## Cache and progress

Migration `20261108_0018` adds `adaptive_note_lessons` and `topic_duration_overrides`. Existing deploy commands already run `scripts/migrate.py`; run it before starting a locally migrated backend too.

Shared lesson variants are keyed by compiler edition, curriculum/unit, proficiency, explanation style, source fingerprint and topic duration. Changes to any of these invalidate reuse, including pinned links. Session minutes select a portion separately and do not trigger unnecessary compilation. The cache contains no student-specific data.

Private reading ledgers stay in the existing locked StudyRouteState runtime. They store page IDs, completion IDs and ordered navigation writes against an actual cached edition; malformed counts/IDs and cross-topic saves are rejected. Reading never creates mastery or assessment evidence. Route restart/time changes preserve reading ledgers and semantic anchors. A source/layout change rebases the bookmark by content/source/subtopic while clearing completion claims for new pages. Legacy v1 bookmarks are translated by their original mapped subtopics. The daily session window is frozen on first open and refreshed from the saved position the next day.

Planning and Study Lab Start Learning enter notes directly across scopes. Continue opens/resumes the existing learning/check session. Ask AI passes exact page ID/version, subtopics, source evidence and proficiency; server retrieval verifies the same page. Return links restore that edition/page or use the rebased position after source/configuration changes. Revision and Exam retain their existing approved hybrid retrieval outside notes-bound tutoring.

## Protected administration

All endpoints require the existing backend admin role and audit successful writes:

- `POST /admin/notes/master-documents`: structured master source with `title`, `document_key`, `class_level`, `subject`, `chapter_ref`, and `sections`. Each section supplies unique `id`, `title`, existing `unit_id`, existing `subtopic_ids`, `text`, `source_pages`, `teaching_format`, `reviewed`, `complete`. Unreviewed/incomplete content never falsely counts as complete teaching coverage. `embed: true` uses the existing embedding provider during indexing; mapped reads are network-free.
- `POST /admin/notes/master-documents/upload`: multipart PDF (25 MB maximum) plus the same metadata and `sections_json`. Sections use exact source pages/anchors or verified text from those pages. Indexed source text, metadata and lexical terms persist in the database.
- `GET /admin/notes/master-documents` with chapter/class/subject, and `GET /admin/notes/master-documents/{id}`: inspect source/binding/review evidence.
- `GET /admin/notes/topic-durations` with `chapter_ref`, `class_level`, `subject`.
- `PUT /admin/notes/topic-duration` with that scope, `unit_id`, `minimum_minutes`, `maximum_minutes`: validates separate integers, bounds and order; persists only that unit and invalidates its variant naturally. The six reviewed Chapter 1 ranges are protected.

## Reproducible validation

`python -m pytest -q` exercises existing features plus actual source persistence/priority, generic subjects, scoping, variant invalidation, dynamic progress, legacy bookmarks, duration authorization and reversible migration. `python scripts/validate_adaptive_notes.py` audits every mapped unit of the bundled Chemistry release against actual source material and all four proficiency views in an isolated database; no live student data or provider is required. The frontend includes desktop/mobile browser checks for dynamic counts, completion, regeneration, source cards, notation, Ask AI return, direct Planning entry and learning/mastery continuation.

Release validation: 549 backend tests and 71 subtests passed; 215 frontend unit tests, typecheck, ESLint and production build passed. The full desktop/mobile browser regression run passed 164 workflows, followed by 36 focused checks after the scientific Markdown/preview changes. Browser API/model transport is mocked; retrieval, indexing, migrations and the 72-unit/288-variant source audit use real application services and source artifacts. The audit found zero missing coverage, identical teaching bodies, depth inversions, or source/progress/cache errors. Local database migration, all 64 Amit vectors, and persistence of all 288 current lesson variants were verified. Hosted rollout and live model wording are not claimed as tested by those local checks.
