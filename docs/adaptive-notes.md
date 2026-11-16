# Adaptive NCERT notes and timed study

See the [chapter-by-chapter recheck and future-subject verification](notes-coverage-verification.md) for the expanded endpoint audit and publication safeguards.

The reader uses the verified NCERT lesson assets and the approved NCERT catalog.

## Retrieval and scope

`notes_scope_service` resolves the existing registered curriculum or approved published catalog, retaining its unit/subtopic IDs and order. The reviewed Chemistry release retains its stable IDs; old grouped links resolve to those IDs or explicit topic choices. An explicit class/subject mismatch fails closed instead of entering an unscoped legacy lookup.

`notes_material_service.retrieve_material` looks up the mapped NCERT sections through reviewed teaching assets, exact PDF sections, and approved NCERT concepts. Opening notes requires no model or embedding request. Migration `20261116_0019` removes the retired source records and cached lessons that used them.

## Teaching and pagination

`adaptive_content_compiler` uses retrieved structured content or source paragraphs, separates definitions/explanations/examples, preserves scientific expressions, and keeps worked solutions atomic. The three Planning proficiencies vary scaffolding, visible explanation detail, conceptual grouping and difficulty of source-backed activities. Concise views retain complete source reasoning in expandable blocks. Explicit detailed preferences restore support. Definitions, laws, formulas, units, conditions and worked solutions are retained. The compiler does not invent examples or numerical answers to fill a page or time quota.

Pages follow meaningful concept/application/solution boundaries. They may be fewer or more than ten. Pattern interrupts are separate typed cards with source-scoped questions, optional hints and revealed explanations. Revision questions use existing source outcomes and proficiency, attached to a meaningful final page.

Topic min/max ranges determine workload; current session time selects a weighted portion without removing learning outcomes. Chapter 1 remains exactly 30–35, 45–50, 45–50, 40–45, 60–75 and 75–90 minutes. Overrides affect only their exact configured curriculum/unit. Future topics can be configured through the protected API below.

## Cache and progress

Migration `20261108_0018` adds `adaptive_note_lessons` and `topic_duration_overrides`. Existing deploy commands already run `scripts/migrate.py`; run it before starting a locally migrated backend too.

Shared lesson variants are keyed by compiler edition, curriculum/unit, proficiency, explanation style, source fingerprint and topic duration. Changes to any of these invalidate reuse, including pinned links. Session minutes select a portion independently of the cached lesson. The cache contains no student-specific data.

Private reading ledgers stay in the existing locked StudyRouteState runtime. They store page IDs, completion IDs and ordered navigation writes against an actual cached edition; malformed counts/IDs and cross-topic saves are rejected. Reading never creates mastery or assessment evidence. Route restart/time changes preserve reading ledgers and semantic anchors. A source/layout change rebases the bookmark by content/source/subtopic while clearing completion claims for new pages. Legacy v1 bookmarks are translated by their original mapped subtopics. The daily session window is frozen on first open and refreshed from the saved position the next day.

Planning and Study Lab Start Learning enter notes directly across scopes. A direct subtopic link keeps a matching bookmark or opens the first page that teaches that subtopic. Opening the link does not update the saved topic bookmark or page completion. Continue opens/resumes the existing learning/check session. Ask AI verifies the notes edition, page and subtopic scope and returns provenance for the matched page. Stale or mismatched page references return a page-unavailable result. Return links restore that edition/page or use the rebased position after source/configuration changes. Revision and Exam retain their existing approved hybrid retrieval outside notes-bound tutoring.

## Timed sessions

Opening the active topic from Planning starts one saved deadline for the route. Finishing its pages records the topic as read and opens the next eligible topic in the planned sequence. The route keeps the next topic available when the timer expires. Quick checks continue to record mastery separately. Saved “Know a Little” choices map to “New to It.” Saved unlimited sessions map to 30 minutes.

## Protected administration

All endpoints require the existing backend admin role and audit successful writes:

- `GET /admin/notes/topic-durations` with `chapter_ref`, `class_level`, `subject`.
- `PUT /admin/notes/topic-duration` with that scope, `unit_id`, `minimum_minutes`, `maximum_minutes`: validates separate integers, bounds and order; persists only that unit and invalidates its variant naturally. The six reviewed Chapter 1 ranges are protected.

## Reproducible validation

`python -m pytest -q` exercises existing features plus NCERT provenance, generic subjects, scoping, variant invalidation, dynamic progress, legacy bookmarks, duration authorization and database migrations. `python scripts/validate_adaptive_notes.py` audits every mapped unit of the bundled Chemistry release against actual source material and all three proficiency views in an isolated database; no live student data or provider is required. The frontend includes desktop/mobile browser checks for dynamic counts, completion, regeneration, source cards, notation, Ask AI return, direct Planning entry and learning/mastery continuation.
