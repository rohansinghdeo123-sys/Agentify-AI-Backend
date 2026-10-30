# Chemistry Chapter 1: fixed durations and scoped notes

- The six existing stable unit IDs, NCERT sequence, prerequisites, statuses and initial proficiency remain unchanged. Their authored min/max durations are now 30–35, 45–50, 45–50, 40–45, 60–75 and 75–90 minutes. `duration_source=manual` keeps the full topic estimate separate from shorter scheduled sessions. Existing saved allocations above the corrected maximum are reconciled without resetting unit/branch/reading state.
- The duplicate global-card Start Learning action is removed. Chapter 1 topic and Start today's route actions open the selected/current topic's notes directly. Other chapters retain their existing reader/routing.
- `data/notes/chapter_1_notes_mapping.json` maps each stable Planning unit to its authored NCERT subtopics and explicit excerpts from the approved bundled master notes. Each excerpt retains its source concept, paragraph and original-text SHA-256. No merged source concept is copied wholesale across topic boundaries.
- `services/topic_notes_service.py` serves only the selected unit's excerpts through the existing authenticated notes API. Reading position uses the existing route `runtime_json.notes_positions`, survives new daily routes, and never claims mastery. No schema change is needed.
- The reader provides Previous/Next, page count, progress, formula-preserving text and contextual Ask AI. The tutor receives chapter/topic/subtopic/page/proficiency and the exact page text through its existing grounding context. The return URL restores the same notes page.

## Source review still required

The available bundled master notes contain mixed-topic summaries and practice material. Their paragraph identities are explicitly listed under `review_required`, not silently assigned. A water-composition example reports percentages totalling 99.97%; that sentence is withheld pending scientific review. Mixed exercise paragraphs need a verified source-level split before they can be included safely. The earlier fine-grained master file covers only opening matter concepts, so it cannot substitute for complete approved notes for all six topics.

Current reader content consists of verified, topic-isolated excerpts, not a claim that every NCERT subtopic now has complete final notes. Amit's authoritative final notes can replace/extend these excerpts using the same mapping contract. No new scientific content or numerical correction was invented.

## Validation

- Full backend: 405 tests passed, 71 subtests passed; existing deprecation warnings remain.
- Source tests verify exact six ranges, short-session ceilings, original excerpt hashes, unique excerpt assignment, complete authored subtopic lists, Masses topic isolation, SQL reading-position persistence and unaffected other chapters.
- Frontend unit suite: 203 tests passed. Typecheck and lint passed.
- Desktop and mobile Playwright: Planning route opens scoped notes, pagination/refresh/resume works, chemical notation displays, Ask AI receives exact page text/proficiency and returns to the same page. Backend/model responses are mocked in browser tests; backend persistence tests use real SQLite.
- Production build and compile checks passed. Hosted deployment and live provider responses are not asserted by these tests.
