# NCERT teaching notes across Class XI Chemistry

The Chapter 1 teacher-notes contract now applies to all nine published Class XI Chemistry chapters: 72 existing Planning units and 720 teaching pages. Chapter 1's six lessons are retained; chapters 2–9 add 66 lessons in `data/notes/teacher_chemistry/chapter_02` through `chapter_09`. Every unit has ten separate learning steps, a connected explanation, essential scientific content, worked applications where relevant, and five to ten answered understanding questions on its final page.

## Sources and identities

`data/notes/teacher_chemistry/curricula.json` binds each lesson to an existing Planning unit, its mapped subtopics and NCERT source-page ranges. Chapter 2 retains its registered curriculum; chapters 3–9 retain the published catalog's unit IDs, curriculum keys, order and duration estimates. The snapshot was assembled using the existing published-curriculum and topic-grouping adapters against the approved release. It does not replace those adapters or change the Planning/mastery schema. Full published slugs, chapter names and short aliases resolve to the same notes; unknown or cross-chapter topics fail closed.

The NCERT PDFs used for chapters 3–9 are checked in alongside the already registered Chapter 1/2 PDFs and explicitly included in the Docker build context. Source digests are verified when compiling and then cached. New lessons are teacher-written explanations checked against those PDFs; the older generated summaries are not copied verbatim. Examples authored for teaching are applications of the source science. Historical textbook conventions and explanatory clarifications are distinguished where needed. No upload, model generation, embedding request, database reseed or content migration is required to read a lesson.

## Existing learner behavior

The shared compiler retains every essential block at all proficiency/style settings. Optional foundations explain prerequisites; optional extensions add distinct application or reasoning. Session guidance divides the existing configured workload without changing durations or hiding the remaining pages. The reader chooses its format from the API response, so another subject's existing concept-notes view still works and does not fetch the same notes twice.

The API supplies the correct catalog authority for Ask AI. Registered Planning topics use the teacher pages directly. Published topics opt into the same retrieval with the notes edition; ordinary published Revision retrieval keeps its existing contract. Exact-page/subtopic requests cannot pull in a later page or another chapter. Both tutor retrieval entry points use the same resolver.

Chapter 2's eight older published teaching-topic links resolve to their corresponding canonical teaching units. Its two archived mixed end-of-chapter review links retain the original scoped reader, since assigning a multi-unit review to one teaching topic would lose or misattribute content. All seven canonical Chapter 2 Planning units use the new ten-page lessons.

Reading positions use the existing `ncert_teacher_v1` edition namespace and stable unit IDs. Heartbeats also allow another chapter selected in the saved portfolio, without changing the active chapter/unit, assessment state or mastery. Chapters outside the saved portfolio do not issue a position write. Assessed learning continues through the existing quick-check, analytics and adaptive-route services.

Revision bookmarks also persist after a route is completed without reopening it or overwriting its saved conversation/check state. Display-only chemistry normalization distinguishes metal charge numbers from atom counts (Fe3+ versus H2+) and retains grouped counts such as Ca(OH)2; persisted teaching/source wording is unchanged.

## Validation and maintenance

Run `python scripts/validate_chemistry_teacher_notes.py` to verify all source hashes, unit/page identities, page counts, subtopic coverage and twelve proficiency/style combinations per unit without a model or database. `tests/test_chemistry_teacher_notes.py` additionally compares snapshots with the real published catalog, checks page-scoped retrieval and aliases, persists multiple chapters' positions, and verifies assessed mastery still advances a published chapter. Browser fixtures are generated through the actual notes service; browser model/API responses remain intercepted.

After an approved curriculum re-ingestion changes IDs, source hashes or unit membership, review and update the lesson snapshot and content together. Never silently attach a lesson to a changed topic or re-estimate a manually configured duration. Hosted rollout and live model wording require separate deployment verification; local tests do not prove either.

Final local validation: 465 backend tests and 71 subtests passed; 214 frontend unit tests passed; 28 production-build desktop/mobile browser checks passed. Typecheck, targeted ESLint, source/lesson validation and visual inspection passed. Independent reviews checked numerical arithmetic, reaction atom/charge balances and source scientific conventions.
