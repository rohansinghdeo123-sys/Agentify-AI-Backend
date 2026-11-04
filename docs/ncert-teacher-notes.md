# NCERT teacher notes: Chapter 1

This is the current notes-content contract from Amit's November 4 prompt. It supersedes the earlier Master Notes-first excerpt presentation. NCERT is the primary scientific source; the earlier seed, original PDF and indexes remain available as reference and for archived-ID resolution, rather than being copied into the new reader.

## Authored lessons and provenance

Six checked-in JSON lessons in `data/notes/teacher_chapter1/` contain exactly ten logically ordered pages each. These are teaching explanations, not verbatim textbook quotations. Every page cites registered NCERT subtopic IDs, which resolve to the source PDF's section/page ranges and verified SHA-256. Teacher-created applications and explicitly marked clarifications are distinguishable from textbook calculations. The content was authored against the actual registered PDF and independently reviewed, including numerical arithmetic.

The opening pages use familiar observations and explain why a concept is needed before formal definitions. Explanations develop in small connected blocks with terminology introduced in context. Formula blocks explain terms/units; worked solutions include givens, target, method, substitution, arithmetic and interpretation. Page 10 revises essentials and supplies 5, 7, 7, 7, 8 and 10 answered understanding/application questions for the six topics. Answers are revealed on demand and are separate from graded mastery checks.

Known source inconsistencies are handled explicitly: current SI definitions take precedence over the historical reference-standard sidebar; exact modern Avogadro constant and rounded calculation constants are distinguished; glucose's printed unrounded 180.162 u result is distinguished from the precision-rule result 180.16 u; the solution-mass subtraction 1250-175.5=1074.5 g is explained where the extracted line loses a digit. No values are silently changed while being attributed to the printed source.

## Adaptation and time

`services/teacher_notes_service.py` validates and compiles the authored contract. All essential blocks remain for every learner, including essential definitions, equations and numerical reasoning. Optional foundation blocks support beginners; detailed preference restores scaffolding; confident learners omit redundant foundations and receive distinct extensions where the configured workload permits. Concise preference removes optional extensions. Saved coach style preferences and the existing analytics-derived effective chapter proficiency are consumed, including plans using archived full chapter slugs.

The manually configured topic ranges remain 30–35, 45–50, 45–50, 40–45, 60–75 and 75–90 minutes. Neither the notes service nor the renderer changes them. The current Planning route's actual main-focus allocation selects a recommended page portion. Relative lesson-step weights divide that configured workload; worked solutions and final practice receive more weight than an opening observation. Refresh does not slide the session window or manufacture extra time. Each day’s notes window resumes from its initial saved page without resetting the persistent route, conversations or prepared checks. Short sessions leave all ten pages available and flag remaining practice; reading never awards mastery.

No model call or remote embedding request is required to open or paginate the notes. Shared authored lessons/source verification are cached; learner adaptation is computed from the existing private route/profile. No new tables or migration are needed.

## Integration and persistence

The existing notes endpoint and canonical/archived topic mapping call the new compiler. Existing page navigation, themes, scroll container and progress bar remain. Structured teaching blocks render as coherent themed points rather than splitting every source sentence into an isolated bullet. Transitions and final review questions are page-local. Optional session guidance belongs to the reader rather than being artificial progress narration in the lessons.

Reading positions for `ncert_teacher_v1` are stored in the existing route JSON separately from the old excerpt layout. The old positions are retained, so the new lesson does not mistake a previously read excerpt index for newly read teaching content. A changed daily route retains edition-specific positions. Assessment records, mastery, adaptive interventions and manual durations are not reset.

Ask AI receives the exact page, content edition, source/subtopic IDs, proficiency, explanation preference and authoritative topic/session times. Both Planning retrieval entry points use the NCERT teaching pages. Page-specific retrieval cannot introduce later-page content; strict subtopic retrieval excludes mixed revision pages; relevance selection restores logical page order. Published Revision and other chapters retain their existing source contracts.

## Validation

Backend tests cover all six lessons across four proficiency levels and three style preferences; ten-page integrity; essential-content retention; strict scope; session splitting; final-page insufficient time; versioned resume/new route preservation; no mastery on reading; legacy-ID adaptation; exact page retrieval; and malformed content rejection. Existing platform regression tests are run alongside them.

Browser transport fixtures are generated by the real service. Desktop/mobile tests exercise ten-page navigation, revision answers, current session portion, edition-specific save/resume and page-bound Ask AI payloads, plus the existing Planning, scrolling and notation workflows. Browser APIs/model responses are mocked; hosted deployment and live model wording are not claimed as browser-verified.

Final local validation: 440 backend tests and 71 subtests passed; 212 frontend unit tests passed; all ten production-build desktop/mobile browser workflows passed. Typecheck, targeted ESLint and the final production build passed. Hosted deployment is not included in these local test results.
