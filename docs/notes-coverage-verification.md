# Chemistry notes coverage and future-subject verification

Verified on November 8, 2026 against the actual bundled Class XI Chemistry sources, indexed Amit notes, authenticated notes endpoints and private progress endpoints. All database/model work used isolated test data; no hosted-account access or live model wording is claimed.

## Every shipped Chemistry chapter

| Chapter | Topics | Mapped subtopics | Links checked | Learner variants | Dynamic pages | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1. Some Basic Concepts of Chemistry | 6 | 36 | 259 | 72 | 5–31 | Passed |
| 2. Structure of Atom | 7 | 28 | 260 | 84 | 8–18 | Passed |
| 3. Classification Of Elements And Periodicity In Properties | 7 | 8 | 34 | 84 | 1–14 | Passed |
| 4. Chemical Bonding And Molecular Structure | 9 | 10 | 45 | 108 | 1–14 | Passed |
| 5. Thermodynamics | 8 | 10 | 48 | 96 | 1–16 | Passed |
| 6. Equilibrium | 10 | 10 | 55 | 120 | 1–18 | Passed |
| 7. Redox Reactions | 6 | 7 | 35 | 72 | 1–15 | Passed |
| 8. Organic Chemistry Some Basic Principles And Techniques | 10 | 10 | 51 | 120 | 1–13 | Passed |
| 9. Hydrocarbons | 9 | 10 | 57 | 108 | 1–12 | Passed |

All 72 topics, 129 mapped subtopics, 844 canonical/archived/catalog links, 864 proficiency/style variants, and 864 save/reopen round trips passed without blank or missing notes. Forty-four archived links that span several topics show explicit valid choices; every destination was reopened. Reading did not alter mastery or the active route. The audit observed 19 distinct page counts, from 1 to 31, determined by actual content and learner settings.

The independent 288-variant source-content audit also passed: Master Notes first, missing-only NCERT fallback, no missing core coverage, no identical scientific teaching bodies across the four balanced levels, and no depth inversions.

## Future Physics and Mathematics

Integration tests create realistic approved Physics and Mathematics chapters through the existing catalog and source services: 8 topics and 16 subtopics across 96 proficiency/style views. Tests cover actual PDF upload and section indexing, Master-first retrieval, missing-only NCERT fallback, worked-example preservation, durations, cache isolation/invalidation, reading progress, page-scoped Ask AI and return links. These fixtures demonstrate engine readiness; future textbooks still require their actual reviewed source content to be uploaded/indexed and approved.

Production-build desktop/mobile checks cover Physics with 4 pages and Mathematics with 12 pages, direct Planning entry, completion/reload and exact Ask AI context. Roman/numeric class labels now share one identity across Planning and catalog caching. The shared Study Lab catalog includes every published subject for the selected class, so adding Physics or Mathematics alongside Chemistry does not hide those chapters.

## Safeguards against future missing notes

- Approval recomputes teaching readiness for every source topic. A title-only topic, empty/ambiguous source ID, or unsupported formula record blocks publication with a specific validation issue; correcting the content clears the current blocker without clearing unrelated source errors.
- Structured formula records preserve their supplied expressions/LaTeX, symbol meanings, units and conditions. Explicit solved Physics/Math examples retain numerical teaching structure and atomic original solutions.
- CI runs the full pytest suite (including the formerly skipped pytest-based notes checks), the scientific-content audit and the authenticated endpoint/progress audit on every main push/PR.
- Reproduce with `python scripts/validate_adaptive_notes.py`, `python scripts/audit_notes_endpoints.py`, and `python -m pytest -q` after installing `requirements-dev.txt`. The detailed per-link/per-chapter reports are generated under ignored `tmp/`.

The final backend suite passed 587 tests and 71 subtests both against the normal test setup and after migrating a fresh isolated source database. Frontend typecheck, ESLint and all 229 unit tests passed. Browser source/model transport is mocked; indexing, retrieval, topic mapping, dynamic compilation, publication validation and progress persistence use real application services in the backend tests.
