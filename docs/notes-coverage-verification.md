# Chemistry notes coverage and future-subject verification

See the [current NCERT-only reader and timed session contract](adaptive-notes.md). The counts below document the November 9 release.

Rechecked on November 9, 2026 against the bundled Class XI Chemistry release, indexed Amit notes, authenticated notes endpoints and private progress endpoints. The audit compares the restored release, curriculum registry and student catalog before opening lessons. Database and browser checks used isolated test data.

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

All 129 direct subtopic links were reopened with a saved topic bookmark. Fifty-seven links had to move to a different page to reach the requested subtopic. The saved bookmark and completed pages remained intact. Ask AI retrieved the exact teaching text and provenance for all 129 entry pages. A fresh database bootstrap restored all nine released chapters and verified installation of Amit's 64 embedded sections.

## Future Physics and Mathematics

Integration tests create realistic approved Physics and Mathematics chapters through the existing catalog and source services: 8 topics and 16 subtopics across 96 proficiency/style views. Tests cover actual PDF upload and section indexing, Master-first retrieval, missing-only NCERT fallback, worked-example preservation, durations, cache isolation/invalidation, reading progress, page-scoped Ask AI and return links. These fixtures demonstrate engine readiness; future textbooks still require their actual reviewed source content to be uploaded/indexed and approved.

Production-build desktop/mobile checks cover Physics with 4 pages and Mathematics with 12 pages, direct Planning entry, completion/reload and exact Ask AI context. Roman/numeric class labels now share one identity across Planning and catalog caching. The shared Study Lab catalog includes every published subject for the selected class, so adding Physics or Mathematics alongside Chemistry does not hide those chapters.

## Safeguards against future missing notes

- Approval recomputes teaching readiness for every source topic. A title-only topic, empty/ambiguous source ID, or unsupported formula record blocks publication with a specific validation issue; correcting the content clears the current blocker without clearing unrelated source errors.
- Structured formula records preserve their supplied expressions/LaTeX, symbol meanings, units and conditions. Explicit solved Physics/Math examples retain numerical teaching structure and atomic original solutions.
- CI runs the full pytest suite (including the formerly skipped pytest-based notes checks), the scientific-content audit and the authenticated endpoint/progress audit on every main push/PR.
- Reproduce with `python scripts/validate_adaptive_notes.py`, `python scripts/audit_notes_endpoints.py`, and `python -m pytest -q` after installing `requirements-dev.txt`. The detailed per-link/per-chapter reports are generated under ignored `tmp/`.

The November 9 backend suite passed 589 tests and 71 subtests. The production frontend build passed 18 focused desktop/mobile checks covering dynamic notes and the future-subject fixtures. Browser API transport is mocked. Backend checks use the real indexing, retrieval, topic mapping, compilation and progress services.
