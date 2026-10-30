# Planning and Study Lab integration

Planning now persists the learning route before handing off to Study Lab. Both surfaces read the same backend portfolio, evidence-derived mastery, review needs and effective proficiency. Browser storage is a cache; it can be rebuilt from the saved backend route.

## State and persistence

`StudyRouteState.runtime_json` captures the itinerary, ordered unit IDs, route completions, private assessment pack, assessment receipts and temporary branch return position. The latest user route is read across dates. Repeating Start with unchanged selections resumes an unfinished route; explicit new selections rebuild it. Heartbeats save the unit, conversation, message position and selected check answers. Done for Today pauses unfinished learning without claiming mastery.

Migration `20261030_0016` adds one nullable JSON column, preserves existing rows, and backfills old itinerary state when read. Procfile and Docker startup already run `scripts/migrate.py`. The shared feature defaults on; an explicit `STUDY_LAB_PLANNING_INTEGRATION=false` still disables it. The local SQLite database was backed up and migrated to this revision.

## Teaching and notes

Planning starts `/dashboard/study?planned=1`; Study Lab loads chapter, initial proficiency, goal, available time, active unit, route position, reason, duration, depth, prerequisites and progress automatically. The existing conversation resumes through its saved ID.

`services/study_route_service.py::learning_notes` is the replaceable notes adapter. It reads approved concept key points when available and falls back to the unit's authored skills/description. `components/study/PlannedLearningContext.tsx::LearningNotes` renders simple bullets. Amit's future notes format can replace this adapter/renderer without changing route logic. Existing free tutoring and practice modes remain available.

## Assessment and adaptation

`POST /planning/route/check/{user_id}` reuses the existing grounded MCQ generator and quota rules. The answer key stays on the backend; the saved pack survives refresh. The existing completion endpoints grade submitted selections against that key. Client-reported score/XP cannot establish mastery.

Assessment, TopicPerformance, study exposure, route movement and the retry receipt are committed together. PostgreSQL mutations lock the active route row. Retries return the saved receipt without duplicating evidence. Mastery keeps the curriculum's practice floor and independent-evidence rules. Strong performance reduces future instructional effort; weak performance retains Needs Review even after it leaves the recent-session window.

A weak unit can trigger an explained prerequisite refresh. Targets must belong to its actual prerequisite graph; forward jumps and nested branches are rejected. Successful review restores the original unit, route index and conversation position. The existing NCERT engine resolves subsequent units without changing conceptual order. Planning refreshes the same persisted route and canonical portfolio on return/focus.

## Important files

Backend: `services/study_route_service.py`, `services/planning_progress_service.py`, `Logic/analytics_engine.py`, `Logic/planning/{portfolio_engine,recommendation_engine}.py`, `models.py`, `schemas.py`, `routers/{planning,coach}.py`, `app/config.py`, and migration `20261030_0016_route_runtime.py`.

Frontend: `features/planning/{PlanningExperience,PlanningActive,contracts}`, `features/study/api.ts`, `app/dashboard/study/page.tsx`, `components/study/{StudySessionWorkspace,PlannedLearningContext,StudyQuickCheck}`, plus scoped styles and tests.

## Validation

- Backend: `python -m pytest -q --disable-warnings`: 401 passed, 71 subtests passed. Existing deprecation warnings remain.
- Backend: `python -m compileall -q main.py app routers services Logic prompts`: passed.
- Integration tests cover cross-date/session persistence, real SQL mastery updates, weak/strong checks, branch return, idempotency, forged evidence rejection, ownership isolation, atomic rollback, retained weak evidence, and repeatable migration/data preservation.
- Migration entrypoint ran twice against an isolated database; local migration and isolated application lifespan startup/health passed.
- Frontend: `npm test`: 202 passed; `npm run typecheck` and `npm run build`: passed.
- Frontend: `npm run lint`: no errors; one pre-existing unused-variable warning on the standalone chapter-notes page.
- Playwright: 10 desktop/mobile Planning/Study light/dark regression checks passed. The extended planned-learning test additionally passed on desktop and mobile through visible bullet notes, refresh, saved answers, conversation continuation, scored completion and return to Planning.

Browser checks intercept authenticated backend/model responses; backend tests exercise actual database/evidence/route logic with model generation stubbed. Hosted deployment and live provider generation are not asserted by these tests. No source-code blocker remains; hosted rollout uses the existing deployment pipeline and migration entrypoint.
