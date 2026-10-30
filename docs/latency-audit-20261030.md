# Latency audit and optimizations — 30 October 2026

## Findings and fixes

| Area | Finding | Implemented change |
|---|---|---|
| Shared frontend API | Forced-fresh GETs bypassed coalescing; cache identity omitted credentials; late reads could refill invalidated entries | Coalesce concurrent fresh GETs, partition caches by endpoint/credentials, bound cache entries, guard against stale refill after writes, and keep caller-owned abort signals independent |
| Planning | Portfolio POST was immediately followed by another complete build in route-start; mount/focus restoration could overlap | Use the authoritative portfolio returned by route-start; serialize restoration/refresh work and remove duplicate mount refresh |
| Notes | Every page query change refetched the full notes payload and required a server navigation; saving fetched route context then recomputed a large portfolio | Read page position from native URL history without RSC navigation/refetch; use active unit metadata already in notes; request compact heartbeat acknowledgements |
| Study Lab rendering | Every streamed token reparsed unchanged Markdown history, wrote conversation snapshots to localStorage, and restarted smooth-scroll animation | Memoize Markdown, batch deltas within network chunks, persist local conversations on completed/stopped turns, and use direct scrolling during streaming |
| Study streaming backend | Synchronous DB/catalog work ran on the event loop; preflight retained a pooled connection throughout streaming | Run the handler preflight in FastAPI's sync worker pool and close its read transaction before the dedicated streaming session starts |
| Revision / Exam / RAG | Shared read amplification and duplicate concurrent embedding/cache misses; repeated PDF source-anchor processing | Apply shared request coalescing, source-text-keyed anchor caching and single-flight embedding/cache builds; retain models, grounding, source filters, evidence and similarity rules |
| Exam upload | Synchronous PDF parsing/analysis/database work blocked the async event loop | Offload quota and the complete parse/persist/serialize operation to a worker while keeping capped asynchronous upload reading |
| Leaderboard | Loaded all profiles and progress records, including inactive accounts and unused columns | One filtered join fetching only rank/name/class/privacy fields; preserve exact Python casefold tie-breaking, per-class positions and email visibility |
| Rankings frontend | Refresh timers could overlap a slow ongoing refresh | Serialize refreshes and continue displaying available leaderboard data while optional challenge data loads |
| Analytics backend | Repeated session/user queries; committing metrics expired loaded ORM rows and caused N+1 serialization reads | Reuse loaded rows, narrow older-evidence queries, materialize the full response before committing changed metrics, and skip redundant metric writes |
| Analytics frontend | Requested unused full analytics and global leaderboard data; name/profile changes refetched data | Request the compact dashboard while still deriving/persisting every displayed cognitive metric; fetch by stable user ID |
| Navigation / transfer | Some routes had no immediate Suspense feedback; large JSON responses uncompressed by the application | Add dashboard loading boundary for immediate navigation feedback; gzip large responses with explicit SSE exclusion (verified against installed Starlette) |
| Database | Missing composite indexes for latest history/evidence, topic updates and session replay joins | Add idempotent migration `20261030_0017`: history `(user_id,date,id)` and `(user_id,topic,id)`, topic performance `(user_id,topic)`, session details `(test_id)` |

## Measured improvement

Reproduce with `python scripts/benchmark_latency.py --baseline-ref 720af30` in the backend repository. Synthetic local SQLite: 3,000 accounts, 100 active accounts, 2,000 sessions and 40 topics; median of 12 requests. Baseline and updated implementations use the same indexed schema to isolate application changes. Timing varies with machine load; this is not a hosted network or provider benchmark.

| Operation | Before | After | SELECT queries |
|---|---:|---:|---:|
| Analytics with metric persistence | 20.35 ms | 3.12 ms | 56 → 4 |
| Leaderboard | 51.37 ms | 2.94 ms | 2 full-table loads → 1 filtered join |
| Background save, full response vs compact | 10.11 ms | 0.81 ms | 8 → 1 |

The background-save response is 69,922 bytes in this fixture versus 14 bytes for the compact acknowledgement. The same writes, ownership validation and progress rules still run. Planning creation drops from two build requests to one; moving between already-loaded note pages now makes zero additional notes API requests or RSC navigations. Eight simultaneous identical fresh frontend GETs or query embeddings share one underlying request/build in regression tests.

## Files changed

Frontend: `lib/apiClient.ts`; `features/planning/{PlanningExperience,PlanningActive}.tsx`; `features/study/api.ts`; `features/rankings/useRankings.ts`; `components/{RichMarkdown,analytics/AnalyticsPage,study/StudySessionWorkspace}.tsx`; `app/dashboard/loading.tsx`, Study home and notes reader; API-client and notes browser tests.

Backend: `Logic/{analytics_engine,embeddings}.py`, `Logic/tools/knowledge_search.py`; `services/{ttl_cache,leaderboard_service,study_route_service,topic_notes_service}.py`; `routers/{coach,exam_papers,planning,progress}.py`; `main.py`, `models.py`, `schemas.py`; index migration, benchmark script, migration/regression tests and this report.

## Validation and delivery scope

Frontend: 210 unit tests, typecheck, lint and optimized production build passed. Backend full suite: 416 tests and 71 subtests passed, followed by focused nullable-history regression checks. Twelve desktop/mobile light/dark Planning/Study/notes regression flows passed, followed by four repeat notes flows after native-history pagination was introduced; the notes test now asserts no extra payload fetch on page change and retains real wheel/touch/keyboard scrolling and Ask AI return checks.

Backend regression tests cover coalescing, failure retry/invalidation, private cache isolation, exact leaderboard/privacy rules, persistent compact saves, metric accuracy and serialization query counts, repeatable index migration with preserved records/query plans, uncompressed SSE, and event-loop responsiveness during paper analysis. The index migration was applied locally after a SQLite backup; integrity check passed. Existing startup migration commands deploy the additive indexes.

Security checks, Firebase revocation verification, ownership checks, model routing, retrieval quality, source approval filters, scoring, mastery, analytics and session persistence remain active. No personalized answer cache or lower-quality model shortcut was introduced. Semantic retrieval already scans the scoped candidate corpus by design; it was not pruned to fake faster timings.

External model-generation time, Firebase/network round trips, hosting cold starts and deployed database contention cannot be eliminated or measured honestly from these local checks. Actual production p50/p95 and model first-token timings still require hosted traces. The existing response-time/request-ID headers remain available. This pass fixes the reproducible code-level bottlenecks; it does not claim every possible production latency source is eliminated.
