# Content Automation — NCERT → published, hands-off

One command downloads NCERT textbook PDFs and runs the **entire** study-content
pipeline, so neither Amit nor Rohan has to do manual ingest/approve work. You
just **monitor the admin page** (Operations → Data & content pipeline) or run
`python scripts/content_report.py`.

## What it does (per chapter, automatically)
1. **Download** the NCERT chapter PDF (polite: rate-limited, identifies itself,
   resumable, validates both the PDF file and expected curriculum title, respects
   robots.txt).
2. **Ingest** → pages + retrieval chunks.
3. **Generate concepts** with the content agents (the real subtopics).
4. **Embed** chunks for semantic retrieval.
5. **Auto-publish — only if it passes the quality gate** (`publish_chapter`
   enforces coverage ≥ `CONTENT_MIN_COVERAGE_SCORE` and no blocking validation
   issues). Chapters that fail the gate are left as **`needs_review`** and show
   up in the admin report for a human to check — they are **never** published
   unverified.

Every chapter is isolated: one failure never stops the run. Re-running skips
PDFs already downloaded and republishes only if the source changed. Replacing a
live chapter is atomic: students keep the last published version unless the new
extraction, concepts, embeddings, and publication gate all succeed.

## How to run (from the `backend/` directory)
```bash
# Class 11 & 12 — Physics, Chemistry, Maths — full pipeline, auto-publish good ones
python scripts/automate_content.py

# Narrow scope
python scripts/automate_content.py --classes 11 --subjects Chemistry

# Resume or repair exact chapters
python scripts/automate_content.py --classes 11 --subjects Chemistry --chapters 3,4
python scripts/automate_content.py --classes 11 --subjects Chemistry --chapters 8 --reingest

# Check configuration without downloading or touching the database
python scripts/automate_content.py --preflight-only

# Just fetch the PDFs (no ingestion)
python scripts/automate_content.py --download-only

# Ingest + generate + embed, but DON'T auto-publish (approve manually in admin)
python scripts/automate_content.py --no-publish

# Be extra polite / probe fewer chapters
python scripts/automate_content.py --delay 6 --max-chapters 25

# Override concept-generation pacing only for a verified provider limit
python scripts/automate_content.py --generation-batch-delay 90
```

`--reingest --no-publish` is intentionally rejected for a chapter that is
already live. The current schema stores one row per chapter, so a live repair
uses auto-publish as one gated transaction and rolls back to the existing
published version if any step fails.

A run summary is written to `content_automation_run.json` (downloaded / published
/ needs_review / failed counts + per-chapter status).

## Requirements
- Runs against whatever `DATABASE_URL` is configured (loads `backend/.env`). Point
  it at the **live DB** to populate production (Neon URL **without** `sslmode`).
- LLM + embeddings keys must be set for concept generation and embedding:
  `GROQ_API_KEY` (concepts) and `EMBEDDINGS_API_KEY` (semantic vectors).
- When Groq is in the configured model route, uncached generation calls default
  to a conservative 61-second interval to respect its cumulative TPM window.
  `CONTENT_GENERATION_BATCH_DELAY_SECONDS` or `--generation-batch-delay` can
  override this for a different verified account limit.
- This is **best run locally by Rohan** (or on an always-on worker): it is a long,
  heavy job and the free-tier web instance sleeps/￼is memory-limited, so it should
  not run inside that instance.

## Scheduling (optional, fully hands-off)
- **Windows:** Task Scheduler → run `python scripts/automate_content.py` daily/weekly.
- **cron (Linux/macOS):** `0 3 * * 0 cd /path/backend && python scripts/automate_content.py`
- Because it's resumable and idempotent, re-runs only fetch/process what's new or changed.

## Scope / adding subjects
Book codes live in `Logic/content_automation.py` → `NCERT_BOOKS`
(`(class, subject) → [book codes]`). Add entries to extend to more classes/subjects.
NCERT chapter PDFs are at `https://ncert.nic.in/textbook/pdf/<code><NN>.pdf`.

## Monitoring (what Amit/Rohan check)
- Admin page → **Operations → Data & content pipeline**: per-chapter status
  (`published` vs `needs_review`), coverage, memory, subtopics, error rate.
- Or `python scripts/content_report.py` for the same report in the terminal.
- Anything `needs_review` is content the gate held back — open it, fix coverage
  (e.g., re-generate concepts), and publish from the admin tools when good.
