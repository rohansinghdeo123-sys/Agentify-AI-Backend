"""Automate the whole study-content pipeline from NCERT to published.

Downloads NCERT chapter PDFs and runs ingest -> generate concepts -> embed ->
auto-publish (only chapters that pass the quality gate; the rest are left as
``needs_review`` for a human to check in the admin page). Runs slowly and
politely, is resumable, and isolates per-chapter failures.

Full ingestion runs against the configured DATABASE_URL (loads backend/.env)
and requires generation plus embedding credentials. Download-only mode does
not require database or AI configuration.

Usage (from the backend/ directory):
    python scripts/automate_content.py                       # Class 11 & 12 PCM, auto-publish
    python scripts/automate_content.py --classes 11 --subjects Chemistry
    python scripts/automate_content.py --classes 11 --subjects Chemistry --chapters 3 --no-publish
    python scripts/automate_content.py --download-only       # just fetch PDFs
    python scripts/automate_content.py --preflight-only      # sanitized readiness check
    python scripts/automate_content.py --no-publish          # ingest+generate+embed, manual approve
    python scripts/automate_content.py --delay 6 --max-chapters 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence
from urllib.parse import urlsplit

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BACKEND_DIR, ".env"))
except Exception:
    pass

DEFAULT_CLASSES = ["11", "12"]
DEFAULT_SUBJECTS = ["Physics", "Chemistry", "Maths"]
SUPPORTED_GENERATION_PROVIDERS = ("groq", "openrouter", "openai")


@dataclass(frozen=True)
class PreflightResult:
    """Sanitized readiness result for a content-automation run."""

    ok: bool
    database_target: str
    generation_enabled: bool
    semantic_embeddings_enabled: bool
    lexical_only: bool
    errors: tuple[str, ...] = ()


def _configured(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or "").strip())


def _configured_generation_providers(env: Mapping[str, str]) -> tuple[str, ...]:
    """Return supported providers in the same order used by the model router."""
    raw = (
        env.get("COACH_PROVIDER_ORDER")
        or env.get("COACH_LLM_PROVIDER")
        or "groq"
    )
    providers: list[str] = []
    for item in str(raw).split(","):
        provider = item.strip().lower()
        if provider in SUPPORTED_GENERATION_PROVIDERS and provider not in providers:
            providers.append(provider)
    return tuple(providers)


def _generation_provider_ready(env: Mapping[str, str], provider: str) -> bool:
    if not _configured(env, f"{provider.upper()}_API_KEY"):
        return False
    if provider == "groq":
        # Groq reviewer and fallback models have application defaults.
        return True
    prefix = provider.upper()
    return any(
        _configured(env, variable)
        for variable in (
            f"{prefix}_REVIEW_MODEL",
            f"{prefix}_MODEL",
            f"{prefix}_FALLBACK_MODEL",
        )
    )


def _database_details(database_url: str) -> tuple[str, str]:
    """Return ``(dialect, sanitized target)`` without credentials or paths."""

    value = database_url.strip()
    if not value:
        return "", "not configured"

    try:
        parsed = urlsplit(value)
    except ValueError:
        return "invalid", "invalid database URL"

    dialect = parsed.scheme.split("+", 1)[0].lower()
    if not dialect:
        return "invalid", "invalid database URL"
    if dialect == "sqlite":
        return dialect, "sqlite (local)"

    # ``hostname`` intentionally excludes user info, password, path, and query.
    # Restrict it again before showing it to prevent terminal/log injection.
    try:
        hostname = parsed.hostname or ""
    except ValueError:
        hostname = ""
    safe_host = hostname if re.fullmatch(r"[A-Za-z0-9.-]+", hostname) else ""
    target = f"{dialect} @ {safe_host}" if safe_host else f"{dialect} (configured)"
    return dialect, target


def check_preflight(
    *,
    env: Mapping[str, str] | None = None,
    download_only: bool = False,
    allow_local_db: bool = False,
    allow_lexical_only: bool = False,
) -> PreflightResult:
    """Validate runtime safety without connecting to the DB or AI providers."""

    values = env if env is not None else os.environ
    database_url = (values.get("DATABASE_URL") or "").strip()
    dialect, database_target = _database_details(database_url)

    if download_only:
        return PreflightResult(
            ok=True,
            database_target="not required (download-only)",
            generation_enabled=False,
            semantic_embeddings_enabled=False,
            lexical_only=False,
        )

    errors: list[str] = []
    if not database_url:
        database_target = "sqlite (implicit local opt-in)" if allow_local_db else "not configured"
        if not allow_local_db:
            errors.append(
                "DATABASE_URL is required for ingestion. To intentionally use the local SQLite "
                "database, pass --allow-local-db."
            )
    elif dialect == "sqlite":
        if not allow_local_db:
            errors.append(
                "DATABASE_URL points to SQLite. Refusing to ingest locally unless "
                "--allow-local-db is supplied."
            )
    elif dialect not in {"postgres", "postgresql"}:
        errors.append(
            "DATABASE_URL must use PostgreSQL (or SQLite with --allow-local-db); "
            f"configured dialect is {dialect or 'unknown'}."
        )

    generation_providers = _configured_generation_providers(values)
    generation_enabled = any(
        _generation_provider_ready(values, provider)
        for provider in generation_providers
    )
    if not generation_enabled:
        if generation_providers == ("groq",):
            errors.append("GROQ_API_KEY is required for structured concept generation.")
        else:
            provider_list = ", ".join(generation_providers) or "no supported provider"
            errors.append(
                "No configured structured-generation route is available for "
                f"provider order: {provider_list}. Configure the provider API key "
                "and a REVIEW_MODEL, MODEL, or FALLBACK_MODEL for OpenRouter/OpenAI."
            )

    semantic_embeddings_enabled = (
        _configured(values, "EMBEDDINGS_API_KEY")
        or _configured(values, "OPENAI_API_KEY")
    )
    lexical_only = allow_lexical_only and not semantic_embeddings_enabled
    if not semantic_embeddings_enabled and not allow_lexical_only:
        errors.append(
            "EMBEDDINGS_API_KEY or OPENAI_API_KEY is required for semantic retrieval. "
            "Pass --allow-lexical-only only for an intentional lexical-only run."
        )

    return PreflightResult(
        ok=not errors,
        database_target=database_target,
        generation_enabled=generation_enabled,
        semantic_embeddings_enabled=semantic_embeddings_enabled,
        lexical_only=lexical_only,
        errors=tuple(errors),
    )


def _print_preflight(result: PreflightResult) -> None:
    print("Preflight:")
    print(f"  Database target     : {result.database_target}")
    print(f"  Concept generation  : {'enabled' if result.generation_enabled else 'disabled'}")
    embedding_status = "enabled" if result.semantic_embeddings_enabled else (
        "disabled (lexical-only opt-in)" if result.lexical_only else "disabled"
    )
    print(f"  Semantic embeddings : {embedding_status}")
    print(f"  Result              : {'READY' if result.ok else 'BLOCKED'}")
    for error in result.errors:
        print(f"  ERROR: {error}", file=sys.stderr)


def _load_automation(*, allow_local_db: bool):
    """Import the DB-backed orchestrator only after CLI preflight passes."""

    if allow_local_db:
        # Explicit CLI consent must also reach database.py's import-time guard.
        os.environ["ALLOW_SQLITE_FALLBACK"] = "true"
    from Logic.content_automation import run_automation

    return run_automation


def _csv(value: str, default):
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _chapter_csv(value: str) -> list[int]:
    if not value:
        return []
    chapters: list[int] = []
    for item in value.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            chapter = int(cleaned)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"chapter numbers must be positive integers; received {cleaned!r}"
            ) from exc
        if chapter <= 0:
            raise argparse.ArgumentTypeError(
                f"chapter numbers must be positive integers; received {cleaned!r}"
            )
        chapters.append(chapter)
    return sorted(set(chapters))


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentifyAI NCERT content automation")
    parser.add_argument("--classes", default="", help="comma list, e.g. 11,12 (default both)")
    parser.add_argument("--subjects", default="", help="comma list, e.g. Physics,Chemistry,Maths")
    parser.add_argument(
        "--chapters",
        type=_chapter_csv,
        default=[],
        help="optional exact chapter numbers to process, e.g. 3 or 3,4",
    )
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between downloads (politeness)")
    parser.add_argument("--max-chapters", type=int, default=30, help="max chapters probed per book part")
    parser.add_argument("--download-only", action="store_true", help="only download PDFs, no ingestion")
    parser.add_argument("--no-publish", action="store_true", help="ingest+generate+embed but do not auto-publish")
    parser.add_argument("--no-skip", action="store_true", help="re-download/re-process even if present")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="re-extract selected local PDFs before generation (use after extraction repairs)",
    )
    parser.add_argument(
        "--allow-local-db",
        action="store_true",
        help="explicitly allow ingestion into local SQLite when DATABASE_URL is absent or SQLite",
    )
    parser.add_argument(
        "--allow-lexical-only",
        action="store_true",
        help="allow ingestion without embeddings (semantic retrieval will be unavailable)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate sanitized runtime readiness, then exit without downloading or ingesting",
    )
    parser.add_argument(
        "--generation-batch-delay",
        type=_nonnegative_float,
        default=None,
        help=(
            "seconds between uncached concept-generation calls; defaults to 61s "
            "when Groq is in the configured route"
        ),
    )
    parser.add_argument("--out", default="content_automation_run.json", help="run summary output file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s")

    if args.generation_batch_delay is not None:
        os.environ["CONTENT_GENERATION_BATCH_DELAY_SECONDS"] = str(
            args.generation_batch_delay
        )

    classes = _csv(args.classes, DEFAULT_CLASSES)
    subjects = _csv(args.subjects, DEFAULT_SUBJECTS)
    preflight = check_preflight(
        download_only=args.download_only,
        allow_local_db=args.allow_local_db,
        allow_lexical_only=args.allow_lexical_only,
    )
    _print_preflight(preflight)
    if not preflight.ok:
        return 2
    if args.preflight_only:
        return 0

    generation_delay = os.getenv("CONTENT_GENERATION_BATCH_DELAY_SECONDS") or "provider default"
    print(f"Scope: classes={classes} subjects={subjects} chapters={args.chapters or 'all'} | delay={args.delay}s | "
          f"generation_delay={generation_delay} | download_only={args.download_only} "
          f"auto_publish={not args.no_publish}")

    # Download-only deliberately has no DB/AI configuration prerequisite. A
    # temporary local DB fallback merely allows importing the legacy combined
    # orchestrator; run_automation never opens a DB session in this mode.
    allow_local_import = args.allow_local_db or (
        args.download_only and not (os.getenv("DATABASE_URL") or "").strip()
    )
    run_automation = _load_automation(allow_local_db=allow_local_import)
    summary = run_automation(
        classes=classes,
        subjects=subjects,
        chapter_numbers=args.chapters,
        delay_seconds=args.delay,
        max_chapters=args.max_chapters,
        auto_publish=not args.no_publish,
        download_only=args.download_only,
        skip_existing=not args.no_skip,
        skip_completed=not args.no_skip and not args.reingest,
        reuse_ingest=not args.reingest,
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f" Sources ready : {summary.get('sources_ready', summary['downloaded'])}")
    print(f" Downloaded new: {summary['downloaded']}")
    print(f" Reused existing: {summary.get('reused_sources', 0)}")
    print(f" Published  : {summary['published']}")
    print(f" Needs review: {summary['needs_review']}")
    print(f" Skipped(done): {summary.get('skipped', 0)}")
    print(f" Failed     : {len(summary['failed'])}")
    print(f" Summary written to {args.out}")
    print("=" * 60)
    print(" Monitor everything in the admin page (Operations -> Data & content pipeline)")
    print(" or run: python scripts/content_report.py")
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
