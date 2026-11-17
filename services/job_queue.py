"""Durable DB-backed background job queue.

Jobs are rows in the existing ``content_ingestion_jobs`` table, so they survive
restarts: queued jobs resume after a crash, and jobs interrupted mid-run are
marked failed on startup. A single daemon worker thread per process claims and
executes jobs with short-lived database sessions.

This intentionally avoids new infrastructure (Redis/Celery). If job volume ever
needs parallel workers across processes, the endpoint contract here (submit ->
job_id -> poll status) maps directly onto Arq/Celery later.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from database import SessionLocal
from models import ContentIngestionJob

logger = logging.getLogger("ai_educator.job_queue")

JobHandler = Callable[[Session, ContentIngestionJob], Optional[Dict[str, Any]]]


class DbJobQueue:
    def __init__(self, session_factory: Callable[[], Session] = SessionLocal, poll_seconds: float = 2.0) -> None:
        self._session_factory = session_factory
        self._poll_seconds = poll_seconds
        self._handlers: Dict[str, JobHandler] = {}
        self._leased_job_types: set[str] = set()
        self._maintenance: dict[str, tuple[float, Callable[[Session], None]]] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register(self, job_type: str, handler: JobHandler, *, leased: bool = False) -> None:
        self._handlers[job_type] = handler
        if leased:
            self._leased_job_types.add(job_type)

    def register_maintenance(self, name: str, interval: float, handler: Callable[[Session], None]) -> None:
        self._maintenance[name] = (interval, handler)

    def wake(self) -> None:
        self._wake.set()

    def submit(
        self,
        db: Session,
        *,
        job_type: str,
        source_path: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> ContentIngestionJob:
        """Create a queued job row and wake the worker."""
        if job_type not in self._handlers:
            raise ValueError(f"No handler registered for job type '{job_type}'")
        job = ContentIngestionJob(
            job_id=f"content_job_{uuid.uuid4().hex[:14]}",
            job_type=job_type,
            status="queued",
            source_path=source_path,
            summary={"request": payload or {}},
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        self._wake.set()
        return job

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._recover_interrupted_jobs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="job-queue-worker", daemon=True)
        self._thread.start()
        logger.info("Job queue worker started (handlers: %s)", sorted(self._handlers))

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("Job queue worker stopped")

    def _recover_interrupted_jobs(self) -> None:
        """Jobs left 'running' by a crashed process can't be resumed safely."""
        db = self._session_factory()
        try:
            stale = (
                db.query(ContentIngestionJob)
                .filter(
                    ContentIngestionJob.status == "running",
                    ContentIngestionJob.job_type.in_(set(self._handlers) - self._leased_job_types),
                )
                .all()
            )
            for job in stale:
                job.status = "failed"
                job.error = "Interrupted by a backend restart before completion."
                job.updated_at = datetime.utcnow()
            if stale:
                db.commit()
                logger.warning("Marked %d interrupted job(s) as failed", len(stale))
        except Exception as exc:
            db.rollback()
            logger.warning("Could not recover interrupted jobs: %s", exc)
        finally:
            db.close()

    # ── worker ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        due: dict[str, float] = {}
        while not self._stop.is_set():
            for name, (interval, handler) in self._maintenance.items():
                now = time.monotonic()
                if due.get(name, 0) <= now:
                    due[name] = now + interval
                    try:
                        with self._session_factory() as db:
                            handler(db)
                    except Exception:
                        logger.exception("Queue maintenance failed for %s", name)
            job_id = self._claim_next()
            if job_id is None:
                self._wake.wait(timeout=self._poll_seconds)
                self._wake.clear()
                continue
            self._execute(job_id)

    def _claim_next(self) -> Optional[str]:
        db = self._session_factory()
        try:
            job = (
                db.query(ContentIngestionJob)
                .filter(
                    ContentIngestionJob.status == "queued",
                    ContentIngestionJob.job_type.in_(list(self._handlers)),
                )
                .order_by(ContentIngestionJob.id.asc())
                .first()
            )
            if job is None:
                return None
            changed = (
                db.query(ContentIngestionJob)
                .filter_by(id=job.id, status="queued")
                .update({"status": "running", "updated_at": datetime.utcnow()}, synchronize_session=False)
            )
            job_id = job.job_id
            db.commit()
            return job_id if changed else None
        except Exception as exc:
            db.rollback()
            logger.warning("Job claim failed: %s", exc)
            return None
        finally:
            db.close()

    def _execute(self, job_id: str) -> None:
        db = self._session_factory()
        try:
            job = db.query(ContentIngestionJob).filter(ContentIngestionJob.job_id == job_id).first()
            if job is None:
                return
            handler = self._handlers.get(job.job_type)
            if handler is None:
                job.status = "failed"
                job.error = f"No handler registered for job type '{job.job_type}'."
                db.commit()
                return
            logger.info("Job %s (%s) started", job.job_id, job.job_type)
            try:
                result = handler(db, job)
                db.refresh(job)
                if result and job.status != "failed":
                    job.summary = {**(job.summary or {}), "result": result}
                if job.status == "running":
                    job.status = "completed"
                job.updated_at = datetime.utcnow()
                db.commit()
                logger.info("Job %s finished with status %s", job.job_id, job.status)
            except Exception as exc:
                db.rollback()
                job = db.query(ContentIngestionJob).filter(ContentIngestionJob.job_id == job_id).first()
                if job is not None:
                    job.status = "failed"
                    job.error = str(exc)[:2000]
                    job.updated_at = datetime.utcnow()
                    db.commit()
                logger.exception("Job %s failed", job_id)
        except Exception as exc:
            db.rollback()
            logger.warning("Job execution wrapper failed for %s: %s", job_id, exc)
        finally:
            db.close()


# ── default handlers ────────────────────────────────────────────────────
def _handle_ingest_folder(db: Session, job: ContentIngestionJob) -> Optional[Dict[str, Any]]:
    from Logic.content_pipeline import run_ingest_folder_job

    request = (job.summary or {}).get("request") or {}
    run_ingest_folder_job(
        db,
        job,
        root_path=request.get("root_path"),
        replace=bool(request.get("replace_existing_extraction", True)),
    )
    return None  # job status/summary already set by run_ingest_folder_job


def _handle_generate_concepts(db: Session, job: ContentIngestionJob) -> Dict[str, Any]:
    from Logic.content_pipeline import generate_concepts_for_chapter, serialize_chapter

    request = (job.summary or {}).get("request") or {}
    chapter = generate_concepts_for_chapter(
        db,
        int(request["chapter_id"]),
        replace=bool(request.get("replace_existing", True)),
        max_batch_chars=int(request.get("max_batch_chars", 9000)),
    )
    db.commit()
    return {"chapter": serialize_chapter(chapter)}


def _handle_embed_chunks(db: Session, job: ContentIngestionJob) -> Dict[str, Any]:
    from Logic.content_pipeline import embed_missing_chunks

    request = (job.summary or {}).get("request") or {}
    chapter_id = request.get("chapter_id")
    result = embed_missing_chunks(db, chapter_id=int(chapter_id) if chapter_id else None)
    db.commit()
    return {"embedding": result}


job_queue = DbJobQueue()
job_queue.register("ingest_folder", _handle_ingest_folder)
job_queue.register("generate_concepts", _handle_generate_concepts)
job_queue.register("embed_chunks", _handle_embed_chunks)
