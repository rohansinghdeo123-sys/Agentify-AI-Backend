"""Environment-backed configuration for the AI Educator backend.

All values are read once at import time. `.env` is loaded by ``main`` before this
module is imported, so ``os.getenv`` here observes the configured environment.
"""

from __future__ import annotations

import os
from typing import Dict, List


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_env(name: str) -> set[str]:
    return {
        item.strip().lower()
        for item in os.getenv(name, "").split(",")
        if item.strip()
    }


# ================= CORS =================
ALLOWED_ORIGINS = env_list(
    "ALLOWED_ORIGINS",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://agentifyai.in",
        "https://www.agentifyai.in",
    ],
)

# ================= RATE LIMITS & QUOTAS =================
RATE_LIMIT_ENABLED = env_bool("RATE_LIMIT_ENABLED", True)
# When behind a reverse proxy (Render/Heroku), key unauthenticated rate limits
# on the proxy-appended (last) X-Forwarded-For hop. Disable for direct exposure.
TRUST_PROXY_HEADERS = env_bool("TRUST_PROXY_HEADERS", True)
RATE_LIMIT_PER_MINUTE = env_int("RATE_LIMIT_PER_MINUTE", 120)
AI_RATE_LIMIT_PER_MINUTE = env_int("AI_RATE_LIMIT_PER_MINUTE", 24)
ADMIN_RATE_LIMIT_PER_MINUTE = env_int("ADMIN_RATE_LIMIT_PER_MINUTE", 180)
AI_DAILY_QUOTA_PER_USER = env_int("AI_DAILY_QUOTA_PER_USER", 180)
EXAM_DAILY_QUOTA_PER_USER = env_int("EXAM_DAILY_QUOTA_PER_USER", 80)
ARTIFACT_DAILY_QUOTA_PER_USER = env_int("ARTIFACT_DAILY_QUOTA_PER_USER", 40)
# Exam-intelligence quotas: paper uploads each run a parse + analysis LLM pass,
# and each written-answer submission runs a teacher-evaluation LLM pass.
EXAM_PAPER_DAILY_QUOTA_PER_USER = env_int("EXAM_PAPER_DAILY_QUOTA_PER_USER", 30)
WRITTEN_PRACTICE_DAILY_QUOTA_PER_USER = env_int("WRITTEN_PRACTICE_DAILY_QUOTA_PER_USER", 120)

# ================= EXAM PAPER UPLOAD LIMITS =================
# Max accepted upload size in bytes (default 8 MB). Enforced while reading the
# stream so an oversize upload is rejected before it is buffered fully.
EXAM_UPLOAD_MAX_BYTES = env_int("EXAM_UPLOAD_MAX_BYTES", 8 * 1024 * 1024)
# Extensions accepted by the parser factory. Image types are accepted at the
# API but routed to the OCR stub until an OCR backend is configured.
EXAM_UPLOAD_ALLOWED_EXTENSIONS = set(
    env_list("EXAM_UPLOAD_ALLOWED_EXTENSIONS", ["pdf", "txt", "png", "jpg", "jpeg"])
)

AI_RATE_LIMIT_PATHS = {
    "/section-ai",
    "/generate-mcqs",
    "/generate-probable-questions",
    "/artifacts/generate",
    "/agent",
    "/coach/chat",
    "/coach/chat/stream",
    "/exam/papers/upload",
    "/exam/pattern/analyze",
    "/exam/probable-questions/generate",
    "/exam/written-practice/question",
    "/exam/written-practice/submit",
}

QUOTA_LIMITS: Dict[str, int] = {
    "coach": AI_DAILY_QUOTA_PER_USER,
    "exam": EXAM_DAILY_QUOTA_PER_USER,
    "artifact": ARTIFACT_DAILY_QUOTA_PER_USER,
    "agent": AI_DAILY_QUOTA_PER_USER,
    "exam_paper": EXAM_PAPER_DAILY_QUOTA_PER_USER,
    "written_practice": WRITTEN_PRACTICE_DAILY_QUOTA_PER_USER,
}

# ================= ADMIN ALLOWLISTS =================
BACKEND_ADMIN_EMAILS = parse_csv_env("BACKEND_ADMIN_EMAILS")
BACKEND_ADMIN_UIDS = parse_csv_env("BACKEND_ADMIN_UIDS")
BACKEND_ADMIN_PHONES = parse_csv_env("BACKEND_ADMIN_PHONES")
# Product-owner identities are deliberately explicit rather than inferred from
# a client-side/Vercel variable. Environment entries remain additive, so
# deployments can grant further founder-console access without changing code.
BUILTIN_FOUNDER_ADMIN_EMAILS = {
    "amit.kumarmunda4@gmail.com",
    "rohan.singhdeo123@gmail.com",
}
BACKEND_FOUNDER_ADMIN_EMAILS = (
    BUILTIN_FOUNDER_ADMIN_EMAILS
    | BACKEND_ADMIN_EMAILS
    | parse_csv_env("BACKEND_FOUNDER_ADMIN_EMAILS")
)
