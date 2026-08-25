# ================= ENV BOOTSTRAP =================
import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ================= IMPORTS =================
import logging
import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import config
from app.lifespan import lifespan
from app.rate_limit import client_rate_key, minute_limit_for_path, rate_limiter
from routers import (
    admin,
    agent,
    coach,
    exam_papers,
    health,
    planning,
    profile,
    progress,
    rivals,
    study,
    written_practice,
)

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_educator.main")

# ================= APP INIT =================
app = FastAPI(
    title="AI Educator Backend - Agentic v2.0 + Secure Admin + Coach API",
    lifespan=lifespan,
)

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-ms", "Retry-After"],
)


# ================= GLOBAL EXCEPTION HANDLER =================
def _cors_headers_for(request: Request) -> dict:
    """CORS headers for handler responses: the 500 handler runs outside
    CORSMiddleware, so without these the browser hides the error body."""
    origin = request.headers.get("origin", "")
    if origin and (origin in config.ALLOWED_ORIGINS or "*" in config.ALLOWED_ORIGINS):
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    return {}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "") or uuid.uuid4().hex
    logger.exception(
        "Unhandled error | request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error. Please retry or contact support with the request id.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id, **_cors_headers_for(request)},
    )


# ================= REQUEST GUARDRAILS =================
@app.middleware("http")
async def production_guardrails(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id

    if config.RATE_LIMIT_ENABLED and request.method.upper() != "OPTIONS":
        limit = minute_limit_for_path(request.url.path)
        allowed, retry_after = rate_limiter.allow(client_rate_key(request), limit, 60)
        if not allowed:
            # This early return runs outside CORSMiddleware, so without these
            # headers the browser hides the 429 and reports a generic
            # "Failed to fetch" instead of surfacing the rate-limit.
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Too many requests. Please slow down and try again.",
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "Retry-After": str(retry_after),
                    **_cors_headers_for(request),
                },
            )

    started = time.perf_counter()
    # Unhandled exceptions propagate to unhandled_exception_handler, which
    # logs them with request context and returns a sanitized 500.
    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = str(round((time.perf_counter() - started) * 1000, 2))
    return response


# ================= ROUTERS =================
app.include_router(study.router)
app.include_router(agent.router)
app.include_router(coach.router)
app.include_router(health.router)
app.include_router(admin.router)
app.include_router(progress.router)
app.include_router(planning.router)
app.include_router(profile.router)
app.include_router(exam_papers.router)
app.include_router(written_practice.router)
app.include_router(rivals.router)


# ================= BACKWARD-COMPAT RE-EXPORTS =================
# These names were previously defined in this module. They are re-exported so
# existing imports (`from main import ...`) and tests keep working after the
# move to app/, routers/, and services/.
from app.request_models import GenerateMCQRequest, SectionAIRequest  # noqa: E402,F401
from app.security import (  # noqa: E402,F401
    require_owned_study_session,
    session_id_belongs_to_user,
)
from app.serializers import (  # noqa: E402,F401
    format_test_session,
    group_conversation_rows as _group_conversation_rows,
    serialize_coach_conversation as _serialize_coach_conversation,
)
from services.coach_service import (  # noqa: E402,F401
    conversation_rows_for_user as _conversation_rows_for_user,
)
