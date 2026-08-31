"""Firebase Admin initialization, token verification, RBAC, and quota enforcement.

Firebase readiness is mutated at application startup (see ``app.lifespan``), so
callers must read it through :func:`firebase_ready` rather than binding the value
at import time.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from hashlib import sha256
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from app import config
from app.rate_limit import daily_quotas
from database import SessionLocal
from models import DailyQuotaUsage

logger = logging.getLogger("ai_educator.security")

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    from firebase_admin import credentials
except Exception as firebase_import_error:  # pragma: no cover - optional dependency guard
    firebase_admin = None
    firebase_auth = None
    credentials = None
    FIREBASE_IMPORT_ERROR = firebase_import_error
else:
    FIREBASE_IMPORT_ERROR = None


# ================= FIREBASE STATE =================
FIREBASE_ADMIN_READY = False
FIREBASE_ADMIN_ERROR: Optional[str] = None


def firebase_ready() -> bool:
    return bool(FIREBASE_ADMIN_READY)


def firebase_error() -> Optional[str]:
    return FIREBASE_ADMIN_ERROR


def initialize_firebase_admin() -> None:
    global FIREBASE_ADMIN_READY, FIREBASE_ADMIN_ERROR

    if firebase_admin is None or credentials is None:
        FIREBASE_ADMIN_READY = False
        FIREBASE_ADMIN_ERROR = f"firebase_admin import failed: {FIREBASE_IMPORT_ERROR}"
        logger.warning(FIREBASE_ADMIN_ERROR)
        return

    try:
        if firebase_admin._apps:
            FIREBASE_ADMIN_READY = True
            FIREBASE_ADMIN_ERROR = None
            return

        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
        app_options = {"projectId": project_id} if project_id else None

        if service_account:
            if service_account.startswith("{"):
                cred = credentials.Certificate(json.loads(service_account))
            else:
                cred = credentials.Certificate(service_account)

            firebase_admin.initialize_app(cred, app_options)
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
            firebase_admin.initialize_app(cred, app_options)
        else:
            firebase_admin.initialize_app(options=app_options)

        FIREBASE_ADMIN_READY = True
        FIREBASE_ADMIN_ERROR = None
        logger.info("Firebase Admin initialized successfully")
    except Exception as exc:
        FIREBASE_ADMIN_READY = False
        FIREBASE_ADMIN_ERROR = str(exc)
        logger.warning("Firebase Admin not initialized: %s", FIREBASE_ADMIN_ERROR)


# ================= TOKEN VERIFICATION =================
def get_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )

    return token.strip()


def verify_firebase_user(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if not FIREBASE_ADMIN_READY or firebase_auth is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase Admin is not configured on backend",
        )

    token = get_bearer_token(authorization)

    try:
        return firebase_auth.verify_id_token(token, check_revoked=True)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
        )


# ================= ROLE / OWNERSHIP =================
def normalize_identity(value: Any) -> str:
    """Normalize verified Firebase identity values for exact allow-list checks."""

    return str(value or "").strip().casefold()


def has_admin_claim(decoded_token: Dict[str, Any]) -> bool:
    if decoded_token.get("admin") is True:
        return True

    if decoded_token.get("role") == "admin":
        return True

    roles = decoded_token.get("roles")
    return isinstance(roles, list) and "admin" in roles


def has_founder_claim(decoded_token: Dict[str, Any]) -> bool:
    if decoded_token.get("founder") is True or decoded_token.get("founderAdmin") is True:
        return True

    roles = decoded_token.get("roles")
    return isinstance(roles, list) and "founder" in roles


def is_backend_admin(decoded_token: Dict[str, Any]) -> bool:
    if has_admin_claim(decoded_token) or has_founder_claim(decoded_token):
        return True

    uid = normalize_identity(decoded_token.get("uid"))
    email = normalize_identity(decoded_token.get("email"))
    phone = normalize_identity(decoded_token.get("phone_number"))

    return (
        uid in config.BACKEND_ADMIN_UIDS
        or email in config.BACKEND_ADMIN_EMAILS
        # Founder-console identities are a strict admin superset. Treating the
        # founder allow-list as additive keeps a founder-only deployment
        # configuration usable while the backend remains the authority for
        # every protected admin operation.
        or email in config.BACKEND_FOUNDER_ADMIN_EMAILS
        or phone in config.BACKEND_ADMIN_PHONES
    )


def require_admin(
    decoded_token: Dict[str, Any] = Depends(verify_firebase_user),
) -> Dict[str, Any]:
    if not is_backend_admin(decoded_token):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    return decoded_token


def is_founder_admin(decoded_token: Dict[str, Any]) -> bool:
    if has_founder_claim(decoded_token):
        return True

    email = normalize_identity(decoded_token.get("email"))
    return bool(email and email in config.BACKEND_FOUNDER_ADMIN_EMAILS and is_backend_admin(decoded_token))


def require_founder_admin(
    decoded_token: Dict[str, Any] = Depends(verify_firebase_user),
) -> Dict[str, Any]:
    if not is_founder_admin(decoded_token):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    return decoded_token


def require_same_user_or_admin(
    user_id: str,
    decoded_token: Dict[str, Any],
) -> None:
    token_uid = str(decoded_token.get("uid", "")).strip()
    target_uid = str(user_id or "").strip()

    if token_uid and (token_uid == target_uid or is_backend_admin(decoded_token)):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not allowed for this user",
    )


def require_authenticated_user_id(decoded_token: Dict[str, Any]) -> str:
    token_uid = str(decoded_token.get("uid", "")).strip()

    if not token_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user id missing from token",
        )

    return token_uid


def session_id_belongs_to_user(session_id: str, user_id: str) -> bool:
    session = str(session_id or "").strip()
    uid = str(user_id or "").strip()
    if not session or not uid:
        return False

    owned_prefixes = (
        f"coach-{uid}-",
        f"coach_{uid}_",
        f"revision-{uid}-",
        f"exam-{uid}-",
        f"probable-{uid}-",
        f"autonomous-{uid}-",
        f"widget-{uid}",
    )
    owned_exact = {
        uid,
        f"coach-{uid}",
        f"coach_{uid}",
        f"widget-{uid}",
    }

    return session in owned_exact or any(session.startswith(prefix) for prefix in owned_prefixes)


def require_owned_study_session(session_id: str, decoded_token: Dict[str, Any]) -> str:
    user_id = require_authenticated_user_id(decoded_token)
    if is_backend_admin(decoded_token):
        return user_id
    if session_id_belongs_to_user(session_id, user_id):
        return user_id

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not allowed for this study session",
    )


# ================= QUOTA ENFORCEMENT =================
def consume_persistent_daily_quota(user_id: str, quota_name: str, limit: int) -> tuple[bool, int]:
    if limit <= 0:
        return True, 0

    quota_date = date.today()
    user_hash = sha256(str(user_id).encode("utf-8")).hexdigest()[:16]
    quota_key = f"{quota_date.isoformat()}:{quota_name}:{user_hash}"
    db = SessionLocal()
    try:
        usage = db.query(DailyQuotaUsage).filter(DailyQuotaUsage.quota_key == quota_key).first()
        if usage is None:
            usage = DailyQuotaUsage(
                quota_key=quota_key,
                user_hash=user_hash,
                quota_name=quota_name,
                quota_date=quota_date,
                count=1,
            )
            db.add(usage)
            db.commit()
            return True, 1

        current = int(usage.count or 0)
        if current >= limit:
            return False, current

        usage.count = current + 1
        usage.updated_at = datetime.utcnow()
        db.commit()
        return True, usage.count
    except Exception as exc:
        db.rollback()
        logger.warning("QUOTA: Persistent quota unavailable, using in-memory fallback: %s", exc)
        return daily_quotas.consume(user_id, quota_name, limit)
    finally:
        db.close()


def enforce_user_quota(user_id: str, quota_name: str) -> None:
    limit = config.QUOTA_LIMITS.get(quota_name, config.AI_DAILY_QUOTA_PER_USER)
    allowed, used = consume_persistent_daily_quota(user_id, quota_name, limit)
    if allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Daily {quota_name} quota reached. Please continue after the quota resets.",
        headers={"Retry-After": "86400"},
    )
