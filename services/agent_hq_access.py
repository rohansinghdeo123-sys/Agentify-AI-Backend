"""Identity and role enforcement for the executive workspace."""

from typing import Any

from fastapi import Depends, HTTPException

from app import config
from app.security import is_backend_admin, normalize_identity, verify_firebase_user

_EXECUTIVE_NAMES = {"rohan": "Rohan Singh Deo", "amit": "Amit"}


def list_agent_hq_executives() -> list[dict[str, str]]:
    return [
        {"id": executive_id, "name": name, "role": "Co-CEO"}
        for executive_id, name in _EXECUTIVE_NAMES.items()
    ]


def agent_hq_executive_identity(token: dict[str, Any]) -> dict[str, str] | None:
    uid = token.get("uid")
    if not isinstance(uid, str) or not uid or not is_backend_admin(token):
        return None

    email = normalize_identity(token.get("email")) if token.get("email_verified") is True else ""
    matches = []
    for executive in list_agent_hq_executives():
        executive_id = executive["id"]
        pinned = config.AGENT_HQ_EXECUTIVE_UIDS.get(executive_id)
        account_matches = uid == pinned if pinned else email == config.BUILTIN_FOUNDER_IDENTITIES[executive_id]
        if account_matches:
            matches.append(executive)

    # Ambiguous account configuration must fail closed.
    return matches[0] if len(matches) == 1 else None


def require_agent_hq_executive(
    decoded_token: dict[str, Any] = Depends(verify_firebase_user),
) -> dict[str, Any]:
    executive = agent_hq_executive_identity(decoded_token)
    if executive is None:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        **decoded_token,
        "executive_id": executive["id"],
        "executive_name": executive["name"],
    }
