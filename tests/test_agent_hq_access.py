from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from app import config
from app.security import verify_firebase_user
from services.agent_hq_access import agent_hq_executive_identity, require_agent_hq_executive


@pytest.fixture(autouse=True)
def unpin_executives(monkeypatch):
    monkeypatch.setattr(config, "AGENT_HQ_EXECUTIVE_UIDS", {"rohan": "", "amit": ""})


@pytest.mark.parametrize("executive_id", ["rohan", "amit"])
def test_verified_co_ceos_have_equal_executive_access(executive_id):
    token = {
        "uid": f"verified-{executive_id}",
        "email": config.BUILTIN_FOUNDER_IDENTITIES[executive_id].upper(),
        "email_verified": True,
    }
    assert agent_hq_executive_identity(token)["role"] == "Co-CEO"
    assert require_agent_hq_executive(token)["executive_id"] == executive_id


@pytest.mark.parametrize("token", [
    {"uid": "ordinary", "name": "Rohan Singh Deo"},
    {"uid": "ordinary", "email": config.BUILTIN_FOUNDER_IDENTITIES["amit"], "email_verified": False},
    {"uid": "ordinary", "admin": True},
    {"uid": "ordinary", "roles": ["admin", "founder"]},
    {"email": config.BUILTIN_FOUNDER_IDENTITIES["rohan"], "email_verified": True},
])
def test_unmapped_or_unverified_identity_cannot_open_executive_endpoint(token):
    app = FastAPI()

    @app.get("/executive")
    def protected(executive=Depends(require_agent_hq_executive)):
        return {"id": executive["executive_id"]}

    app.dependency_overrides[verify_firebase_user] = lambda: token
    with TestClient(app) as client:
        assert client.get("/executive").status_code == 404


def test_uid_pin_is_exact_replaces_email_and_requires_admin(monkeypatch):
    monkeypatch.setattr(config, "AGENT_HQ_EXECUTIVE_UIDS", {"rohan": "CaseSensitiveUID", "amit": ""})
    displaced_email_token = {
        "uid": "old-uid",
        "email": config.BUILTIN_FOUNDER_IDENTITIES["rohan"],
        "email_verified": True,
    }
    assert agent_hq_executive_identity(displaced_email_token) is None
    assert agent_hq_executive_identity({"uid": "casesensitiveuid", "admin": True}) is None
    assert agent_hq_executive_identity({"uid": "CaseSensitiveUID"}) is None
    assert agent_hq_executive_identity({"uid": "CaseSensitiveUID", "admin": True})["id"] == "rohan"


def test_duplicate_uid_mapping_fails_closed(monkeypatch):
    monkeypatch.setattr(config, "AGENT_HQ_EXECUTIVE_UIDS", {"rohan": "duplicate-uid", "amit": "duplicate-uid"})
    assert agent_hq_executive_identity({"uid": "duplicate-uid", "admin": True}) is None
