import time

import jwt as pyjwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_chatbot import auth as auth_module
from rag_chatbot.auth import UserStore, create_access_token, get_current_user
from rag_chatbot.db import UserRecord


def creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def make_store() -> UserStore:
    """In-memory SQLite stands in for Postgres — same ORM code path, no network DB
    needed for unit tests. Doesn't exercise Postgres-specific concurrent-write
    behavior, but that's the same tradeoff already accepted for FAISS in CI."""
    return UserStore(create_engine("sqlite:///:memory:"))


# ── UserStore ────────────────────────────────────────────────────────────────


def test_create_and_verify_user():
    store = make_store()
    user_id = store.create("alice", "hunter22")
    assert store.verify("alice", "hunter22") == user_id


def test_verify_rejects_wrong_password():
    store = make_store()
    store.create("alice", "hunter22")
    assert store.verify("alice", "wrong-password") is None


def test_verify_rejects_unknown_user():
    store = make_store()
    assert store.verify("ghost", "whatever") is None


def test_duplicate_username_rejected():
    store = make_store()
    store.create("alice", "hunter22")
    with pytest.raises(ValueError):
        store.create("alice", "different-password")


def test_password_is_hashed_not_stored_in_plaintext():
    store = make_store()
    store.create("alice", "hunter22")
    with Session(store.engine) as session:
        record = session.scalar(select(UserRecord).where(UserRecord.username == "alice"))
    assert "hunter22" not in record.password_hash


def test_persistence_roundtrip(tmp_path):
    # file-based SQLite (not :memory:) so two separate UserStore instances
    # share the same underlying data — stands in for "restart the process".
    url = f"sqlite:///{tmp_path / 'test.db'}"
    store = UserStore(create_engine(url))
    user_id = store.create("alice", "hunter22")

    restored = UserStore(create_engine(url))
    assert restored.verify("alice", "hunter22") == user_id


# ── tokens ───────────────────────────────────────────────────────────────────


def test_access_token_round_trip():
    token = create_access_token("user-123")
    assert get_current_user(creds(token)) == "user-123"


def test_missing_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(None)
    assert exc_info.value.status_code == 401


def test_invalid_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(creds("not-a-real-token"))
    assert exc_info.value.status_code == 401


def test_expired_token_raises_401():
    payload = {"sub": "user-123", "exp": time.time() - 10}
    token = pyjwt.encode(payload, auth_module._SECRET, algorithm=auth_module.JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(creds(token))
    assert exc_info.value.status_code == 401
