"""Password hashing, JWT issuance/verification, and a Postgres-backed user store.

Framework-agnostic except for `get_current_user`, which is a ready-to-use
FastAPI dependency.
"""

import logging
import secrets
import time
import uuid

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rag_chatbot.config import settings
from rag_chatbot.db import Base, UserRecord

log = logging.getLogger("rag.auth")

JWT_ALGORITHM = "HS256"

if settings.jwt_secret:
    _SECRET = settings.jwt_secret
else:
    _SECRET = secrets.token_hex(32)
    log.warning(
        "RAG_JWT_SECRET is not set — using a random secret generated for this process. "
        "Every restart invalidates all existing tokens. Set RAG_JWT_SECRET before deploying."
    )

_bearer_scheme = HTTPBearer(auto_error=False)


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserStore:
    """Postgres-backed user store (SQLite works too, e.g. in tests).

    Username uniqueness is enforced by the database's UNIQUE constraint, not
    by Python — a threading.Lock only serializes one process; the DB
    constraint is correct no matter how many worker processes are running.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        Base.metadata.create_all(engine)

    def create(self, username: str, password: str) -> str:
        user_id = uuid.uuid4().hex[:12]
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with Session(self.engine) as session:
            session.add(UserRecord(id=user_id, username=username, password_hash=password_hash))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise ValueError("username already taken") from None
        return user_id

    def verify(self, username: str, password: str) -> str | None:
        with Session(self.engine) as session:
            record = session.scalar(select(UserRecord).where(UserRecord.username == username))
        if record is None:
            return None
        if not bcrypt.checkpw(password.encode(), record.password_hash.encode()):
            return None
        return record.id


def create_access_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": time.time() + settings.jwt_expire_minutes * 60}
    return jwt.encode(payload, _SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency: returns the caller's user_id, or raises 401."""
    if creds is None:
        raise HTTPException(401, "Missing bearer token")
    try:
        payload = jwt.decode(creds.credentials, _SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token") from None
    return payload["sub"]
