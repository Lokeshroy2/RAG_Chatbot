"""SQLAlchemy models and engine helpers. Postgres in production, SQLite in tests."""

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def normalize_url(url: str) -> str:
    """Providers (Neon included) hand out a bare `postgresql://` URL, which
    SQLAlchemy defaults to the psycopg2 dialect. We install psycopg3 instead,
    so rewrite it to the `+psycopg` driver SQLAlchemy needs to find it."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url
