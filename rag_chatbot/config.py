"""Application settings, overridable via RAG_* environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8001
    ollama_url: str = "http://localhost:11434"  # base URL, no path
    ollama_model: str = "mistral:latest"
    ollama_timeout: int = 180  # seconds
    embed_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800  # target characters per chunk
    chunk_overlap: int = 150  # characters carried over between chunks
    top_k: int = 5  # max chunks sent to the LLM
    score_threshold: float = 0.25  # minimum cosine similarity to count as relevant
    max_file_mb: int = 25
    history_window: int = 6  # messages of history included in the prompt
    history_char_budget: int = 4000  # hard cap on history characters in the prompt
    condense_queries: bool = True  # rewrite follow-ups into standalone questions
    persist: bool = True  # save/load the index across restarts
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    open_browser: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=_str("RAG_HOST", cls.host),
            port=_int("RAG_PORT", cls.port),
            ollama_url=_str("RAG_OLLAMA_URL", cls.ollama_url).rstrip("/"),
            ollama_model=_str("RAG_OLLAMA_MODEL", cls.ollama_model),
            ollama_timeout=_int("RAG_OLLAMA_TIMEOUT", cls.ollama_timeout),
            embed_model_name=_str("RAG_EMBED_MODEL", cls.embed_model_name),
            chunk_size=_int("RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int("RAG_TOP_K", cls.top_k),
            score_threshold=_float("RAG_SCORE_THRESHOLD", cls.score_threshold),
            max_file_mb=_int("RAG_MAX_FILE_MB", cls.max_file_mb),
            history_window=_int("RAG_HISTORY_WINDOW", cls.history_window),
            history_char_budget=_int("RAG_HISTORY_CHAR_BUDGET", cls.history_char_budget),
            condense_queries=_bool("RAG_CONDENSE_QUERIES", cls.condense_queries),
            persist=_bool("RAG_PERSIST", cls.persist),
            data_dir=Path(_str("RAG_DATA_DIR", str(cls.data_dir))),
            open_browser=_bool("RAG_OPEN_BROWSER", cls.open_browser),
        )


settings = Settings.from_env()
