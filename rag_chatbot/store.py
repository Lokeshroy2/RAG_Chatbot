"""Thread-safe multi-document vector store with FAISS (NumPy fallback) and
optional on-disk persistence.

Framework-agnostic: raises KeyError for unknown documents; the API layer maps
that to HTTP errors.
"""

import json
import logging
import os
import threading
import uuid
from pathlib import Path

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

log = logging.getLogger("rag.store")

_STORE_VERSION = 1


class VectorStore:
    """Per-chunk metadata + FAISS inner-product index (NumPy fallback)."""

    def __init__(self, dim: int, persist_dir: Path | None = None):
        self.dim = dim
        self.lock = threading.Lock()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.docs: dict = {}  # doc_id -> {"filename", "enabled", "chunks", "chars"}
        self.meta: list[dict] = []  # per chunk: {"doc_id", "filename", "text"}
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.index = faiss.IndexFlatIP(dim) if HAS_FAISS else None
        if self.persist_dir:
            self._load()

    # ── mutations ────────────────────────────────────────────────────────────

    def add_document(self, filename: str, chunks: list[str], vecs: np.ndarray) -> str:
        with self.lock:
            doc_id = uuid.uuid4().hex[:8]
            self.docs[doc_id] = {
                "filename": filename,
                "enabled": True,
                "chunks": len(chunks),
                "chars": sum(len(c) for c in chunks),
            }
            self.meta.extend(
                {"doc_id": doc_id, "filename": filename, "text": c} for c in chunks
            )
            self.embeddings = np.vstack([self.embeddings, vecs])
            if HAS_FAISS:
                self.index.add(vecs)
            self._save()
            return doc_id

    def remove_document(self, doc_id: str):
        with self.lock:
            if doc_id not in self.docs:
                raise KeyError(doc_id)
            keep = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
            self.meta = [self.meta[i] for i in keep]
            self.embeddings = (
                self.embeddings[keep] if keep else np.zeros((0, self.dim), np.float32)
            )
            del self.docs[doc_id]
            self._rebuild_index()
            self._save()

    def set_enabled(self, doc_id: str, enabled: bool):
        with self.lock:
            if doc_id not in self.docs:
                raise KeyError(doc_id)
            self.docs[doc_id]["enabled"] = enabled
            self._save()

    def clear(self):
        with self.lock:
            self.docs.clear()
            self.meta.clear()
            self.embeddings = np.zeros((0, self.dim), np.float32)
            self._rebuild_index()
            self._save()

    # ── queries ──────────────────────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        with self.lock:
            return [{"id": k, **v} for k, v in self.docs.items()]

    def search(
        self, q_vec: np.ndarray, top_k: int = 5, score_threshold: float = 0.25
    ) -> list[dict]:
        """Top chunks from enabled documents scoring above the threshold."""
        with self.lock:
            enabled_rows = [
                i for i, m in enumerate(self.meta) if self.docs[m["doc_id"]]["enabled"]
            ]
            if not enabled_rows:
                return []
            if HAS_FAISS and len(enabled_rows) == len(self.meta) and self.index.ntotal:
                k = min(len(self.meta), top_k)
                scores, ids = self.index.search(q_vec[None, :].astype(np.float32), k)
                ranked = [
                    (int(i), float(s))
                    for i, s in zip(ids[0].tolist(), scores[0].tolist(), strict=True)
                    if i >= 0
                ]
            else:
                # some docs disabled (or no FAISS): brute-force over the enabled subset
                sub = self.embeddings[enabled_rows]
                scores = sub @ q_vec
                order = np.argsort(scores)[::-1][:top_k]
                ranked = [(enabled_rows[int(i)], float(scores[i])) for i in order]

            return [
                {
                    "filename": self.meta[i]["filename"],
                    "text": self.meta[i]["text"],
                    "score": round(score, 4),
                }
                for i, score in ranked
                if score >= score_threshold
            ][:top_k]

    # ── persistence ──────────────────────────────────────────────────────────

    def _rebuild_index(self):
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            if len(self.embeddings):
                self.index.add(self.embeddings)

    def _save(self):
        """Persist state atomically. Caller must hold self.lock."""
        if not self.persist_dir:
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "version": _STORE_VERSION,
                "dim": self.dim,
                "docs": self.docs,
                "meta": self.meta,
            }
            tmp_json = self.persist_dir / "store.json.tmp"
            tmp_json.write_text(json.dumps(state), encoding="utf-8")
            os.replace(tmp_json, self.persist_dir / "store.json")
            tmp_npy = self.persist_dir / "embeddings.npy.tmp"
            with open(tmp_npy, "wb") as f:
                np.save(f, self.embeddings)
            os.replace(tmp_npy, self.persist_dir / "embeddings.npy")
        except OSError as e:
            log.warning("Could not persist vector store: %s", e)

    def _load(self):
        json_path = self.persist_dir / "store.json"
        npy_path = self.persist_dir / "embeddings.npy"
        if not (json_path.exists() and npy_path.exists()):
            return
        try:
            state = json.loads(json_path.read_text(encoding="utf-8"))
            if state.get("dim") != self.dim:
                log.warning(
                    "Persisted index has dim=%s but model produces dim=%d — ignoring it. "
                    "Delete %s to silence this warning.",
                    state.get("dim"), self.dim, self.persist_dir,
                )
                return
            embeddings = np.load(npy_path)
            if len(embeddings) != len(state["meta"]):
                log.warning("Persisted store is inconsistent — ignoring it.")
                return
            self.docs = state["docs"]
            self.meta = state["meta"]
            self.embeddings = embeddings.astype(np.float32)
            self._rebuild_index()
            log.info(
                "Restored %d document(s) / %d chunk(s) from %s",
                len(self.docs), len(self.meta), self.persist_dir,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            log.warning("Could not load persisted vector store: %s", e)
