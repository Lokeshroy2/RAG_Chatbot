"""Lazy-loaded sentence-transformers embeddings.

The model is only loaded on first use, so importing this module (e.g. from
tests or tooling) stays cheap and does not require torch.
"""

import logging

import numpy as np

from .config import settings

log = logging.getLogger("rag.embeddings")

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model (%s)…", settings.embed_model_name)
        try:
            _model = SentenceTransformer(settings.embed_model_name)
        except Exception:
            # no internet / SSL trouble -> use the locally cached copy if present
            log.warning("Could not reach HuggingFace, trying local cache…")
            _model = SentenceTransformer(settings.embed_model_name, local_files_only=True)
        log.info("Embedding model ready (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def embedding_dim() -> int:
    return get_model().get_sentence_embedding_dimension()


def embed_texts(texts: list[str]) -> np.ndarray:
    # normalized embeddings -> inner product == cosine similarity
    vecs = get_model().encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)
