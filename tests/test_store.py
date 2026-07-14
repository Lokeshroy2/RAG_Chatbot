import numpy as np
import pytest

from rag_chatbot.store import VectorStore

DIM = 8


def unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def make_store(persist_dir=None) -> VectorStore:
    return VectorStore(DIM, persist_dir=persist_dir)


def add_doc(store, name: str, axis: int, n_chunks: int = 2) -> str:
    """Document whose chunks all point along one basis axis."""
    chunks = [f"{name} chunk {i}" for i in range(n_chunks)]
    vecs = np.stack([basis(axis)] * n_chunks)
    return store.add_document(name, chunks, vecs)


def test_add_and_list():
    store = make_store()
    doc_id = add_doc(store, "a.txt", axis=0, n_chunks=3)
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id
    assert docs[0]["chunks"] == 3
    assert docs[0]["enabled"] is True


def test_search_returns_most_relevant_chunks():
    store = make_store()
    add_doc(store, "zero.txt", axis=0)
    add_doc(store, "one.txt", axis=1)
    hits = store.search(basis(1), top_k=2, score_threshold=0.25)
    assert hits
    assert all(h["filename"] == "one.txt" for h in hits)
    assert all(h["score"] >= 0.99 for h in hits)


def test_search_respects_score_threshold():
    store = make_store()
    add_doc(store, "zero.txt", axis=0)
    # orthogonal query -> similarity 0 -> nothing above threshold
    assert store.search(basis(1), top_k=5, score_threshold=0.25) == []


def test_disabled_documents_are_excluded():
    store = make_store()
    add_doc(store, "zero.txt", axis=0)
    keep_id = add_doc(store, "also-zero.txt", axis=0)
    for doc in store.list_documents():
        if doc["id"] != keep_id:
            store.set_enabled(doc["id"], False)
    hits = store.search(basis(0), top_k=10, score_threshold=0.25)
    assert hits
    assert all(h["filename"] == "also-zero.txt" for h in hits)


def test_search_finds_enabled_doc_even_among_many_disabled():
    """Regression: disabled docs must not crowd enabled ones out of the fetch window."""
    store = make_store()
    ids = [add_doc(store, f"noise{i}.txt", axis=0, n_chunks=10) for i in range(5)]
    add_doc(store, "signal.txt", axis=0, n_chunks=1)
    for doc_id in ids:
        store.set_enabled(doc_id, False)
    hits = store.search(basis(0), top_k=5, score_threshold=0.25)
    assert hits
    assert all(h["filename"] == "signal.txt" for h in hits)


def test_remove_document():
    store = make_store()
    doc_id = add_doc(store, "a.txt", axis=0)
    add_doc(store, "b.txt", axis=1)
    store.remove_document(doc_id)
    assert len(store.list_documents()) == 1
    assert store.search(basis(0), top_k=5, score_threshold=0.25) == []
    assert store.search(basis(1), top_k=5, score_threshold=0.25)


def test_unknown_document_id_raises_keyerror():
    store = make_store()
    with pytest.raises(KeyError):
        store.remove_document("nope")
    with pytest.raises(KeyError):
        store.set_enabled("nope", False)


def test_clear():
    store = make_store()
    add_doc(store, "a.txt", axis=0)
    store.clear()
    assert store.list_documents() == []
    assert store.search(basis(0), top_k=5, score_threshold=0.0) == []


def test_persistence_roundtrip(tmp_path):
    store = make_store(persist_dir=tmp_path)
    doc_id = add_doc(store, "a.txt", axis=0, n_chunks=3)
    store.set_enabled(doc_id, False)

    restored = make_store(persist_dir=tmp_path)
    docs = restored.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "a.txt"
    assert docs[0]["enabled"] is False
    assert len(restored.meta) == 3
    assert restored.embeddings.shape == (3, DIM)


def test_persisted_dim_mismatch_is_ignored(tmp_path):
    store = make_store(persist_dir=tmp_path)
    add_doc(store, "a.txt", axis=0)
    other = VectorStore(DIM + 1, persist_dir=tmp_path)
    assert other.list_documents() == []
