import numpy as np
import pytest

from rag_chatbot import jobs
from rag_chatbot.store import VectorStore

DIM = 8


def fake_embed(chunks):
    return np.ones((len(chunks), DIM), dtype=np.float32)


def test_process_upload_indexes_a_document(monkeypatch):
    store = VectorStore(DIM)
    jobs.configure(store)
    monkeypatch.setattr(jobs.embeddings, "embed_texts", fake_embed)

    result = jobs.process_upload("note.txt", b"Hello world. This is a test document.", "alice")

    assert result["filename"] == "note.txt"
    assert result["chunks"] >= 1
    docs = store.list_documents("alice")
    assert len(docs) == 1
    assert docs[0]["filename"] == "note.txt"


def test_process_upload_is_owner_scoped(monkeypatch):
    store = VectorStore(DIM)
    jobs.configure(store)
    monkeypatch.setattr(jobs.embeddings, "embed_texts", fake_embed)

    jobs.process_upload("alice.txt", b"Alice's content here.", "alice")
    jobs.process_upload("bob.txt", b"Bob's content here.", "bob")

    assert [d["filename"] for d in store.list_documents("alice")] == ["alice.txt"]
    assert [d["filename"] for d in store.list_documents("bob")] == ["bob.txt"]


def test_process_upload_raises_on_no_extractable_text(monkeypatch):
    store = VectorStore(DIM)
    jobs.configure(store)
    monkeypatch.setattr(jobs.embeddings, "embed_texts", fake_embed)

    with pytest.raises(ValueError):
        jobs.process_upload("empty.txt", b"   \n\n   ", "alice")

    assert store.list_documents("alice") == []
