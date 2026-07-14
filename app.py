"""
Multi-Document RAG Chatbot - 100% Free, No API Key, Uses Ollama locally

Install:
    pip install -r requirements.txt

Run:
    python app.py

Open: http://127.0.0.1:8001
"""
import json
import logging
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_chatbot import __version__, embeddings
from rag_chatbot.chunking import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileError,
    chunk_text,
    extract_text,
)
from rag_chatbot.config import settings
from rag_chatbot.llm import OllamaClient, OllamaError, build_prompt, condense_question
from rag_chatbot.store import HAS_FAISS, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

ollama = OllamaClient(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
store: VectorStore | None = None


def get_store() -> VectorStore:
    if store is None:
        raise HTTPException(503, "Server is still starting up. Try again in a moment.")
    return store


def check_ollama():
    """Warn early if Ollama is down or the configured model is missing."""
    try:
        models = ollama.list_models()
    except OllamaError as e:
        log.warning("%s — chat will fail until Ollama is running (ollama serve).", e.message)
        return
    base = settings.ollama_model.split(":")[0]
    if not any(m == settings.ollama_model or m.startswith(f"{base}:") for m in models):
        log.warning(
            "Model '%s' not found in Ollama (available: %s). Run: ollama pull %s",
            settings.ollama_model, ", ".join(models) or "none", base,
        )
    else:
        log.info("Ollama is up, model '%s' is available.", settings.ollama_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    dim = embeddings.embedding_dim()  # loads the model once, up front
    store = VectorStore(dim, persist_dir=settings.data_dir if settings.persist else None)
    check_ollama()
    yield


app = FastAPI(title="Multi-Document RAG Chatbot", version=__version__, lifespan=lifespan)


# ── models ────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = []


class DocumentUpdate(BaseModel):
    enabled: bool


# ── pages & status ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h2>frontend/index.html not found</h2>", status_code=404)
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    docs = get_store().list_documents()
    return {
        "status": "ok",
        "version": __version__,
        "model": settings.ollama_model,
        "vector_backend": "faiss" if HAS_FAISS else "numpy",
        "documents": len(docs),
        "chunks": sum(d["chunks"] for d in docs),
    }


# ── document management ───────────────────────────────────────────────────────
@app.get("/documents")
def list_documents():
    docs = get_store().list_documents()
    return {"documents": docs, "total_chunks": sum(d["chunks"] for d in docs)}


@app.patch("/documents/{doc_id}")
def update_document(doc_id: str, update: DocumentUpdate):
    try:
        get_store().set_enabled(doc_id, update.enabled)
    except KeyError:
        raise HTTPException(404, f"Unknown document id '{doc_id}'") from None
    return {"status": "ok", "id": doc_id, "enabled": update.enabled}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        get_store().remove_document(doc_id)
    except KeyError:
        raise HTTPException(404, f"Unknown document id '{doc_id}'") from None
    return {"status": "deleted", "id": doc_id}


@app.post("/upload")
def upload_files(files: list[UploadFile] = File(...)):
    added, skipped = [], []
    for file in files:
        filename = file.filename or "untitled.txt"
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append({"filename": filename, "reason": f"Unsupported type '{ext}'"})
            continue
        raw = file.file.read()
        if len(raw) > settings.max_file_mb * 1024 * 1024:
            skipped.append(
                {"filename": filename, "reason": f"Larger than {settings.max_file_mb} MB"}
            )
            continue
        try:
            text = extract_text(raw, filename)
        except UnsupportedFileError as e:
            raise HTTPException(500, str(e)) from e
        except Exception as e:
            skipped.append({"filename": filename, "reason": f"Could not parse file: {e}"})
            continue
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            skipped.append({"filename": filename, "reason": "No extractable text"})
            continue
        vecs = embeddings.embed_texts(chunks)
        doc_id = get_store().add_document(filename, chunks, vecs)
        log.info(
            "Indexed %s: %d chars -> %d chunks (id=%s)", filename, len(text), len(chunks), doc_id
        )
        added.append(
            {"id": doc_id, "filename": filename, "chunks": len(chunks), "chars": len(text)}
        )

    if not added and skipped:
        raise HTTPException(400, "; ".join(f"{s['filename']}: {s['reason']}" for s in skipped))

    docs = get_store().list_documents()
    return {
        "added": added,
        "skipped": skipped,
        "total_docs": len(docs),
        "total_chunks": sum(d["chunks"] for d in docs),
        "embed_model": settings.embed_model_name,
    }


@app.delete("/reset")
def reset():
    get_store().clear()
    return {"status": "cleared"}


# ── chat ──────────────────────────────────────────────────────────────────────
NOT_FOUND_ANSWER = "I couldn't find anything relevant in the documents."


def _ensure_ready():
    docs = get_store().list_documents()
    if not docs:
        raise HTTPException(400, "No documents loaded. Upload at least one file first.")
    if not any(d["enabled"] for d in docs):
        raise HTTPException(400, "All documents are disabled. Enable at least one.")


def _retrieve(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Condense the question (if it's a follow-up), embed it, and search."""
    retrieval_query = question
    if settings.condense_queries and history:
        retrieval_query = condense_question(ollama, question, history)
        if retrieval_query != question:
            log.info("Condensed query: %r -> %r", question, retrieval_query)
    q_vec = embeddings.embed_texts([retrieval_query])[0]
    sources = get_store().search(q_vec, settings.top_k, settings.score_threshold)
    return retrieval_query, sources


@app.post("/chat")
def chat(req: ChatRequest):
    _ensure_ready()
    history = [m.model_dump() for m in req.history]
    retrieval_query, sources = _retrieve(req.question, history)
    if not sources:
        return {"answer": NOT_FOUND_ANSWER, "sources": [], "retrieval_query": retrieval_query}
    prompt = build_prompt(
        req.question, sources, history, settings.history_window, settings.history_char_budget
    )
    try:
        answer = ollama.generate(prompt)
    except OllamaError as e:
        raise HTTPException(e.status, e.message) from e
    return {"answer": answer, "sources": sources, "retrieval_query": retrieval_query}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Same as /chat but streams the answer token-by-token as Server-Sent Events."""
    _ensure_ready()
    history = [m.model_dump() for m in req.history]

    def generate():
        try:
            retrieval_query, sources = _retrieve(req.question, history)
            yield _sse(
                {"type": "sources", "sources": sources, "retrieval_query": retrieval_query}
            )
            if not sources:
                yield _sse({"type": "delta", "text": NOT_FOUND_ANSWER})
                yield _sse({"type": "done"})
                return
            prompt = build_prompt(
                req.question, sources, history,
                settings.history_window, settings.history_char_budget,
            )
            for fragment in ollama.generate_stream(prompt):
                yield _sse({"type": "delta", "text": fragment})
            yield _sse({"type": "done"})
        except OllamaError as e:
            yield _sse({"type": "error", "message": e.message})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── run ───────────────────────────────────────────────────────────────────────
def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{settings.host}:{settings.port}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Multi-Document RAG Chatbot is running!")
    print("  Open this in your browser:")
    print(f"  --> http://{settings.host}:{settings.port}")
    print("=" * 50 + "\n")
    if settings.open_browser:
        threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=settings.host, port=settings.port)
