"""
Multi-Document RAG Chatbot - Groq/Gemini hosted inference, Postgres accounts

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
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from rag_chatbot import __version__, embeddings, jobs
from rag_chatbot.auth import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserStore,
    create_access_token,
    get_current_user,
)
from rag_chatbot.chunking import SUPPORTED_EXTENSIONS
from rag_chatbot.config import settings
from rag_chatbot.db import normalize_url
from rag_chatbot.llm import (
    FallbackLLMClient,
    GeminiClient,
    GroqClient,
    LLMError,
    build_prompt,
    condense_question,
)
from rag_chatbot.store import HAS_FAISS, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

def _build_llm_client():
    groq = GroqClient(settings.groq_api_key, settings.groq_model, settings.llm_timeout)
    gemini = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout)
    primary, fallback = (groq, gemini) if settings.llm_provider == "groq" else (gemini, groq)
    return FallbackLLMClient(primary, fallback)


llm = _build_llm_client()
store: VectorStore | None = None
users: UserStore | None = None


def get_store() -> VectorStore:
    if store is None:
        raise HTTPException(503, "Server is still starting up. Try again in a moment.")
    return store


def get_users() -> UserStore:
    if users is None:
        raise HTTPException(503, "Server is still starting up. Try again in a moment.")
    return users


def check_llm_config():
    """Warn early if an API key is missing — chat will 502/503 until it's set."""
    missing = [
        name
        for name, key in (
            ("RAG_GROQ_API_KEY", settings.groq_api_key),
            ("RAG_GEMINI_API_KEY", settings.gemini_api_key),
        )
        if not key
    ]
    if missing:
        log.warning(
            "%s not set — chat will fail until it's configured. Primary provider is '%s'.",
            " and ".join(missing), settings.llm_provider,
        )
    else:
        log.info("LLM provider '%s' configured, with fallback available.", settings.llm_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, users
    if not settings.database_url:
        raise RuntimeError(
            "RAG_DATABASE_URL is not set. User accounts are stored in Postgres as of "
            "Phase 3 — set it (e.g. in a .env file) before running."
        )
    if not settings.redis_url:
        raise RuntimeError(
            "RAG_REDIS_URL is not set. Document uploads are processed via a background "
            "job queue as of Phase 6 — set it (e.g. in a .env file) before running."
        )
    dim = embeddings.embedding_dim()  # loads the model once, up front
    store = VectorStore(dim, persist_dir=settings.data_dir if settings.persist else None)
    engine = create_engine(normalize_url(settings.database_url), pool_pre_ping=True)
    users = UserStore(engine)
    jobs.configure(store)
    jobs.start_worker()
    check_llm_config()
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
    """Unauthenticated on purpose — uptime monitors need to hit this without a token.
    Reports counts across all owners, not just one user's documents."""
    docs = get_store().list_documents(None)
    model = settings.groq_model if settings.llm_provider == "groq" else settings.gemini_model
    return {
        "status": "ok",
        "version": __version__,
        "model": f"{settings.llm_provider}:{model}",
        "vector_backend": "faiss" if HAS_FAISS else "numpy",
        "documents": len(docs),
        "chunks": sum(d["chunks"] for d in docs),
    }


# ── auth ─────────────────────────────────────────────────────────────────────
@app.post("/auth/signup", response_model=TokenResponse)
def signup(req: SignupRequest):
    try:
        user_id = get_users().create(req.username, req.password)
    except ValueError:
        raise HTTPException(409, "Username already taken.") from None
    return TokenResponse(access_token=create_access_token(user_id))


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user_id = get_users().verify(req.username, req.password)
    if user_id is None:
        raise HTTPException(401, "Invalid username or password.")
    return TokenResponse(access_token=create_access_token(user_id))


# ── document management ───────────────────────────────────────────────────────
@app.get("/documents")
def list_documents(current_user: str = Depends(get_current_user)):
    docs = get_store().list_documents(current_user)
    return {"documents": docs, "total_chunks": sum(d["chunks"] for d in docs)}


@app.patch("/documents/{doc_id}")
def update_document(
    doc_id: str, update: DocumentUpdate, current_user: str = Depends(get_current_user)
):
    try:
        get_store().set_enabled(doc_id, update.enabled, current_user)
    except KeyError:
        raise HTTPException(404, f"Unknown document id '{doc_id}'") from None
    return {"status": "ok", "id": doc_id, "enabled": update.enabled}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, current_user: str = Depends(get_current_user)):
    try:
        get_store().remove_document(doc_id, current_user)
    except KeyError:
        raise HTTPException(404, f"Unknown document id '{doc_id}'") from None
    return {"status": "deleted", "id": doc_id}


@app.post("/upload")
def upload_files(
    files: list[UploadFile] = File(...), current_user: str = Depends(get_current_user)
):
    """Validates each file fast, then hands the slow work (extract/chunk/embed)
    off to a background job — see rag_chatbot/jobs.py. Poll /jobs/{id} for status."""
    queued, skipped = [], []
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
        job = jobs.queue.enqueue(
            jobs.process_upload, filename, raw, current_user, meta={"owner_id": current_user}
        )
        queued.append({"job_id": job.id, "filename": filename})

    if not queued and skipped:
        raise HTTPException(400, "; ".join(f"{s['filename']}: {s['reason']}" for s in skipped))

    return {"queued": queued, "skipped": skipped}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, current_user: str = Depends(get_current_user)):
    job = jobs.get_job(job_id)
    if job is None or job.meta.get("owner_id") != current_user:
        raise HTTPException(404, f"Unknown job id '{job_id}'")
    payload = {"id": job.id, "status": job.get_status()}
    if payload["status"] == "finished":
        payload["result"] = job.result
    elif payload["status"] == "failed":
        payload["error"] = str(job.exc_info or "error").strip().splitlines()[-1][:300]
    return payload


@app.delete("/reset")
def reset(current_user: str = Depends(get_current_user)):
    get_store().clear(current_user)
    return {"status": "cleared"}


# ── chat ──────────────────────────────────────────────────────────────────────
NOT_FOUND_ANSWER = "I couldn't find anything relevant in the documents."


def _ensure_ready(current_user: str):
    docs = get_store().list_documents(current_user)
    if not docs:
        raise HTTPException(400, "No documents loaded. Upload at least one file first.")
    if not any(d["enabled"] for d in docs):
        raise HTTPException(400, "All documents are disabled. Enable at least one.")


def _retrieve(question: str, history: list[dict], current_user: str) -> tuple[str, list[dict]]:
    """Condense the question (if it's a follow-up), embed it, and search."""
    retrieval_query = question
    if settings.condense_queries and history:
        retrieval_query = condense_question(llm, question, history)
        if retrieval_query != question:
            log.info("Condensed query: %r -> %r", question, retrieval_query)
    q_vec = embeddings.embed_texts([retrieval_query])[0]
    sources = get_store().search(
        q_vec,
        owner_id=current_user,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
    )
    return retrieval_query, sources


@app.post("/chat")
def chat(req: ChatRequest, current_user: str = Depends(get_current_user)):
    _ensure_ready(current_user)
    history = [m.model_dump() for m in req.history]
    retrieval_query, sources = _retrieve(req.question, history, current_user)
    if not sources:
        return {"answer": NOT_FOUND_ANSWER, "sources": [], "retrieval_query": retrieval_query}
    prompt = build_prompt(
        req.question, sources, history, settings.history_window, settings.history_char_budget
    )
    try:
        answer = llm.generate(prompt)
    except LLMError as e:
        raise HTTPException(e.status, e.message) from e
    return {"answer": answer, "sources": sources, "retrieval_query": retrieval_query}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, current_user: str = Depends(get_current_user)):
    """Same as /chat but streams the answer token-by-token as Server-Sent Events."""
    _ensure_ready(current_user)
    history = [m.model_dump() for m in req.history]

    def generate():
        try:
            retrieval_query, sources = _retrieve(req.question, history, current_user)
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
            for fragment in llm.generate_stream(prompt):
                yield _sse({"type": "delta", "text": fragment})
            yield _sse({"type": "done"})
        except LLMError as e:
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
