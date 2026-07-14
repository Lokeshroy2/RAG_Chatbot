"""
Multi-Document RAG Chatbot - 100% Free, No API Key, Uses Ollama locally

Install:
    pip install -r requirements.txt

Run:
    python app.py
    

Open: http://127.0.0.1:8001
"""

import io
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import List, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ── required / optional deps ──────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers is required.\n"
        "Run: pip install sentence-transformers"
    )

try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# ── config ────────────────────────────────────────────────────────────────────
HOST            = "127.0.0.1"
PORT            = 8001
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "mistral:latest"
CHUNK_SIZE      = 800     # target characters per chunk
CHUNK_OVERLAP   = 150     # characters carried over between chunks
TOP_K           = 5       # max chunks sent to the LLM
SCORE_THRESHOLD = 0.25    # minimum cosine similarity to count as relevant
MAX_FILE_MB     = 25
HISTORY_WINDOW  = 6       # messages of history included in the prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

log.info("Loading embedding model (all-MiniLM-L6-v2)…")
try:
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    # no internet / SSL trouble -> use the locally cached copy if present
    log.warning("Could not reach HuggingFace, trying local cache…")
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
log.info("Embedding model ready (dim=%d, faiss=%s)", EMBED_DIM, HAS_FAISS)


# ── vector store ──────────────────────────────────────────────────────────────
class VectorStore:
    """Multi-document store: per-chunk metadata + FAISS index (numpy fallback)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.lock = threading.Lock()
        self.docs: dict = {}          # doc_id -> {"filename", "enabled", "chunks", "chars"}
        self.meta: List[dict] = []    # per chunk: {"doc_id", "filename", "text"}
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.index = faiss.IndexFlatIP(dim) if HAS_FAISS else None

    def _rebuild_index(self):
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            if len(self.embeddings):
                self.index.add(self.embeddings)

    def add_document(self, filename: str, chunks: List[str], vecs: np.ndarray) -> str:
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
            return doc_id

    def remove_document(self, doc_id: str):
        with self.lock:
            if doc_id not in self.docs:
                raise HTTPException(404, f"Unknown document id '{doc_id}'")
            keep = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
            self.meta = [self.meta[i] for i in keep]
            self.embeddings = self.embeddings[keep] if keep else np.zeros((0, self.dim), np.float32)
            del self.docs[doc_id]
            self._rebuild_index()

    def set_enabled(self, doc_id: str, enabled: bool):
        with self.lock:
            if doc_id not in self.docs:
                raise HTTPException(404, f"Unknown document id '{doc_id}'")
            self.docs[doc_id]["enabled"] = enabled

    def clear(self):
        with self.lock:
            self.docs.clear()
            self.meta.clear()
            self.embeddings = np.zeros((0, self.dim), np.float32)
            self._rebuild_index()

    def list_documents(self) -> List[dict]:
        with self.lock:
            return [{"id": k, **v} for k, v in self.docs.items()]

    def search(self, q_vec: np.ndarray, top_k: int = TOP_K) -> List[dict]:
        """Return the most relevant chunks from enabled documents, with scores."""
        with self.lock:
            n = len(self.meta)
            if n == 0 or not any(d["enabled"] for d in self.docs.values()):
                return []
            # fetch extra so disabled docs can be filtered out afterwards
            k_fetch = min(n, max(top_k * 4, 20))
            if HAS_FAISS and self.index.ntotal:
                scores, ids = self.index.search(q_vec[None, :].astype(np.float32), k_fetch)
                ranked = list(zip(ids[0].tolist(), scores[0].tolist()))
            else:
                scores = self.embeddings @ q_vec
                order = np.argsort(scores)[::-1][:k_fetch]
                ranked = [(int(i), float(scores[i])) for i in order]

            results = []
            for i, score in ranked:
                if i < 0:
                    continue
                m = self.meta[i]
                if not self.docs[m["doc_id"]]["enabled"]:
                    continue
                results.append({"filename": m["filename"], "text": m["text"], "score": round(float(score), 4)})

            relevant = [r for r in results if r["score"] >= SCORE_THRESHOLD][:top_k]
            # nothing clears the bar -> still give the LLM the best two leads
            return relevant or results[:2]


store = VectorStore(EMBED_DIM)


# ── document processing ───────────────────────────────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if not HAS_PDF:
            raise HTTPException(500, "pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str) -> List[str]:
    """Sentence-aware chunking: pack sentences up to CHUNK_SIZE with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + " " + sentence if CHUNK_OVERLAP else sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    # hard-split anything still oversized (e.g. text with no sentence breaks)
    out = []
    for chunk in chunks:
        while len(chunk) > CHUNK_SIZE * 1.5:
            out.append(chunk[:CHUNK_SIZE])
            chunk = chunk[CHUNK_SIZE - CHUNK_OVERLAP:]
        out.append(chunk)
    return [c.strip() for c in out if c.strip()]


def embed_texts(texts: List[str]) -> np.ndarray:
    # normalized embeddings -> inner product == cosine similarity
    vecs = EMBED_MODEL.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


# ── Ollama ────────────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise HTTPException(502, f"Ollama error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(503, f"Ollama not reachable. Is it running? Error: {e}")
    if "error" in data:
        raise HTTPException(502, f"Ollama error: {data['error']}")
    return data.get("response", "").strip()


def build_prompt(question: str, sources: List[dict], history: List["ChatMessage"]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (from {s['filename']})\n{s['text']}" for i, s in enumerate(sources)
    )
    history_str = "".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}\n"
        for m in history[-HISTORY_WINDOW:]
    )
    return (
        "You are a helpful assistant. Answer ONLY from the numbered document "
        "sources below. Cite sources inline like [1] or [2] where relevant.\n"
        "If the answer is not in the sources, say: "
        "'I don't have enough information in the documents to answer that.'\n\n"
        f"=== SOURCES ===\n{context}\n\n"
        f"=== CONVERSATION HISTORY ===\n{history_str}\n"
        f"=== QUESTION ===\nUser: {question}\nAssistant:"
    )


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Document RAG Chatbot")

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
def root():
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h2>frontend/index.html not found</h2>", status_code=404)
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    docs = store.list_documents()
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "vector_backend": "faiss" if HAS_FAISS else "numpy",
        "documents": len(docs),
        "chunks": sum(d["chunks"] for d in docs),
    }


@app.get("/documents")
def list_documents():
    docs = store.list_documents()
    return {"documents": docs, "total_chunks": sum(d["chunks"] for d in docs)}


class DocumentUpdate(BaseModel):
    enabled: bool


@app.patch("/documents/{doc_id}")
def update_document(doc_id: str, update: DocumentUpdate):
    store.set_enabled(doc_id, update.enabled)
    return {"status": "ok", "id": doc_id, "enabled": update.enabled}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    store.remove_document(doc_id)
    return {"status": "deleted", "id": doc_id}


@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...)):
    added, skipped = [], []
    for file in files:
        filename = file.filename or "untitled.txt"
        ext = Path(filename).suffix.lower()
        if ext not in {".pdf", ".txt", ".md"}:
            skipped.append({"filename": filename, "reason": f"Unsupported type '{ext}'"})
            continue
        raw = file.file.read()
        if len(raw) > MAX_FILE_MB * 1024 * 1024:
            skipped.append({"filename": filename, "reason": f"Larger than {MAX_FILE_MB} MB"})
            continue
        try:
            text = extract_text(raw, filename)
        except HTTPException:
            raise
        except Exception as e:
            skipped.append({"filename": filename, "reason": f"Could not parse file: {e}"})
            continue
        chunks = chunk_text(text)
        if not chunks:
            skipped.append({"filename": filename, "reason": "No extractable text"})
            continue
        vecs = embed_texts(chunks)
        doc_id = store.add_document(filename, chunks, vecs)
        log.info("Indexed %s: %d chars -> %d chunks (id=%s)", filename, len(text), len(chunks), doc_id)
        added.append({"id": doc_id, "filename": filename, "chunks": len(chunks), "chars": len(text)})

    if not added and skipped:
        raise HTTPException(400, "; ".join(f"{s['filename']}: {s['reason']}" for s in skipped))

    docs = store.list_documents()
    return {
        "added": added,
        "skipped": skipped,
        "total_docs": len(docs),
        "total_chunks": sum(d["chunks"] for d in docs),
        "embed_model": "all-MiniLM-L6-v2",
    }


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: List[ChatMessage] = []


@app.post("/chat")
def chat(req: ChatRequest):
    docs = store.list_documents()
    if not docs:
        raise HTTPException(400, "No documents loaded. Upload at least one file first.")
    if not any(d["enabled"] for d in docs):
        raise HTTPException(400, "All documents are disabled. Enable at least one.")
    q_vec = embed_texts([req.question])[0]
    sources = store.search(q_vec)
    if not sources:
        return {"answer": "I couldn't find anything relevant in the documents.", "sources": []}
    answer = call_ollama(build_prompt(req.question, sources, req.history))
    return {"answer": answer, "sources": sources}


@app.delete("/reset")
def reset():
    store.clear()
    return {"status": "cleared"}


# ── run ───────────────────────────────────────────────────────────────────────
def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Multi-Document RAG Chatbot is running!")
    print("  Open this in your browser:")
    print(f"  --> http://{HOST}:{PORT}")
    print("=" * 50 + "\n")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT)
