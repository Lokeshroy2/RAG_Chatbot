"""
RAG Chatbot - 100% Free, No API Key, Uses Ollama locally

Install:
    pip install fastapi uvicorn python-multipart pypdf sentence-transformers numpy

Run:
    python app.py

Open: http://127.0.0.1:8001
"""

import re, json, math, io, threading, time, webbrowser
import numpy as np
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import urllib.request, urllib.error

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_EMBED = True
except ImportError:
    HAS_EMBED = False
    EMBED_MODEL = None

# ── config ────────────────────────────────────────────────────────────────────
HOST         = "127.0.0.1"
PORT         = 8001
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest "
CHUNK_SIZE   = 500
CHUNK_OVERLAP= 100
TOP_K        = 5

# ── in-memory store ───────────────────────────────────────────────────────────
doc_store = {"chunks": [], "embeddings": None, "filenames": []}

# ── helpers ───────────────────────────────────────────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if not HAS_PDF:
            raise HTTPException(500, "pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="replace")

def chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(texts: List[str]) -> np.ndarray:
    if HAS_EMBED and EMBED_MODEL is not None:
        return EMBED_MODEL.encode(texts, show_progress_bar=False)
    vocab: dict = {}
    for t in texts:
        for w in t.lower().split():
            vocab.setdefault(w, len(vocab))
    vecs = []
    for t in texts:
        v = [0.0] * len(vocab)
        for w in t.lower().split():
            if w in vocab:
                v[vocab[w]] += 1.0
        norm = math.sqrt(sum(x*x for x in v)) or 1.0
        vecs.append([x/norm for x in v])
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n else 0.0

def retrieve(query: str) -> List[str]:
    chunks, embeddings = doc_store["chunks"], doc_store["embeddings"]
    if not chunks or embeddings is None:
        return []
    q_vec = embed_texts([query])[0]
    scores = [cosine_sim(q_vec, embeddings[i]) for i in range(len(chunks))]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [chunks[i] for i in top_idx]

def call_ollama(prompt: str) -> str:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        raise HTTPException(503, f"Ollama not reachable. Is it running? Error: {e}")

# ── FastAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="RAG Chatbot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── serve frontend HTML from file ─────────────────────────────────────────────
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
def root():
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h2>frontend/index.html not found</h2>", status_code=404)
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "has_embed": HAS_EMBED,
        "has_doc": bool(doc_store["chunks"])
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"[UPLOAD] Received file: {file.filename}, content_type: {file.content_type}")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, f"Unsupported type '{ext}'. Use PDF, TXT, or MD.")
    raw = await file.read()
    print(f"[UPLOAD] File size: {len(raw)} bytes")
    text = extract_text(raw, file.filename)
    print(f"[UPLOAD] Extracted {len(text)} characters")
    if not text.strip():
        raise HTTPException(400, "Could not extract text from file.")
    chunks = chunk_text(text)
    print(f"[UPLOAD] Created {len(chunks)} chunks")
    new_embeddings = embed_texts(chunks)
    existing = doc_store["embeddings"]
    all_embeddings = np.vstack([existing, new_embeddings]) if existing is not None else new_embeddings
    all_chunks = doc_store["chunks"] + chunks
    doc_store.update(chunks=all_chunks, embeddings=all_embeddings)
    doc_store["filenames"].append(file.filename)
    return {
        "filenames": doc_store["filenames"],
        "new_file": file.filename,
        "chars": len(text),
        "chunks": len(all_chunks),
        "embed_model": "all-MiniLM-L6-v2" if HAS_EMBED else "tf-idf-fallback"
    }

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []

@app.post("/chat")
def chat(req: ChatRequest):
    if not doc_store["chunks"]:
        raise HTTPException(400, "No documents loaded. Upload at least one file first.")
    context_chunks = retrieve(req.question)
    context = "\n\n---\n\n".join(context_chunks)
    history_str = ""
    for msg in (req.history or [])[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    prompt = (
        "You are a helpful assistant. Answer ONLY from the document context provided.\n"
        "If the answer is not in the document, say: "
        "'I don't have enough information in the document to answer that.'\n\n"
        f"=== DOCUMENT CONTEXT ===\n{context}\n\n"
        f"=== CONVERSATION HISTORY ===\n{history_str}\n"
        f"=== QUESTION ===\nUser: {req.question}\nAssistant:"
    )
    answer = call_ollama(prompt)
    return {"answer": answer, "sources": context_chunks}

@app.delete("/reset")
def reset():
    doc_store.update(chunks=[], embeddings=None, filenames=[])
    return {"status": "cleared"}

# ── run ───────────────────────────────────────────────────────────────────────
def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print(f"  RAG Chatbot is running!")
    print(f"  Open this in your browser:")
    print(f"  --> http://{HOST}:{PORT}")
    print("="*50 + "\n")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT)