# RAG Chatbot

**A multi-tenant RAG (Retrieval-Augmented Generation) chatbot.**
Sign up, upload PDF/TXT/Markdown files, and chat with *your own* documents — every answer is grounded in retrieved passages, cited inline, and streamed token-by-token.

[![CI](https://github.com/Lokeshroy2/RAG_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Lokeshroy2/RAG_Chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-accounts-336791?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-job%20queue-DC382D?logo=redis&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Instead of asking an LLM to answer from memory, this application first **retrieves** the most relevant passages from a user's own documents, then passes them to the model as grounded context:

```
Question ──► Condense (follow-ups) ──► Embed ──► FAISS Search (owner-scoped) ──► Top-K Chunks
          ──► Groq (primary) / Gemini (fallback on 429) ──► Streamed, Cited Answer
```

Every document is tagged with an owner at upload time, and every retrieval, list, enable/disable, and delete operation is scoped to the authenticated user — one tenant can never see or confirm the existence of another's documents (a mismatched owner gets a plain 404, not a 403, so existence itself isn't leaked).

## Features

- **Multi-tenant from the ground up** — JWT auth (bcrypt-hashed passwords, Postgres-backed user store), every document and chat owner-scoped
- **Background job queue for uploads** — `/upload` enqueues to Redis/RQ and returns immediately; poll `/jobs/{id}` for status, so a large batch upload never blocks the request that triggered it
- **Hosted LLM inference with automatic failover** — Groq primary, Gemini fallback on a 429 rate-limit, no self-hosted model to run
- **Conversational retrieval** — follow-up questions ("*and when does it close?*") are automatically rewritten into standalone queries before retrieval
- **Streaming answers** — tokens appear as the model generates them (Server-Sent Events)
- **Source attribution** — every answer lists the file, text, and relevance score of each retrieved chunk, with inline citations (`[1]`, `[2]`)
- **Quality retrieval** — sentence-aware chunking with word-aligned overlap and a similarity threshold that keeps irrelevant context away from the LLM
- **Measured, not vibed** — two independent eval harnesses (see [Evaluation](#evaluation) below): retrieval quality (hit@k, keyword recall, MRR) and answer quality (RAGAS faithfulness, answer relevancy)
- **Dockerized** — multi-stage build (CPU-only PyTorch, no ~2GB of unused CUDA libraries shipped)
- **Tested & linted** — 57 pytest tests + ruff, enforced in CI on Python 3.10–3.12

## Quick Start

### Prerequisites

- Python 3.10+
- A Postgres database (e.g. a free [Neon](https://neon.tech) instance)
- A Redis instance (e.g. free tier on [Upstash](https://upstash.com))
- Free API keys: [Groq](https://console.groq.com) (primary) and [Gemini](https://aistudio.google.com) (fallback)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Lokeshroy2/RAG_Chatbot.git
cd RAG_Chatbot

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# fill in RAG_DATABASE_URL, RAG_GROQ_API_KEY, RAG_GEMINI_API_KEY (see .env.example)
```

### Run

```bash
python app.py
```

Or with Docker (brings up the app only — point `.env` at your own hosted Postgres/Redis):

```bash
docker compose up
```

The app opens automatically at **http://127.0.0.1:8001**. Sign up, upload a document, and start asking questions.

## Configuration

Everything is configurable via environment variables (defaults in [`rag_chatbot/config.py`](rag_chatbot/config.py)):

| Variable | Default | Description |
|---|---|---|
| `RAG_DATABASE_URL` | *(required)* | Postgres connection string (Neon pooled URL recommended) |
| `RAG_REDIS_URL` | *(required)* | Redis connection string for the upload job queue |
| `RAG_JWT_SECRET` | *(ephemeral if unset)* | JWT signing secret — set a real value before deploying, or every restart invalidates all sessions |
| `RAG_JWT_EXPIRE_MINUTES` | `1440` | Access token lifetime (24h, no refresh token yet) |
| `RAG_GROQ_API_KEY` / `RAG_GROQ_MODEL` | — / `llama-3.3-70b-versatile` | Primary hosted LLM |
| `RAG_GEMINI_API_KEY` / `RAG_GEMINI_MODEL` | — / `gemini-2.0-flash` | Fallback hosted LLM, used on a Groq 429 |
| `RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8001` | Server bind address (`PORT` also respected, for platforms that inject it) |
| `RAG_CHUNK_SIZE` | `800` | Target characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Characters carried over between chunks |
| `RAG_TOP_K` | `5` | Maximum chunks sent to the LLM per question |
| `RAG_SCORE_THRESHOLD` | `0.25` | Minimum cosine similarity for a chunk to count as relevant |
| `RAG_MAX_FILE_MB` | `25` | Upload size limit per file |
| `RAG_CONDENSE_QUERIES` | `true` | Rewrite follow-up questions before retrieval (one extra LLM call) |
| `RAG_PERSIST` | `true` | Save/load the index across restarts |
| `RAG_DATA_DIR` | `./data` | Where the persisted index lives |
| `RAG_OPEN_BROWSER` | `true` | Auto-open the browser on startup |

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/signup` | — | Create an account, returns a JWT |
| `POST` | `/auth/login` | — | Log in, returns a JWT |
| `GET` | `/health` | — | Server status, version, model name, vector backend, index size |
| `POST` | `/upload` | ✓ | Enqueue one or more files for background processing (multipart field `files`) |
| `GET` | `/jobs/{job_id}` | ✓ | Poll an upload job's status (`queued` / `finished` / `failed`) |
| `GET` | `/documents` | ✓ | List this user's indexed documents with id, chunk count, enabled state |
| `PATCH` | `/documents/{id}` | ✓ | Enable/disable a document in retrieval — body: `{"enabled": false}` |
| `DELETE` | `/documents/{id}` | ✓ | Remove a single document and rebuild the index |
| `POST` | `/chat` | ✓ | Ask a question — body: `{"question": "...", "history": [...]}` |
| `POST` | `/chat/stream` | ✓ | Same as `/chat`, but streams the answer as Server-Sent Events |
| `DELETE` | `/reset` | ✓ | Clear all of this user's documents |

Routes marked ✓ require `Authorization: Bearer <token>` from `/auth/login` or `/auth/signup`.

<details>
<summary><strong>Example: <code>POST /chat</code></strong></summary>

Request:

```json
{
  "question": "and when does it close?",
  "history": [
    {"role": "user", "content": "What time does the cafeteria open?"},
    {"role": "assistant", "content": "The cafeteria opens at 8 AM [1]."}
  ]
}
```

Response:

```json
{
  "answer": "The cafeteria closes at 7:30 PM on weekdays [1].",
  "retrieval_query": "When does the cafeteria close?",
  "sources": [
    {"filename": "handbook.pdf", "text": "…", "score": 0.83}
  ]
}
```

</details>

<details>
<summary><strong>Example: <code>POST /chat/stream</code> (SSE)</strong></summary>

```
data: {"type": "sources", "sources": [...], "retrieval_query": "..."}

data: {"type": "delta", "text": " The"}

data: {"type": "delta", "text": " cafeteria"}

data: {"type": "done"}
```

</details>

## How It Works

1. **Upload** — `/upload` validates size/type per file, then enqueues extraction to a Redis-backed job queue and returns immediately (poll `/jobs/{id}`)
2. **Extract** — text is pulled from each file (`pypdf` for PDFs)
3. **Chunk** — text is split into ~800-character, sentence-aware chunks with word-aligned 150-character overlap
4. **Embed** — chunks are encoded with `all-MiniLM-L6-v2` (normalized vectors)
5. **Index** — vectors are stored in a FAISS inner-product index, tagged with the uploading user's id, and persisted to disk
6. **Condense** — follow-up questions are rewritten into standalone queries using the conversation history
7. **Retrieve** — the query is embedded and matched only against *this user's* enabled documents; chunks below the similarity threshold are discarded
8. **Generate** — the top chunks, conversation history, and question are sent to Groq (falling back to Gemini on a rate limit), which streams an answer with inline source citations

## Evaluation

Two independent harnesses, because retrieval quality and answer quality can fail independently (great retrieval + a model that still hallucinates is still a broken answer):

### Retrieval quality (no LLM, free, fast)

```bash
python -m eval.run_eval
```

Indexes the sample corpus in [`eval/corpus/`](eval/corpus/) and runs the golden questions in [`eval/golden.json`](eval/golden.json) through the exact chunk → embed → search pipeline the app uses. Reports **hit@k**, **keyword recall**, and **MRR**, and fails (non-zero exit) if quality drops below the configured floor.

Current result: **hit@5 100%, keyword recall 100%, MRR 1.000** (13/13 golden questions).

### Answer quality (makes real LLM calls, needs `RAG_GROQ_API_KEY`)

```bash
pip install -r requirements-dev.txt   # adds ragas, openai, truststore
python -m eval.run_ragas_eval
```

Runs the *actual* retrieve → prompt → generate pipeline the app uses for each golden question, then scores the real generated answers with [RAGAS](https://github.com/explodinggardens/ragas):

- **faithfulness** — is every claim in the answer supported by the retrieved context (catches hallucination)
- **answer relevancy** — does the answer actually address the question asked

Current result (13/13 questions): **mean faithfulness 1.000, mean answer relevancy 0.909**.

## Development

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests cover auth (signup/login/token verification), chunking, the vector store (owner-scoped isolation, persistence, disabled-document paths), the upload job queue, prompt construction, query condensation fallbacks, and configuration parsing — 57 tests total. They run without torch/FAISS installed — heavy dependencies are loaded lazily, and the store falls back to NumPy search. CI runs the suite on Python 3.10, 3.11, and 3.12 plus `ruff check`, against a SQLite stand-in for Postgres.

### Lint

```bash
ruff check .
```

## Project Structure

```
RAG_Chatbot/
├── app.py                   # FastAPI routes, auth, SSE streaming, startup wiring
├── rag_chatbot/
│   ├── auth.py               # password hashing, JWT issuance/verification, Postgres-backed user store
│   ├── config.py             # env-driven settings (RAG_*)
│   ├── chunking.py           # text extraction + sentence-aware chunking
│   ├── db.py                 # SQLAlchemy models + engine helpers (Postgres / SQLite in CI)
│   ├── embeddings.py         # lazy-loaded sentence-transformers wrapper
│   ├── jobs.py                # Redis/RQ background queue for document ingestion
│   ├── llm.py                 # Groq/Gemini clients + fallback, prompts, query condensation
│   └── store.py               # thread-safe, owner-scoped FAISS/NumPy vector store + persistence
├── frontend/
│   └── index.html            # Single-file UI (auth, chat, document manager, streaming)
├── tests/                    # pytest unit tests (57 tests, no GPU/model needed)
├── eval/                     # retrieval-quality harness + RAGAS answer-quality harness + golden dataset
├── Dockerfile                 # multi-stage build (CPU-only PyTorch)
├── docker-compose.yml
├── .github/workflows/ci.yml  # lint + tests on 3.10–3.12
├── requirements.txt          # pinned runtime deps
└── requirements-dev.txt      # + pytest, ruff, ragas, openai, truststore (eval-only)
```

## Limitations

- Answer quality depends on the configured model — smaller models occasionally miss citations
- Query condensation adds one extra LLM round-trip per follow-up question (disable with `RAG_CONDENSE_QUERIES=0`)
- No PII redaction — if a document contains personal data (phone numbers, etc.), the chatbot will recite it verbatim on request. Isolation between tenants is enforced (one user can never retrieve another's documents), but within a user's own documents there's no redaction layer
- JWT access tokens can't be revoked before they expire; there's no refresh-token flow yet
- The in-process FAISS index is per-instance — running two app instances would need a shared vector store, not just a shared database

## Roadmap

- [x] Streaming responses
- [x] Persistent vector store (save/load index to disk)
- [x] Conversational retrieval (query condensation)
- [x] Retrieval evaluation harness (hit@k, keyword recall, MRR)
- [x] Multi-tenant auth + owner-scoped document isolation
- [x] Background job queue for uploads
- [x] Hosted LLM inference with automatic fallback (Groq → Gemini)
- [x] Docker deployment
- [x] Answer-quality evaluation harness (RAGAS: faithfulness, answer relevancy)
- [ ] Hybrid search (semantic + BM25 keyword)
- [ ] Cross-encoder reranking
- [ ] Refresh tokens / token revocation
- [ ] Chat history persistence

## Contributing

Issues and pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change. Please run `pytest` and `ruff check .` before submitting.

## License

[MIT](LICENSE)
