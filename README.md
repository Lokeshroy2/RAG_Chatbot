# RAG Chatbot

**A fully local, multi-document RAG (Retrieval-Augmented Generation) chatbot.**
Upload PDF, TXT, or Markdown files and chat with them using a local LLM — no API keys, no cloud services, no cost.

[![CI](https://github.com/Lokeshroy2/RAG_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Lokeshroy2/RAG_Chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Mistral%207B-black)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Offline](https://img.shields.io/badge/100%25-local%20%26%20private-2ea44f)

---

## Overview

Instead of asking an LLM to answer from memory, this application first **retrieves** the most relevant passages from your documents, then passes them to the model as grounded context:

```
Question ──► Condense (follow-ups) ──► Embed ──► FAISS Search ──► Top-K Chunks ──► Ollama (Mistral) ──► Streamed, Cited Answer
```

Everything runs on your machine: embeddings via Sentence Transformers, vector search via FAISS, and generation via Ollama. The frontend uses system fonts and makes no external requests — truly offline.

## Features

- **Multi-document knowledge base** — upload many PDF / TXT / MD files; batch upload and drag & drop supported
- **Streaming answers** — tokens appear as the model generates them (Server-Sent Events)
- **Conversational retrieval** — follow-up questions ("*and when does it close?*") are automatically rewritten into standalone queries before retrieval, so context carries across turns
- **Persistent index** — documents survive server restarts (saved to `data/`, delete it or set `RAG_PERSIST=0` to opt out)
- **Per-document control** — include/exclude individual documents from retrieval, or delete them, without resetting the index
- **FAISS vector search** — fast cosine-similarity retrieval with automatic NumPy fallback
- **Source attribution** — every answer lists the file, text, and relevance score of each retrieved chunk, with inline citations (`[1]`, `[2]`)
- **Quality retrieval** — sentence-aware chunking with word-aligned overlap and a similarity threshold that keeps irrelevant context away from the LLM
- **Measured, not vibed** — a retrieval evaluation harness reports hit@k, keyword recall, and MRR against a golden dataset
- **Tested & linted** — unit test suite + ruff, enforced in CI on Python 3.10–3.12

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Lokeshroy2/RAG_Chatbot.git
cd RAG_Chatbot

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull the LLM and start Ollama
ollama pull mistral
ollama serve
```

### Run

```bash
python app.py
```

The app opens automatically at **http://127.0.0.1:8001**. Upload a document and start asking questions.

## Configuration

Everything is configurable via environment variables (defaults in [`rag_chatbot/config.py`](rag_chatbot/config.py)):

| Variable | Default | Description |
|---|---|---|
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `RAG_OLLAMA_MODEL` | `mistral:latest` | Any model available in your Ollama installation |
| `RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8001` | Server bind address |
| `RAG_CHUNK_SIZE` | `800` | Target characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Characters carried over between chunks |
| `RAG_TOP_K` | `5` | Maximum chunks sent to the LLM per question |
| `RAG_SCORE_THRESHOLD` | `0.25` | Minimum cosine similarity for a chunk to count as relevant |
| `RAG_MAX_FILE_MB` | `25` | Upload size limit per file |
| `RAG_CONDENSE_QUERIES` | `true` | Rewrite follow-up questions before retrieval (one extra LLM call) |
| `RAG_PERSIST` | `true` | Save/load the index across restarts |
| `RAG_DATA_DIR` | `./data` | Where the persisted index lives |
| `RAG_OPEN_BROWSER` | `true` | Auto-open the browser on startup |

Example: `RAG_OLLAMA_MODEL=llama3:8b RAG_PORT=9000 python app.py`

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status, version, model name, vector backend, index size |
| `POST` | `/upload` | Upload one or more files (multipart field `files`) |
| `GET` | `/documents` | List indexed documents with id, chunk count, enabled state |
| `PATCH` | `/documents/{id}` | Enable/disable a document in retrieval — body: `{"enabled": false}` |
| `DELETE` | `/documents/{id}` | Remove a single document and rebuild the index |
| `POST` | `/chat` | Ask a question — body: `{"question": "...", "history": [...]}` |
| `POST` | `/chat/stream` | Same as `/chat`, but streams the answer as Server-Sent Events |
| `DELETE` | `/reset` | Clear all documents |

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

1. **Extract** — text is pulled from each uploaded file (`pypdf` for PDFs)
2. **Chunk** — text is split into ~800-character, sentence-aware chunks with word-aligned 150-character overlap
3. **Embed** — chunks are encoded with `all-MiniLM-L6-v2` (normalized vectors)
4. **Index** — vectors are stored in a FAISS inner-product index and persisted to disk; each chunk keeps its source filename
5. **Condense** — follow-up questions are rewritten into standalone queries using the conversation history
6. **Retrieve** — the query is embedded and matched against all *enabled* documents; chunks below the similarity threshold are discarded
7. **Generate** — the top chunks, conversation history, and question are sent to Ollama, which streams an answer with inline source citations

## Development

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests cover chunking, the vector store (including persistence and the disabled-document paths), prompt construction, query condensation fallbacks, and configuration parsing. They run without torch/FAISS installed — heavy dependencies are loaded lazily, and the store falls back to NumPy search. CI runs the suite on Python 3.10, 3.11, and 3.12 plus `ruff check`.

### Evaluate retrieval quality

```bash
python -m eval.run_eval
```

Indexes the sample corpus in [`eval/corpus/`](eval/corpus/) and runs the golden questions in [`eval/golden.json`](eval/golden.json) through the exact chunk → embed → search pipeline the app uses. Reports **hit@k**, **keyword recall**, and **MRR**, and fails (non-zero exit) if quality drops below the configured floor — so retrieval regressions are caught before they ship. Extend `golden.json` with your own documents and questions to tune `RAG_CHUNK_SIZE` / `RAG_SCORE_THRESHOLD` against data instead of guessing.

### Lint

```bash
ruff check .
```

## Project Structure

```
RAG_Chatbot/
├── app.py                   # FastAPI routes, SSE streaming, startup wiring
├── rag_chatbot/
│   ├── config.py            # env-driven settings (RAG_*)
│   ├── chunking.py          # text extraction + sentence-aware chunking
│   ├── embeddings.py        # lazy-loaded sentence-transformers wrapper
│   ├── store.py             # thread-safe FAISS/NumPy vector store + persistence
│   └── llm.py               # Ollama client (blocking + streaming), prompts, query condensation
├── frontend/
│   └── index.html           # Single-file UI (chat, document manager, streaming) — zero external requests
├── tests/                   # pytest unit tests (no GPU/model needed)
├── eval/                    # retrieval quality harness + golden dataset
├── .github/workflows/ci.yml # lint + tests on 3.10–3.12
├── requirements.txt         # pinned runtime deps
└── requirements-dev.txt     # + pytest, ruff
```

## Limitations

- Answer quality depends on the local model (Mistral 7B by default) — small models occasionally miss citations
- Query condensation adds one extra LLM round-trip per follow-up question (disable with `RAG_CONDENSE_QUERIES=0`)
- No authentication — intended for local, single-user use

## Roadmap

- [x] Streaming responses
- [x] Persistent vector store (save/load index to disk)
- [x] Conversational retrieval (query condensation)
- [x] Retrieval evaluation harness
- [ ] Hybrid search (semantic + BM25 keyword)
- [ ] Cross-encoder reranking
- [ ] Docker deployment
- [ ] Chat history persistence

## Contributing

Issues and pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change. Please run `pytest` and `ruff check .` before submitting.

## License

[MIT](LICENSE)
