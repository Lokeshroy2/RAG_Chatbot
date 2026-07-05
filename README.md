# RAG Chatbot

**A fully local, multi-document RAG (Retrieval-Augmented Generation) chatbot.**
Upload PDF, TXT, or Markdown files and chat with them using a local LLM — no API keys, no cloud services, no cost.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Mistral%207B-black)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![Offline](https://img.shields.io/badge/100%25-local%20%26%20private-2ea44f)

---

## Overview

Instead of asking an LLM to answer from memory, this application first **retrieves** the most relevant passages from your documents, then passes them to the model as grounded context:

```
Question ──► Embed ──► FAISS Search ──► Top-K Chunks ──► Ollama (Mistral) ──► Cited Answer
```

Everything runs on your machine: embeddings via Sentence Transformers, vector search via FAISS, and generation via Ollama.

## Features

- **Multi-document knowledge base** — upload many PDF / TXT / MD files; batch upload and drag & drop supported
- **Per-document control** — include/exclude individual documents from retrieval, or delete them, without resetting the index
- **FAISS vector search** — fast cosine-similarity retrieval with automatic NumPy fallback
- **Source attribution** — every answer lists the file, text, and relevance score of each retrieved chunk, with inline citations (`[1]`, `[2]`)
- **Quality retrieval** — sentence-aware chunking with overlap and a similarity threshold that keeps irrelevant context away from the LLM
- **Conversation history** — follow-up questions work naturally
- **Private by design** — documents never leave your machine

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

All settings live at the top of [`app.py`](app.py):

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral:latest` | Any model available in your Ollama installation |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters carried over between chunks |
| `TOP_K` | `5` | Maximum chunks sent to the LLM per question |
| `SCORE_THRESHOLD` | `0.25` | Minimum cosine similarity for a chunk to count as relevant |
| `MAX_FILE_MB` | `25` | Upload size limit per file |

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status, model name, vector backend, index size |
| `POST` | `/upload` | Upload one or more files (multipart field `files`) |
| `GET` | `/documents` | List indexed documents with id, chunk count, enabled state |
| `PATCH` | `/documents/{id}` | Enable/disable a document in retrieval — body: `{"enabled": false}` |
| `DELETE` | `/documents/{id}` | Remove a single document and rebuild the index |
| `POST` | `/chat` | Ask a question — body: `{"question": "...", "history": [...]}` |
| `DELETE` | `/reset` | Clear all documents |

<details>
<summary><strong>Example: <code>POST /chat</code></strong></summary>

Request:

```json
{
  "question": "What time does the cafeteria open?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response:

```json
{
  "answer": "The cafeteria opens at 8 AM [1].",
  "sources": [
    {"filename": "handbook.pdf", "text": "…", "score": 0.83}
  ]
}
```

</details>

## How It Works

1. **Extract** — text is pulled from each uploaded file (`pypdf` for PDFs)
2. **Chunk** — text is split into ~800-character, sentence-aware chunks with 150-character overlap
3. **Embed** — chunks are encoded with `all-MiniLM-L6-v2` (normalized vectors)
4. **Index** — vectors are stored in a FAISS inner-product index; each chunk keeps its source filename
5. **Retrieve** — questions are embedded and matched against all *enabled* documents; chunks below the similarity threshold are discarded
6. **Generate** — the top chunks, conversation history, and question are sent to Ollama, which answers with inline source citations

## Tech Stack

| Technology | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | Backend API |
| [Ollama](https://ollama.com/) + Mistral 7B | Local LLM inference |
| [Sentence Transformers](https://www.sbert.net/) | Embedding generation (`all-MiniLM-L6-v2`) |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [pypdf](https://pypdf.readthedocs.io/) | PDF text extraction |
| Vanilla HTML / CSS / JS | Frontend (single file, no build step) |

## Project Structure

```
RAG_Chatbot/
├── app.py               # FastAPI backend: vector store, retrieval, Ollama client
├── frontend/
│   └── index.html       # Single-file UI (chat, document manager, upload)
├── requirements.txt
└── README.md
```

## Limitations

- The vector index is in-memory and resets when the server restarts
- Responses are not streamed (the full answer arrives at once)
- No authentication — intended for local, single-user use

## Roadmap

- [ ] Streaming responses
- [ ] Persistent vector store (save/load index to disk)
- [ ] Hybrid search (semantic + keyword)
- [ ] Docker deployment
- [ ] Chat history persistence

## Contributing

Issues and pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change.
