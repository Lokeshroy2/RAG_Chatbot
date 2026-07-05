# Local RAG Chatbot using Ollama + FastAPI

A fully local and 100% free Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, Ollama, and Sentence Transformers.

Upload multiple documents (PDF/TXT/Markdown) and ask questions across all of them using a local LLM — no paid API keys or cloud services required.


---

# What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

Instead of asking an LLM to answer from its own memory, RAG first retrieves relevant information from documents and then sends that context to the LLM to generate accurate answers.

## Normal Chatbot

```text
Question → LLM → Answer
```

Problem:
- Hallucinations
- Limited knowledge
- Cannot access your custom files

---

## RAG Chatbot

```text
Question → Retrieve Relevant Context → LLM → Contextual Answer
```

Benefits:
- More accurate answers
- Uses your own documents
- Reduces hallucination
- Better contextual understanding

---

# Why This Project?

Most AI chatbots depend on:
- Paid APIs
- Internet connection
- Cloud infrastructure
- Expensive vector databases

This project provides a completely local and beginner-friendly alternative.

## Benefits

- 100% free
- No OpenAI API key required
- Fully offline
- Privacy friendly
- Simple architecture
- Easy to understand RAG pipeline
- Beginner-friendly implementation
- Fast local inference using Ollama

---

# Features

- Upload multiple PDF, TXT, and Markdown files simultaneously (single batch request)
- Drag & drop support for multiple files at once
- FAISS vector index for fast semantic search (numpy fallback if FAISS not installed)
- Per-document management: enable/disable individual documents in retrieval, delete individually
- Source attribution — every answer shows which file each chunk came from, with relevance scores
- Inline citations ([1], [2]) in LLM answers
- Sentence-aware chunking with overlap
- Similarity threshold filtering (irrelevant chunks are not sent to the LLM)
- Local LLM inference using Ollama
- Conversation history support
- FastAPI backend, simple frontend UI
- Fully offline setup, no external APIs

---

# Architecture Overview

```text
                ┌──────────────────┐
                │  Upload File(s)  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Extract Text    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   Chunk Text     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Generate Embeds  │  ← Only new chunks embedded per upload
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Stack + Store   │  ← Appended to existing embeddings
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Vector Retrieval │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Ollama + Mistral │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Final Response  │
                └──────────────────┘
```

---

# How Retrieval Works

## Step 1 — Upload Documents

The user uploads one or more PDF, TXT, or Markdown files. Multiple files can be selected at once or dragged and dropped together.

---

## Step 2 — Text Extraction

The backend extracts raw text from each uploaded file.

---

## Step 3 — Chunking

Large text is split into smaller overlapping chunks to improve retrieval accuracy.

```text
Chunk 1
Chunk 2
Chunk 3
```

---

## Step 4 — Embedding Generation

Each new chunk is converted into vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

New embeddings are stacked onto existing ones using `np.vstack` — previously uploaded documents are not re-embedded.

---

## Step 5 — Semantic Search

When the user asks a question:

1. The question is converted into embeddings
2. Cosine similarity is calculated against all stored chunk embeddings
3. The most relevant chunks are retrieved across all documents

---

## Step 6 — LLM Response

Relevant chunks are sent to Ollama + Mistral to generate the final answer.

---

# Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core programming language |
| FastAPI | Backend API framework |
| Ollama | Local LLM inference |
| Mistral | Language model |
| Sentence Transformers | Embedding generation |
| FAISS | Vector similarity search |
| NumPy | Vector math (and FAISS fallback) |
| PyPDF | PDF text extraction |
| HTML/CSS/JS | Frontend UI |

---

# Project Structure

```text
local-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
│
├── frontend/
│   └── index.html
│
├── screenshots/
│   ├── home.png
│   ├── upload.png
│   ├── chat.png
│   └── demo.gif
│
└── sample_docs/
    └── sample.pdf
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/local-rag-chatbot.git
cd local-rag-chatbot
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Install Ollama

Download Ollama: https://ollama.com

---

## 4. Pull Mistral Model

```bash
ollama pull mistral
```

---

## 5. Start Ollama

```bash
ollama serve
```

---

## 6. Run Application

```bash
python app.py
```

---

## 7. Open in Browser

```text
http://127.0.0.1:8001
```

---

# Requirements

```txt
fastapi
uvicorn
python-multipart
pypdf
numpy
sentence-transformers
faiss-cpu
```

---

# Supported File Types

- PDF
- TXT
- Markdown (.md)

---

# Uploading Multiple Documents

You can upload multiple documents in two ways:

**Via file picker:** Click the upload zone and select multiple files at once (hold Ctrl or Cmd to multi-select).

**Via drag & drop:** Drag multiple files from your file explorer and drop them onto the upload zone.

Each file is processed and appended to the shared index. The sidebar shows all loaded filenames. You can reset the entire index using the ✕ button.

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "model": "mistral:latest",
  "vector_backend": "faiss",
  "documents": 2,
  "chunks": 87
}
```

---

## Upload Files

```http
POST /upload
```

Multipart form upload. Send one or more files under the `files` field in a single request.

Response:

```json
{
  "added": [{"id": "d0f0ccad", "filename": "doc1.pdf", "chunks": 42, "chars": 31000}],
  "skipped": [{"filename": "notes.docx", "reason": "Unsupported type '.docx'"}],
  "total_docs": 2,
  "total_chunks": 87,
  "embed_model": "all-MiniLM-L6-v2"
}
```

---

## List Documents

```http
GET /documents
```

Returns every indexed document with its id, filename, chunk count, and enabled state.

---

## Toggle a Document

```http
PATCH /documents/{doc_id}
```

Body: `{"enabled": false}` — excludes the document from retrieval without deleting it.

---

## Delete a Document

```http
DELETE /documents/{doc_id}
```

Removes a single document and rebuilds the vector index.

---

## Chat with Documents

```http
POST /chat
```

Request:

```json
{
  "question": "What is the document about?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

Response includes `answer` plus `sources` — a list of `{filename, text, score}` objects.

---

## Reset Memory

```http
DELETE /reset
```

Clears all loaded documents and the vector index.

---

# Example Workflow

## 1. Upload Documents

Select or drag multiple PDF/TXT/MD files. Each is processed and added to the shared index.

## 2. Ask Questions

```text
Summarize the key points across all documents.
```

## 3. Retrieval

The system retrieves the most relevant chunks from any of the loaded documents.

## 4. Final Response

Ollama generates a contextual answer using retrieved information.

---

# Application Screenshots

## Home Page

![Home](screenshots/home.png)

## Upload Document

![Upload](screenshots/upload.png)

## Chat Interface

![Chat](screenshots/chat.png)

---

# Advantages of Local RAG

## Privacy Friendly

Your documents never leave your system.

## No API Costs

No OpenAI or paid cloud APIs required.

## Offline Usage

Works completely offline after setup.

## Beginner Friendly

Simple and understandable implementation for learning RAG concepts.

---

# Current Limitations

- In-memory vector storage only (resets on server restart)
- No persistent database
- No authentication
- No streaming responses

---

# Future Improvements

- ChromaDB / persistent vector store
- Authentication system
- Streaming responses
- Docker deployment
- Hybrid search (semantic + keyword)
- Chat memory persistence

---

# Recommended Improvements

## Add Environment Variables

```python
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
```

## Add Logging

Replace print statements with proper logging.

## Add Docker Support

Containerize the project for deployment.

---

# Who Is This Project For?

- Beginners learning RAG
- Students learning LLMs
- Developers exploring Ollama
- AI engineers building document chatbots
- Anyone wanting offline AI systems

---

# Learning Outcomes

By building this project, you can learn:

- RAG architecture
- Semantic search
- Vector embeddings
- Cosine similarity
- FastAPI backend development
- Ollama integration
- Local LLM deployment
- Document processing pipelines
- Incremental embedding strategies

---
