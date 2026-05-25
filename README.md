# Local RAG Chatbot using Ollama + FastAPI

A fully local and 100% free Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, Ollama, and Sentence Transformers.

This project allows users to upload documents (PDF/TXT/Markdown) and ask questions about them using a local Large Language Model (LLM) without requiring any paid API keys or cloud services.

---

# Demo

![Demo](screenshots/demo.gif)

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

- Upload PDF, TXT, and Markdown files
- Semantic document retrieval
- Local LLM inference using Ollama
- Conversation history support
- FastAPI backend
- Simple frontend UI
- Fully offline setup
- No external APIs
- Lightweight and easy to run

---

# Architecture Overview

```text
                ┌──────────────────┐
                │   Upload File    │
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
                │ Generate Embeds  │
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

## Step 1 — Upload Document

The user uploads a PDF, TXT, or Markdown file.

---

## Step 2 — Text Extraction

The backend extracts raw text from the uploaded file.

Example:

```text
PDF → Extracted Text
```

---

## Step 3 — Chunking

Large text is split into smaller chunks.

Example:

```text
Chunk 1
Chunk 2
Chunk 3
```

This improves retrieval accuracy.

---

## Step 4 — Embedding Generation

Each chunk is converted into vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

## Step 5 — Semantic Search

When the user asks a question:

1. The question is converted into embeddings
2. Cosine similarity is calculated
3. Most relevant chunks are retrieved

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
| NumPy | Vector similarity calculations |
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

Download Ollama:

https://ollama.com

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
sentence-transformers
numpy
```

---

# Supported File Types

- PDF
- TXT
- Markdown (.md)

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Upload File

```http
POST /upload
```

Uploads and processes a document.

---

## Chat with Document

```http
POST /chat
```

Request:

```json
{
  "question": "What is the document about?"
}
```

---

## Reset Memory

```http
DELETE /reset
```

Clears loaded document embeddings.

---

# Example Workflow

## 1. Upload Document

Upload a PDF file.

---

## 2. Ask Questions

Example:

```text
What are the key points in this document?
```

---

## 3. Retrieval

The system retrieves relevant chunks.

---

## 4. Final Response

Ollama generates a contextual answer using retrieved information.

---

# Application Screenshots

## Home Page

![Home](screenshots/home.png)

---

## Upload Document

![Upload](screenshots/upload.png)

---

## Chat Interface

![Chat](screenshots/chat.png)

---

# Advantages of Local RAG

## Privacy Friendly

Your documents never leave your system.

---

## No API Costs

No OpenAI or paid cloud APIs required.

---

## Offline Usage

Works completely offline after setup.

---

## Beginner Friendly

Simple and understandable implementation for learning RAG concepts.

---

# Current Limitations

- In-memory vector storage only
- Single document support
- No persistent database
- Basic frontend UI
- No authentication
- No streaming responses

---

# Future Improvements

- FAISS integration
- ChromaDB support
- Multi-document retrieval
- Better frontend UI
- Authentication system
- Streaming responses
- Docker deployment
- LangChain integration
- Hybrid search
- Chat memory persistence

---

# Recommended Improvements

## Add Environment Variables

Instead of hardcoding model names:

```python
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
```

---

## Add Logging

Replace print statements with proper logging.

---

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

---

# License

MIT License

---

# Acknowledgements

- Ollama
- FastAPI
- Sentence Transformers
- Mistral AI

---

# Star This Repository

If you found this project useful, consider giving it a star.

```text
⭐ Helps support and improve the project
```