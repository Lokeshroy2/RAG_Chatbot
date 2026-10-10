# Multi-stage build: the "builder" stage has pip and installs packages into a
# venv; the final stage copies only that venv + your source code, not pip's
# cache or any build tooling. That's the whole point of multi-stage — you
# ship the result of building, not the tools you built it with.
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# sentence-transformers pulls in torch as a dependency. PyPI's default torch
# wheel bundles CUDA libraries for a GPU that doesn't exist in a free-tier
# container — installing the CPU-only build first (from PyTorch's own index)
# means pip sees torch already satisfied later and skips the ~2GB GPU version.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied explicitly, not `COPY . .` — this is deliberate, not an oversight.
# A blanket copy would bake .env (and therefore your Postgres password and
# API keys) directly into an image layer. Anyone who ever pulls or inspects
# that image could extract them, even after you "delete" the file later —
# image layers don't forget. Only copy what the app actually needs to run.
COPY app.py .
COPY rag_chatbot/ rag_chatbot/
COPY frontend/ frontend/

# 127.0.0.1 (the app's local-dev default) means "only reachable from inside
# this exact process' network namespace" — i.e. unreachable from outside the
# container even with a port published. 0.0.0.0 means "listen on every
# interface," which is what lets Docker's port mapping actually reach it.
ENV RAG_HOST=0.0.0.0
ENV RAG_OPEN_BROWSER=false

EXPOSE 8001

CMD ["python", "app.py"]
