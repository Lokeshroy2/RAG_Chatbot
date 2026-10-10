"""Background job queue for document ingestion, so a large upload doesn't
block the HTTP request that triggered it.

The worker runs as a background thread inside this same process (via RQ's
SimpleWorker), not a separate OS process — see the module-level reasoning in
app.py's lifespan / the project's roadmap notes: VectorStore is a single
in-memory object with one threading.Lock, and a genuinely separate worker
process would need two processes safely sharing one set of flat files, which
that store was never built for. SimpleWorker also happens to be required on
Windows regardless, since the default Worker class depends on os.fork().

Redis still does real work here: jobs are durably queued (they survive an API
restart) and have inspectable status via RQ's Job API.
"""

import logging
import threading

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.timeouts import TimerDeathPenalty
from rq.worker import SimpleWorker

from rag_chatbot import embeddings
from rag_chatbot.chunking import chunk_text, extract_text
from rag_chatbot.config import settings

log = logging.getLogger("rag.jobs")

redis_conn: Redis | None = Redis.from_url(settings.redis_url) if settings.redis_url else None
queue: Queue | None = Queue("uploads", connection=redis_conn) if redis_conn else None

_store = None  # set via configure() from app.py's lifespan — avoids a circular import


def configure(store) -> None:
    global _store
    _store = store


def process_upload(filename: str, raw: bytes, owner_id: str) -> dict:
    """The actual slow work: extract -> chunk -> embed -> store.

    Runs on the worker thread, against the *same* VectorStore instance the
    API uses (set via configure()) — not a copy, not a reload from disk.
    """
    text = extract_text(raw, filename)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No extractable text")
    vecs = embeddings.embed_texts(chunks)
    doc_id = _store.add_document(filename, chunks, vecs, owner_id)
    log.info(
        "Indexed %s: %d chars -> %d chunks (id=%s)", filename, len(text), len(chunks), doc_id
    )
    return {"id": doc_id, "filename": filename, "chunks": len(chunks), "chars": len(text)}


def start_worker() -> None:
    """Run an RQ SimpleWorker on a daemon thread for this process's lifetime."""
    def _run():
        worker = SimpleWorker([queue], connection=redis_conn)
        # death_penalty_class defaults to RQ's SIGALRM-based job timeout, a
        # Unix-only signal Windows doesn't have at all — and signals don't
        # work off the main thread regardless of OS. TimerDeathPenalty gives
        # the same per-job timeout enforcement via threading.Timer instead.
        # It's a class attribute, not a constructor argument, in this RQ version.
        worker.death_penalty_class = TimerDeathPenalty
        # work() also unconditionally tries to install SIGINT/SIGTERM handlers,
        # which Python only allows from the main thread. We don't need graceful
        # signal handling here anyway: this is a daemon thread, so the process
        # exiting kills it regardless; any in-flight job just gets re-run from
        # the queue next startup.
        worker._install_signal_handlers = lambda: None
        worker.work(with_scheduler=False)

    threading.Thread(target=_run, daemon=True, name="rq-worker").start()
    log.info("Background upload worker started (queue=%s).", queue.name)


def get_job(job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return None


__all__ = ["configure", "get_job", "process_upload", "queue", "start_worker"]
