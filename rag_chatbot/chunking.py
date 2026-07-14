"""Text extraction and sentence-aware chunking. Pure functions, no app state."""

import io
import re
from pathlib import Path

try:
    from pypdf import PdfReader

    HAS_PDF = True
except ImportError:
    HAS_PDF = False

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileError(Exception):
    pass


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if not HAS_PDF:
            raise UnsupportedFileError("pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="replace")


def _overlap_tail(text: str, overlap: int) -> str:
    """Last `overlap` characters of `text`, trimmed to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Pack sentences into ~chunk_size character chunks with word-aligned overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current)
            tail = _overlap_tail(current, chunk_overlap)
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)

    # hard-split anything still oversized (e.g. text with no sentence breaks)
    out: list[str] = []
    for chunk in chunks:
        while len(chunk) > chunk_size * 1.5:
            out.append(chunk[:chunk_size])
            chunk = chunk[chunk_size - chunk_overlap :]
        out.append(chunk)
    return [c.strip() for c in out if c.strip()]
