"""Hosted-LLM clients (blocking + streaming) and prompt construction.

Groq is primary, Gemini is the fallback on rate-limit — no self-hosted
transport anymore (Ollama needs RAM no free host provides).

Framework-agnostic: raises LLMError; the API layer maps it to HTTP errors.
History messages are plain dicts: {"role": "user"|"assistant", "content": str}.
"""

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator

from rag_chatbot import __version__

log = logging.getLogger("rag.llm")

# Without an explicit User-Agent, urllib sends "Python-urllib/x.y", which
# Cloudflare's bot protection (sitting in front of most hosted LLM APIs)
# blocks outright with a 403 before the request ever reaches the provider.
USER_AGENT = f"rag-chatbot/{__version__}"


class LLMError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class LLMClient:
    """Common interface implemented by GroqClient, GeminiClient, and FallbackLLMClient."""

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        raise NotImplementedError


class GroqClient(LLMClient):
    """OpenAI-compatible chat-completions API."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, prompt: str, stream: bool, temperature: float):
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": stream,
            }
        ).encode()
        req = urllib.request.Request(
            self.URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            status = 429 if e.code == 429 else 502
            raise LLMError(status, f"Groq error ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise LLMError(503, f"Groq not reachable: {e}") from e

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        with self._request(prompt, stream=False, temperature=temperature) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        """Groq streams OpenAI-style SSE lines: `data: {...}`, ending `data: [DONE]`."""
        with self._request(prompt, stream=True, temperature=temperature) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


class GeminiClient(LLMClient):
    """Google's Generative Language REST API (v1beta)."""

    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _url(self, method: str) -> str:
        return f"{self.BASE}/{self.model}:{method}?key={self.api_key}"

    def _body(self, prompt: str, temperature: float) -> bytes:
        return json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            }
        ).encode()

    def _post(self, url: str, body: bytes):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:300]
            status = 429 if e.code == 429 else 502
            raise LLMError(status, f"Gemini error ({e.code}): {body_text}") from e
        except urllib.error.URLError as e:
            raise LLMError(503, f"Gemini not reachable: {e}") from e

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        with self._post(self._url("generateContent"), self._body(prompt, temperature)) as resp:
            data = json.loads(resp.read())
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(502, f"Unexpected Gemini response shape: {data}") from e

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        """Gemini's SSE stream (`alt=sse`) wraps each chunk in `data: {...}`."""
        url = self._url("streamGenerateContent") + "&alt=sse"
        with self._post(url, self._body(prompt, temperature)) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload:
                    continue
                chunk = json.loads(payload)
                try:
                    text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    continue
                if text:
                    yield text


class FallbackLLMClient(LLMClient):
    """Tries the primary client; retries once against the fallback on a 429.

    Safe for streaming specifically because a rate-limit is returned by the
    provider before any response body streams — the primary never yields a
    partial answer before failing, so switching clients can't duplicate text.
    A failure for any other reason propagates instead of silently falling back.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            return self.primary.generate(prompt, temperature)
        except LLMError as e:
            if e.status != 429:
                raise
            return self.fallback.generate(prompt, temperature)

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        try:
            yield from self.primary.generate_stream(prompt, temperature)
        except LLMError as e:
            if e.status != 429:
                raise
            yield from self.fallback.generate_stream(prompt, temperature)


# ── prompt construction (pure functions) ──────────────────────────────────────

NO_ANSWER_SENTENCE = "I don't have enough information in the documents to answer that."


def trim_history(
    history: list[dict], max_messages: int = 6, char_budget: int = 4000
) -> list[dict]:
    """Keep the most recent messages within both a count and a character budget."""
    recent = history[-max_messages:] if max_messages else []
    trimmed: list[dict] = []
    total = 0
    for msg in reversed(recent):
        total += len(msg["content"])
        if trimmed and total > char_budget:
            break
        trimmed.append(msg)
    return list(reversed(trimmed))


def format_history(history: list[dict]) -> str:
    return "".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n" for m in history
    )


def build_prompt(
    question: str,
    sources: list[dict],
    history: list[dict],
    max_messages: int = 6,
    char_budget: int = 4000,
) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (from {s['filename']})\n{s['text']}" for i, s in enumerate(sources)
    )
    history_str = format_history(trim_history(history, max_messages, char_budget))
    return (
        "You are a helpful assistant. Answer ONLY from the numbered document "
        "sources below. Cite sources inline like [1] or [2] where relevant.\n"
        "If the answer is not in the sources, say: "
        f"'{NO_ANSWER_SENTENCE}'\n\n"
        f"=== SOURCES ===\n{context}\n\n"
        f"=== CONVERSATION HISTORY ===\n{history_str}\n"
        f"=== QUESTION ===\nUser: {question}\nAssistant:"
    )


def build_condense_prompt(question: str, history: list[dict]) -> str:
    return (
        "Given the conversation below, rewrite the user's last question as a "
        "single standalone question that can be understood without the "
        "conversation. Keep it in the user's language and intent. If it is "
        "already standalone, return it unchanged.\n"
        "Return ONLY the rewritten question — no explanation, no quotes.\n\n"
        f"=== CONVERSATION ===\n{format_history(history)}\n"
        f"=== LAST QUESTION ===\n{question}\n\n"
        "Standalone question:"
    )


def condense_question(client: LLMClient, question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question for retrieval.

    Falls back to the original question if the LLM is unavailable or returns
    something implausible.
    """
    if not history:
        return question
    try:
        rewritten = client.generate(build_condense_prompt(question, history), temperature=0.0)
    except LLMError as e:
        log.warning(
            "Query condensing failed (%s) — retrieving on the raw question instead: %r",
            e.message, question,
        )
        return question
    rewritten = rewritten.strip().strip('"').strip()
    if not rewritten or len(rewritten) > max(4 * len(question), 300):
        return question
    return rewritten
