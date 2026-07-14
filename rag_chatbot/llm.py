"""Ollama client (blocking + streaming) and prompt construction.

Framework-agnostic: raises OllamaError; the API layer maps it to HTTP errors.
History messages are plain dicts: {"role": "user"|"assistant", "content": str}.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Iterator


class OllamaError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, prompt: str, stream: bool, temperature: float):
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {"temperature": temperature},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(502, f"Ollama error ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise OllamaError(503, f"Ollama not reachable. Is it running? Error: {e}") from e

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        with self._request(prompt, stream=False, temperature=temperature) as resp:
            data = json.loads(resp.read())
        if "error" in data:
            raise OllamaError(502, f"Ollama error: {data['error']}")
        return data.get("response", "").strip()

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        """Yield response text fragments as Ollama produces them."""
        with self._request(prompt, stream=True, temperature=temperature) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise OllamaError(502, f"Ollama error: {data['error']}")
                if data.get("response"):
                    yield data["response"]
                if data.get("done"):
                    break

    def list_models(self) -> list[str]:
        req = urllib.request.Request(f"{self.base_url}/api/tags")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(503, f"Ollama not reachable: {e}") from e
        return [m.get("name", "") for m in data.get("models", [])]


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


def condense_question(client: OllamaClient, question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question for retrieval.

    Falls back to the original question if the LLM is unavailable or returns
    something implausible.
    """
    if not history:
        return question
    try:
        rewritten = client.generate(build_condense_prompt(question, history), temperature=0.0)
    except OllamaError:
        return question
    rewritten = rewritten.strip().strip('"').strip()
    if not rewritten or len(rewritten) > max(4 * len(question), 300):
        return question
    return rewritten
