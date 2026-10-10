import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from rag_chatbot.llm import (
    NO_ANSWER_SENTENCE,
    FallbackLLMClient,
    GeminiClient,
    GroqClient,
    LLMClient,
    LLMError,
    build_condense_prompt,
    build_prompt,
    condense_question,
    trim_history,
)

SOURCES = [
    {"filename": "a.txt", "text": "Alpha content.", "score": 0.9},
    {"filename": "b.txt", "text": "Beta content.", "score": 0.8},
]


def msg(role, content):
    return {"role": role, "content": content}


def test_build_prompt_numbers_sources():
    prompt = build_prompt("What is alpha?", SOURCES, [])
    assert "[1] (from a.txt)\nAlpha content." in prompt
    assert "[2] (from b.txt)\nBeta content." in prompt
    assert "What is alpha?" in prompt
    assert NO_ANSWER_SENTENCE in prompt


def test_build_prompt_includes_recent_history_only():
    history = [msg("user", f"question {i}") for i in range(10)]
    prompt = build_prompt("q", SOURCES, history, max_messages=4)
    assert "question 9" in prompt
    assert "question 5" not in prompt


def test_trim_history_respects_char_budget():
    history = [msg("user", "x" * 500) for _ in range(6)]
    trimmed = trim_history(history, max_messages=6, char_budget=1200)
    assert len(trimmed) == 2
    # most recent messages are the ones kept
    assert trimmed == history[-2:]


def test_trim_history_always_keeps_last_message():
    history = [msg("user", "x" * 9999)]
    assert trim_history(history, max_messages=6, char_budget=100) == history


def test_condense_prompt_contains_history_and_question():
    history = [msg("user", "What is FAISS?"), msg("assistant", "A vector library.")]
    prompt = build_condense_prompt("how fast is it?", history)
    assert "What is FAISS?" in prompt
    assert "how fast is it?" in prompt


class FakeClient(LLMClient):
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def generate(self, prompt, temperature=0.2):
        if self.error:
            raise self.error
        return self.reply


HISTORY = [msg("user", "What is FAISS?"), msg("assistant", "A vector search library.")]


def test_condense_question_uses_llm_rewrite():
    client = FakeClient(reply="How fast is FAISS?")
    assert condense_question(client, "how fast is it?", HISTORY) == "How fast is FAISS?"


def test_condense_question_skips_when_no_history():
    client = FakeClient(reply="SHOULD NOT BE USED")
    assert condense_question(client, "What is FAISS?", []) == "What is FAISS?"


def test_condense_question_falls_back_on_llm_error():
    client = FakeClient(error=LLMError(503, "down"))
    assert condense_question(client, "how fast is it?", HISTORY) == "how fast is it?"


def test_condense_question_rejects_implausible_rewrites():
    assert condense_question(FakeClient(reply=""), "how fast?", HISTORY) == "how fast?"
    assert condense_question(FakeClient(reply="word " * 500), "how fast?", HISTORY) == "how fast?"


# ── FallbackLLMClient ────────────────────────────────────────────────────────


def test_fallback_uses_primary_when_healthy():
    client = FallbackLLMClient(FakeClient(reply="from primary"), FakeClient(reply="from fallback"))
    assert client.generate("hi") == "from primary"


def test_fallback_switches_to_secondary_on_429():
    client = FallbackLLMClient(
        FakeClient(error=LLMError(429, "rate limited")), FakeClient(reply="from fallback")
    )
    assert client.generate("hi") == "from fallback"


def test_fallback_propagates_non_429_errors():
    client = FallbackLLMClient(
        FakeClient(error=LLMError(503, "down")), FakeClient(reply="from fallback")
    )
    with pytest.raises(LLMError) as exc_info:
        client.generate("hi")
    assert exc_info.value.status == 503


# ── GroqClient / GeminiClient wire format ───────────────────────────────────
# Mocked at the urllib layer — these catch JSON-shape bugs before a live call does.


class _FakeResponse:
    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None):
        self._body = body
        self._lines = lines or []

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_groq_generate_parses_openai_shaped_response():
    body = json.dumps({"choices": [{"message": {"content": "  Paris.  "}}]}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        assert GroqClient("key", "model").generate("capital of France?") == "Paris."


def test_groq_generate_stream_parses_sse_lines():
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
        b"data: [DONE]\n",
    ]
    with patch("urllib.request.urlopen", return_value=_FakeResponse(lines=lines)):
        assert "".join(GroqClient("key", "model").generate_stream("hi")) == "Hello"


def test_groq_maps_429_to_rate_limit_status():
    err = urllib.error.HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO(b"slow down"))
    with patch("urllib.request.urlopen", side_effect=err), pytest.raises(LLMError) as exc_info:
        GroqClient("key", "model").generate("hi")
    assert exc_info.value.status == 429


def test_gemini_generate_parses_candidate_shaped_response():
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": "Paris."}]}}]}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        assert GeminiClient("key", "model").generate("capital of France?") == "Paris."


def test_gemini_generate_stream_parses_sse_lines():
    lines = [
        b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n',
        b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n',
    ]
    with patch("urllib.request.urlopen", return_value=_FakeResponse(lines=lines)):
        assert "".join(GeminiClient("key", "model").generate_stream("hi")) == "Hello"


def test_gemini_maps_429_to_rate_limit_status():
    err = urllib.error.HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO(b"slow down"))
    with patch("urllib.request.urlopen", side_effect=err), pytest.raises(LLMError) as exc_info:
        GeminiClient("key", "model").generate("hi")
    assert exc_info.value.status == 429
