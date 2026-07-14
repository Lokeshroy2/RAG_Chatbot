from rag_chatbot.llm import (
    NO_ANSWER_SENTENCE,
    OllamaClient,
    OllamaError,
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


class FakeClient(OllamaClient):
    def __init__(self, reply=None, error=None):
        super().__init__("http://localhost:11434", "fake")
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


def test_condense_question_falls_back_on_ollama_error():
    client = FakeClient(error=OllamaError(503, "down"))
    assert condense_question(client, "how fast is it?", HISTORY) == "how fast is it?"


def test_condense_question_rejects_implausible_rewrites():
    assert condense_question(FakeClient(reply=""), "how fast?", HISTORY) == "how fast?"
    assert condense_question(FakeClient(reply="word " * 500), "how fast?", HISTORY) == "how fast?"
