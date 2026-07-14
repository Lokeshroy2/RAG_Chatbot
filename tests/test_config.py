from rag_chatbot.config import Settings


def test_defaults():
    s = Settings()
    assert s.port == 8001
    assert s.ollama_model == "mistral:latest"
    assert s.chunk_size == 800
    assert s.condense_queries is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_PORT", "9000")
    monkeypatch.setenv("RAG_OLLAMA_MODEL", "llama3:8b")
    monkeypatch.setenv("RAG_OLLAMA_URL", "http://other-host:11434/")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.4")
    monkeypatch.setenv("RAG_CONDENSE_QUERIES", "false")
    monkeypatch.setenv("RAG_PERSIST", "0")
    s = Settings.from_env()
    assert s.port == 9000
    assert s.ollama_model == "llama3:8b"
    assert s.ollama_url == "http://other-host:11434"  # trailing slash stripped
    assert s.score_threshold == 0.4
    assert s.condense_queries is False
    assert s.persist is False


def test_bool_parsing(monkeypatch):
    for raw, expected in [("1", True), ("true", True), ("YES", True), ("0", False), ("no", False)]:
        monkeypatch.setenv("RAG_CONDENSE_QUERIES", raw)
        assert Settings.from_env().condense_queries is expected
