from rag_chatbot.chunking import _overlap_tail, chunk_text, extract_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]


def test_whitespace_is_normalized():
    chunks = chunk_text("Hello\n\n  world.\tThis   is fine.")
    assert chunks == ["Hello world. This is fine."]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 * 1.5 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=60)
    # the start of chunk N+1 must repeat text from the end of chunk N
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        overlap_start = nxt.split(" ")[0]
        assert overlap_start in prev


def test_overlap_tail_starts_on_word_boundary():
    tail = _overlap_tail("the quick brown fox jumps over the lazy dog", 15)
    assert not tail.startswith(" ")
    assert tail in "the quick brown fox jumps over the lazy dog"
    # never begins mid-word: the char before the tail in the source is a space
    assert tail == "over the lazy dog"[-len(tail):]


def test_text_without_sentence_breaks_is_hard_split():
    text = "x" * 5000
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 800 * 1.5 for c in chunks)


def test_no_content_is_lost():
    text = " ".join(f"Word{i}." for i in range(500))
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    for i in range(500):
        assert any(f"Word{i}." in c for c in chunks)


def test_extract_text_decodes_utf8():
    assert extract_text("héllo wörld".encode(), "note.txt") == "héllo wörld"


def test_extract_text_survives_bad_bytes():
    out = extract_text(b"\xff\xfe broken", "note.txt")
    assert isinstance(out, str)
