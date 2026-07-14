"""Retrieval quality evaluation.

Indexes the documents in eval/corpus/, runs every question in eval/golden.json
through the same chunk -> embed -> search pipeline the app uses, and reports:

  - hit@k         : the expected source file appears in the top-k results
  - keyword recall: the chunk containing the expected answer text is retrieved
  - MRR           : mean reciprocal rank of the first hit from the expected file

Run:  python -m eval.run_eval        (from the repo root)
Requires the embedding model (downloads ~80 MB on first run). No LLM needed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_chatbot import embeddings  # noqa: E402
from rag_chatbot.chunking import chunk_text  # noqa: E402
from rag_chatbot.config import settings  # noqa: E402
from rag_chatbot.store import VectorStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.json"

# fail the run if retrieval quality drops below these floors
MIN_HIT_RATE = 0.9
MIN_KEYWORD_RECALL = 0.8


def build_store() -> VectorStore:
    store = VectorStore(embeddings.embedding_dim())
    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks = chunk_text(path.read_text(encoding="utf-8"),
                            settings.chunk_size, settings.chunk_overlap)
        store.add_document(path.name, chunks, embeddings.embed_texts(chunks))
    return store


def main() -> int:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    store = build_store()
    print(f"Indexed {len(store.list_documents())} docs / {len(store.meta)} chunks. "
          f"Evaluating {len(golden)} questions (top_k={settings.top_k}, "
          f"threshold={settings.score_threshold})\n")

    hits = kw_hits = 0
    reciprocal_ranks = []
    failures = []

    for case in golden:
        q_vec = embeddings.embed_texts([case["question"]])[0]
        results = store.search(q_vec, settings.top_k, settings.score_threshold)

        rank = next(
            (i + 1 for i, r in enumerate(results) if r["filename"] == case["expected_file"]),
            None,
        )
        file_hit = rank is not None
        retrieved_text = " ".join(r["text"] for r in results)
        kw_hit = all(kw.lower() in retrieved_text.lower() for kw in case["expected_keywords"])

        hits += file_hit
        kw_hits += kw_hit
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        status = "PASS" if (file_hit and kw_hit) else "FAIL"
        if status == "FAIL":
            failures.append(case["question"])
        top = f"{results[0]['filename']} ({results[0]['score']:.2f})" if results else "—"
        print(f"  [{status}] {case['question']}\n"
              f"         expected {case['expected_file']}, top hit: {top}")

    n = len(golden)
    hit_rate = hits / n
    kw_recall = kw_hits / n
    mrr = sum(reciprocal_ranks) / n
    print(f"\nhit@{settings.top_k}:         {hit_rate:.0%}  ({hits}/{n})")
    print(f"keyword recall: {kw_recall:.0%}  ({kw_hits}/{n})")
    print(f"MRR:            {mrr:.3f}")

    if hit_rate < MIN_HIT_RATE or kw_recall < MIN_KEYWORD_RECALL:
        print(f"\nFAILED: below floor (hit@k >= {MIN_HIT_RATE:.0%}, "
              f"keyword recall >= {MIN_KEYWORD_RECALL:.0%})")
        for q in failures:
            print(f"  - {q}")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
