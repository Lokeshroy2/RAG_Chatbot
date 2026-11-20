"""RAGAS-based answer-quality evaluation.

run_eval.py checks retrieval quality alone (no LLM, free, fast). This script
goes one step further: it runs the real chunk -> embed -> search -> prompt ->
LLM pipeline the app uses, then scores the generated answers with RAGAS:

  - faithfulness     : is every claim in the answer supported by the
                       retrieved context? (catches hallucination)
  - answer_relevancy : does the answer actually address the question asked?
                       (catches technically-grounded-but-off-topic answers)

Both metrics use the same hosted LLM (Groq, the app's primary) as judge, and
the same sentence-transformers model the app uses for embeddings - no new
external dependency on the answer-generation side, only on the judging side.

Requires a real RAG_GROQ_API_KEY (costs a handful of LLM calls per question).
Run:  python -m eval.run_ragas_eval        (from the repo root)
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# truststore fixes TLS verification for httpx-based clients (openai SDK,
# used by ragas) on machines where the OS trusts a locally-injected root CA
# (e.g. antivirus HTTPS scanning) that the bundled certifi store doesn't.
# Harmless and a no-op on machines without this issue.
import truststore

truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402
from ragas.embeddings import HuggingFaceEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, Faithfulness  # noqa: E402

from rag_chatbot import embeddings  # noqa: E402
from rag_chatbot.chunking import chunk_text  # noqa: E402
from rag_chatbot.config import settings  # noqa: E402
from rag_chatbot.llm import FallbackLLMClient, GeminiClient, GroqClient, build_prompt  # noqa: E402
from rag_chatbot.store import VectorStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.json"

# chosen floors: below this, answers are hallucinating or off-topic often
# enough to not trust in front of a user
MIN_FAITHFULNESS = 0.7
MIN_ANSWER_RELEVANCY = 0.7


def build_store() -> VectorStore:
    store = VectorStore(embeddings.embedding_dim())
    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks = chunk_text(
            path.read_text(encoding="utf-8"), settings.chunk_size, settings.chunk_overlap
        )
        store.add_document(path.name, chunks, embeddings.embed_texts(chunks))
    return store


async def main() -> int:
    if not settings.groq_api_key:
        print("RAG_GROQ_API_KEY is not set - this eval makes real LLM calls and needs it.")
        return 1

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    store = build_store()

    # the exact same answer-generation path app.py uses for a real chat turn
    answer_client = FallbackLLMClient(
        primary=GroqClient(settings.groq_api_key, settings.groq_model, settings.llm_timeout),
        fallback=GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout),
    )

    judge_client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
    ragas_llm = llm_factory(model=settings.groq_model, provider="openai", client=judge_client)
    ragas_embeddings = HuggingFaceEmbeddings(
        model=settings.embed_model_name, normalize_embeddings=True
    )
    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    print(f"Evaluating {len(golden)} questions (answer model: {settings.groq_model}, "
          f"judge model: {settings.groq_model})\n")

    faithfulness_scores = []
    relevancy_scores = []

    for case in golden:
        question = case["question"]
        q_vec = embeddings.embed_texts([question])[0]
        results = store.search(
            q_vec, top_k=settings.top_k, score_threshold=settings.score_threshold
        )
        contexts = [r["text"] for r in results]

        prompt = build_prompt(question, results, history=[])
        answer = answer_client.generate(prompt, temperature=0.0)

        f_score = (await faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )).value
        r_score = (await answer_relevancy.ascore(user_input=question, response=answer)).value

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        print(f"  faithfulness={f_score:.2f}  relevancy={r_score:.2f}  {question}")
        time.sleep(10)  # stay under Groq's free-tier rate limit across the ~3 calls/question above

    n = len(golden)
    mean_faithfulness = sum(faithfulness_scores) / n
    mean_relevancy = sum(relevancy_scores) / n
    print(f"\nMean faithfulness:     {mean_faithfulness:.3f}")
    print(f"Mean answer relevancy: {mean_relevancy:.3f}")

    if mean_faithfulness < MIN_FAITHFULNESS or mean_relevancy < MIN_ANSWER_RELEVANCY:
        print(f"\nFAILED: below floor (faithfulness >= {MIN_FAITHFULNESS}, "
              f"answer relevancy >= {MIN_ANSWER_RELEVANCY})")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
