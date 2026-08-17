"""Evaluation harness: retrieval accuracy + latency over a labeled question set.

Uses the exact same pipeline as src/main.py and src/api.py (chunk_directory,
embed_texts_async, build_faiss_index, retrieve_chunks, explain_code) -- no
separate eval-only implementation to drift out of sync with production code.

Run: python -m eval.run
"""

import asyncio
import json
import os
import subprocess
import time

from src.chunker import chunk_directory
from src.embeddings import EMBED_MODEL, embed_texts_async
from src.index_store import build_faiss_index
from src.main import explain_code, retrieve_chunks

CORPUS_DIR = "corpus"
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "RESULTS.md")
K_VALUES = (1, 3, 5)
GENERATION_MODEL = "llama3"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index():
    chunks = chunk_directory(CORPUS_DIR)
    embeddings = asyncio.run(embed_texts_async([c.source for c in chunks]))
    index = build_faiss_index(embeddings)
    return chunks, index


def evaluate_retrieval(questions, chunks, index):
    max_k = max(K_VALUES)
    ranks = []
    for q in questions:
        retrieved = retrieve_chunks(q["question"], chunks, index, k=max_k)
        rank = None
        for i, c in enumerate(retrieved, start=1):
            if c.file_path == q["file"] and c.qualified_name == q["qualified_name"]:
                rank = i
                break
        ranks.append(rank)

    metrics = {}
    for k in K_VALUES:
        hits = [r is not None and r <= k for r in ranks]
        recall = sum(hits) / len(ranks)
        mrr = sum((1.0 / r) if (r is not None and r <= k) else 0.0 for r in ranks) / len(ranks)
        metrics[k] = {"recall": recall, "mrr": mrr}
    return metrics, ranks


def measure_latency(questions, chunks, index):
    retrieval_ms, generation_ms = [], []
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        retrieved = retrieve_chunks(q["question"], chunks, index, k=3)
        retrieval_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        explain_code(retrieved, q["question"])
        generation_ms.append((time.perf_counter() - start) * 1000)
        print(f"  latency sample {i}/{len(questions)}")
    return retrieval_ms, generation_ms


def percentile(data, pct):
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def write_results(commit_sha, date, corpus_size, question_count, metrics, retrieval_ms, generation_ms):
    lines = [
        "# Evaluation results",
        "",
        f"- Commit: `{commit_sha}`",
        f"- Date: {date}",
        f"- Embedding model: `{EMBED_MODEL}` (Ollama)",
        f"- Generation model: `{GENERATION_MODEL}` (Ollama)",
        f"- Corpus size: {corpus_size} chunks (`corpus/`)",
        f"- Question set: {question_count} questions (`eval/questions.jsonl`)",
        "- Machine: single local machine, sequential (one query at a time, not concurrent)",
        "",
        "Reproduce: `python -m eval.run`",
        "",
        "## Retrieval accuracy",
        "",
        "Recall@k: fraction of questions where the ground-truth chunk appears in the top-k "
        "retrieved results. MRR@k: mean reciprocal rank of the ground-truth chunk, 0 if it "
        "doesn't appear in the top-k.",
        "",
        "| k | Recall@k | MRR@k |",
        "|---|---|---|",
    ]
    for k in K_VALUES:
        lines.append(f"| {k} | {metrics[k]['recall']:.2f} | {metrics[k]['mrr']:.2f} |")

    lines += [
        "",
        "## Latency",
        "",
        "Retrieval (embed query + FAISS search) and generation (full Llama3 completion via "
        "Ollama) are reported separately -- they are different orders of magnitude and "
        "conflating them would hide what's actually slow.",
        "",
        "| Stage | p50 | p95 |",
        "|---|---|---|",
        f"| Retrieval | {percentile(retrieval_ms, 50):.1f}ms | {percentile(retrieval_ms, 95):.1f}ms |",
        f"| Generation | {percentile(generation_ms, 50):.1f}ms | {percentile(generation_ms, 95):.1f}ms |",
        "",
    ]

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    date = subprocess.check_output(
        ["git", "log", "-1", "--format=%cd", "--date=short"]
    ).decode().strip()

    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    chunks, index = build_index()
    print(f"Indexed {len(chunks)} chunks from {CORPUS_DIR}")

    print("Evaluating retrieval accuracy...")
    metrics, _ranks = evaluate_retrieval(questions, chunks, index)
    for k in K_VALUES:
        print(f"  Recall@{k}: {metrics[k]['recall']:.2f}  MRR@{k}: {metrics[k]['mrr']:.2f}")

    print("Measuring latency (retrieval + generation per question)...")
    retrieval_ms, generation_ms = measure_latency(questions, chunks, index)

    write_results(commit_sha, date, len(chunks), len(questions), metrics, retrieval_ms, generation_ms)
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
