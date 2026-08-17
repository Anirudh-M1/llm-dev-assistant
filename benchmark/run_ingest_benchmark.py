"""Benchmark: sequential vs. asyncio-concurrent embedding generation.

Run: python -m benchmark.run_ingest_benchmark
"""

import asyncio
import os
import time

from src.chunker import chunk_directory
from src.embeddings import DEFAULT_CONCURRENCY, embed_texts_async, embed_texts_sequential

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def main():
    chunks = chunk_directory(CORPUS_DIR)
    texts = [c.source for c in chunks]
    print(f"Benchmarking ingestion over {len(texts)} chunks from {CORPUS_DIR}")

    start = time.perf_counter()
    embed_texts_sequential(texts)
    sequential_seconds = time.perf_counter() - start

    start = time.perf_counter()
    asyncio.run(embed_texts_async(texts, concurrency=DEFAULT_CONCURRENCY))
    async_seconds = time.perf_counter() - start

    speedup = sequential_seconds / async_seconds if async_seconds else float("inf")

    print(f"Sequential:                    {sequential_seconds:.2f}s")
    print(f"Async (concurrency={DEFAULT_CONCURRENCY}):          {async_seconds:.2f}s")
    print(f"Speedup:                       {speedup:.2f}x")


if __name__ == "__main__":
    main()
