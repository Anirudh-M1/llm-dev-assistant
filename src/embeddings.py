"""Embedding generation via Ollama's HTTP embed endpoint.

Uses ollama's embedding-capable model (nomic-embed-text) over HTTP rather
than an in-process model, so that bulk ingestion can genuinely be
parallelized with asyncio: concurrent HTTP requests to Ollama's local
server, bounded by a semaphore. Local in-process embedding compute is
CPU/GPU-bound and asyncio cannot parallelize that without a thread pool,
so this is the honest way to make "async ingestion" real rather than
asyncio syntax wrapped around a thread pool.
"""

import asyncio

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
DEFAULT_CONCURRENCY = 8


def embed_text(text):
    """Embed a single string synchronously. Used for one-off interactive queries."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return np.array(response["embeddings"][0], dtype="float32")


def embed_texts_sequential(texts):
    """Embed many strings one HTTP request at a time. Baseline for the async benchmark."""
    embeddings = [ollama.embed(model=EMBED_MODEL, input=t)["embeddings"][0] for t in texts]
    return np.array(embeddings, dtype="float32")


async def embed_texts_async(texts, concurrency=DEFAULT_CONCURRENCY):
    """Embed many strings concurrently over HTTP, bounded by a semaphore."""
    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(concurrency)

    async def embed_one(text):
        async with semaphore:
            response = await client.embed(model=EMBED_MODEL, input=text)
            return response["embeddings"][0]

    results = await asyncio.gather(*(embed_one(t) for t in texts))
    return np.array(results, dtype="float32")
