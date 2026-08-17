# Ingestion benchmark: sequential vs. async embedding generation

Reproduce: `python -m benchmark.generate_corpus && python -m benchmark.run_ingest_benchmark`

- Commit: `ed71ecc9b9d61199bf8c77a77b9465314e56372b`
- Date: 2026-08-16
- Machine: single local machine (macOS, Apple Silicon), Ollama serving `nomic-embed-text` on localhost
- Corpus: `benchmark/corpus/` — 60 synthetic files, 180 functions (generated fixture, not a real codebase)
- Concurrency: `asyncio.Semaphore(8)`, `ollama.AsyncClient`

## Raw runs

| Run | Sequential | Async (concurrency=8) | Speedup |
|---|---|---|---|
| 1 (cold — first model load in process) | 11.85s | 1.52s | 7.82x |
| 2 (warm) | 3.75s | 1.37s | 2.73x |
| 3 (warm) | 3.86s | 1.23s | 3.13x |

## Honest takeaway

Run 1 is inflated by Ollama loading `nomic-embed-text` into memory for the
first request of the process — that cost is paid once regardless of
sequential/async and shouldn't be read as the steady-state speedup. Runs 2–3
are the representative numbers: **~2.7–3.1x faster** with `asyncio` +
bounded semaphore (concurrency=8) than one HTTP request at a time, on a
180-chunk synthetic corpus, on this machine, against a local Ollama server.

This is not a "sub-second at scale" claim — it's a measured, reproducible
speedup on a fixed synthetic workload. Numbers will vary with corpus size,
concurrency setting, hardware, and whatever else is competing for the local
Ollama server's request queue at the time.
