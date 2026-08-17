# Multi-repo concurrent retrieval latency benchmark

Reproduce:
```
python -m benchmark.generate_multi_repo_corpus
python -m benchmark.run_latency_benchmark
```

- Commit: `debd9cf8a9a5fe1f166d87547414e76d47a1bbcb`
- Date: 2026-08-16
- Machine: single local machine (macOS, Apple Silicon), real `uvicorn` server (`src.api:app`), Ollama serving `nomic-embed-text` on localhost
- Repos: 12 separate synthetic repos (`benchmark/multi_repo_corpus/repo_0` .. `repo_11`), each ingested under its own `repo_id` via `POST /ingest/{repo_id}` into an isolated FAISS index (`src/repo_store.py`)
- Load: 5 rounds, 12 genuinely concurrent `POST /query/{repo_id}` requests per round (`asyncio.gather` + `httpx.AsyncClient`, real HTTP, not in-process calls) — 60 samples total
- `include_explanation=False` on every request, so this measures retrieval (embed query via Ollama + FAISS search) in isolation from LLM generation, which is a different order of magnitude (~20s locally) and reported separately in `benchmark/RESULTS.md` / `eval/RESULTS.md`

## Results (2 runs)

| Run | Metric | p50 | p95 | max |
|---|---|---|---|---|
| 1 | server-side retrieval_ms | 60.39ms | 88.61ms | 95.57ms |
| 1 | client wall-clock (HTTP round trip) | 73.18ms | 102.63ms | — |
| 2 | server-side retrieval_ms | 60.14ms | 87.90ms | 93.84ms |
| 2 | client wall-clock (HTTP round trip) | 72.45ms | 102.44ms | — |

## Honest takeaway

Retrieval latency (query embedding via Ollama's `/api/embed` + FAISS search)
stayed **sub-second — around 60ms at p50 and under 100ms at p95** — under 12
concurrent requests spread across 12 independently-indexed repos, on this
machine. This supports "sub-second retrieval latency across 10+ concurrent
repositories" specifically for the retrieval step.

This does **not** claim sub-second end-to-end response time: the LLM
explanation step (local Ollama llama3) took ~20s per request in prior
testing and was deliberately excluded here via `include_explanation=False`
so it wouldn't dominate or obscure the retrieval number. If a caller wants
the full retrieve+explain response, `generation_ms` is reported separately
in every `/query` response — see the structured logging in `src/api.py`.

Corpora here are synthetic fixtures (12 x ~20 trivial functions each), not
real production-sized repositories; numbers will shift with corpus size,
concurrency, and hardware.
