# Evaluation results

- Commit: `a921c96c09cabacdd9ac4bdaa432487729d27490`
- Date: 2026-08-16
- Embedding model: `nomic-embed-text` (Ollama)
- Generation model: `llama3` (Ollama)
- Corpus size: 19 chunks (`corpus/`)
- Question set: 19 questions (`eval/questions.jsonl`)
- Machine: single local machine, sequential (one query at a time, not concurrent)

Reproduce: `python -m eval.run`

## Retrieval accuracy

Recall@k: fraction of questions where the ground-truth chunk appears in the top-k retrieved results. MRR@k: mean reciprocal rank of the ground-truth chunk, 0 if it doesn't appear in the top-k.

| k | Recall@k | MRR@k |
|---|---|---|
| 1 | 0.89 | 0.89 |
| 3 | 1.00 | 0.95 |
| 5 | 1.00 | 0.95 |

## Latency

Retrieval (embed query + FAISS search) and generation (full Llama3 completion via Ollama) are reported separately -- they are different orders of magnitude and conflating them would hide what's actually slow.

| Stage | p50 | p95 |
|---|---|---|
| Retrieval | 32.6ms | 119.9ms |
| Generation | 17374.7ms | 20659.1ms |
