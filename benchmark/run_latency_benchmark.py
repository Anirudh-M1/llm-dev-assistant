"""Multi-repo concurrent retrieval latency benchmark.

Starts a real uvicorn server, ingests N separate synthetic repos, then fires
concurrent /query requests across all of them simultaneously (asyncio +
httpx.AsyncClient -- genuine concurrent HTTP, not a thread pool) with
include_explanation=False so the measurement isolates retrieval latency
(embed query + FAISS search) from LLM generation time, which is reported
separately elsewhere and is two orders of magnitude slower locally.

Run:
    python -m benchmark.generate_multi_repo_corpus
    python -m benchmark.run_latency_benchmark
"""

import asyncio
import os
import subprocess
import sys
import time

import httpx

HOST = "127.0.0.1"
PORT = 8199
BASE_URL = f"http://{HOST}:{PORT}"
CORPUS_ROOT = os.path.join(os.path.dirname(__file__), "multi_repo_corpus")
ROUNDS = 5
QUERIES = ["How does op_0_0 work?", "What does op_1_2 return?", "Explain op_2_3"]


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for _ in range(50):
        try:
            httpx.get(f"{BASE_URL}/health", timeout=1)
            return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not become healthy in time")


async def ingest_all_repos(client, repo_ids):
    for repo_id in repo_ids:
        directory = os.path.join(CORPUS_ROOT, repo_id)
        resp = await client.post(f"{BASE_URL}/ingest/{repo_id}", json={"directory": directory})
        resp.raise_for_status()


async def query_repo(client, repo_id):
    query = QUERIES[hash(repo_id) % len(QUERIES)]
    start = time.perf_counter()
    resp = await client.post(
        f"{BASE_URL}/query/{repo_id}",
        json={"query": query, "k": 2, "include_explanation": False},
    )
    wall_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return resp.json()["retrieval_ms"], wall_ms


async def run_benchmark():
    repo_ids = sorted(os.listdir(CORPUS_ROOT))
    async with httpx.AsyncClient(timeout=60) as client:
        print(f"Ingesting {len(repo_ids)} repos...")
        await ingest_all_repos(client, repo_ids)

        retrieval_samples = []
        wall_samples = []
        for round_i in range(ROUNDS):
            results = await asyncio.gather(*(query_repo(client, r) for r in repo_ids))
            retrieval_samples.extend(r for r, _ in results)
            wall_samples.extend(w for _, w in results)
            print(f"round {round_i + 1}/{ROUNDS}: fired {len(repo_ids)} concurrent queries")

    return repo_ids, retrieval_samples, wall_samples


def percentile(data, pct):
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def main():
    proc = start_server()
    try:
        repo_ids, retrieval_samples, wall_samples = asyncio.run(run_benchmark())
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    print()
    print(f"Repos: {len(repo_ids)}  |  Rounds: {ROUNDS}  |  Concurrent queries/round: {len(repo_ids)}")
    print(f"Total samples: {len(retrieval_samples)}")
    print()
    print("Server-side retrieval latency (embed query + FAISS search), include_explanation=False:")
    print(f"  p50: {percentile(retrieval_samples, 50):.2f}ms")
    print(f"  p95: {percentile(retrieval_samples, 95):.2f}ms")
    print(f"  max: {max(retrieval_samples):.2f}ms")
    print()
    print("Client-observed wall-clock (retrieval + HTTP round trip, no generation):")
    print(f"  p50: {percentile(wall_samples, 50):.2f}ms")
    print(f"  p95: {percentile(wall_samples, 95):.2f}ms")


if __name__ == "__main__":
    main()
