"""FastAPI backend.

Endpoints:
  GET  /health
  POST /ingest/{repo_id}  -- chunk + embed + index a directory under repo_id
  POST /query/{repo_id}   -- retrieve + explain against a previously ingested repo

Every /query request is logged as a single structured JSON line with
retrieval_ms and generation_ms broken out separately.
"""

import json
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.chunker import chunk_directory
from src.embeddings import embed_texts_async
from src.index_store import build_faiss_index
from src.main import explain_code, retrieve_chunks
from src.repo_store import RepoStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="LLM Dev Assistant API")
store = RepoStore()


class IngestRequest(BaseModel):
    directory: str


class IngestResponse(BaseModel):
    repo_id: str
    chunk_count: int


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class RetrievedChunk(BaseModel):
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int


class QueryResponse(BaseModel):
    explanation: str
    retrieved: list[RetrievedChunk]
    retrieval_ms: float
    generation_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/{repo_id}", response_model=IngestResponse)
async def ingest(repo_id: str, req: IngestRequest):
    chunks = chunk_directory(req.directory)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No chunks found in {req.directory}")

    embeddings = await embed_texts_async([c.source for c in chunks])
    index = build_faiss_index(embeddings)
    store.put(repo_id, chunks, index)

    return IngestResponse(repo_id=repo_id, chunk_count=len(chunks))


@app.post("/query/{repo_id}", response_model=QueryResponse)
def query(repo_id: str, req: QueryRequest):
    repo = store.get(repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repo '{repo_id}' has not been ingested.")

    retrieval_start = time.perf_counter()
    retrieved = retrieve_chunks(req.query, repo["chunks"], repo["index"], k=req.k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    generation_start = time.perf_counter()
    explanation = explain_code(retrieved, req.query)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    logger.info(json.dumps({
        "event": "query",
        "repo_id": repo_id,
        "query": req.query,
        "k": req.k,
        "chunk_count": len(retrieved),
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
    }))

    return QueryResponse(
        explanation=explanation,
        retrieved=[
            RetrievedChunk(
                qualified_name=c.qualified_name,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
            )
            for c in retrieved
        ],
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
    )
