from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api import app, store

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_and_query_roundtrip(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "sample.py").write_text("def greet():\n    return 'hi'\n")

    async def fake_embed_texts_async(texts, concurrency=8):
        return np.random.rand(len(texts), 8).astype("float32")

    with patch("src.api.embed_texts_async", side_effect=fake_embed_texts_async):
        resp = client.post("/ingest/test-repo", json={"directory": str(corpus_dir)})

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"repo_id": "test-repo", "chunk_count": 1}

    ingested_chunks = store.get("test-repo")["chunks"]
    with patch("src.api.retrieve_chunks", return_value=ingested_chunks) as mock_retrieve, \
            patch("src.api.explain_code", return_value="explanation text") as mock_explain:
        resp = client.post("/query/test-repo", json={"query": "what does greet do?", "k": 1})

    mock_retrieve.assert_called_once()
    mock_explain.assert_called_once()
    assert resp.status_code == 200
    body = resp.json()
    assert body["explanation"] == "explanation text"
    assert body["retrieved"][0]["qualified_name"] == "greet"
    assert body["retrieval_ms"] >= 0
    assert body["generation_ms"] >= 0


def test_query_unknown_repo_returns_404():
    resp = client.post("/query/does-not-exist", json={"query": "hi"})
    assert resp.status_code == 404


def test_ingest_empty_directory_returns_400(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    resp = client.post("/ingest/empty-repo", json={"directory": str(empty_dir)})
    assert resp.status_code == 400
