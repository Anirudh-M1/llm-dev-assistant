import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np

from src.embeddings import embed_text, embed_texts_async


def test_embed_text_returns_the_response_vector():
    fake_response = {"embeddings": [[0.1, 0.2, 0.3]]}
    with patch("src.embeddings.ollama.embed", return_value=fake_response) as mock_embed:
        vector = embed_text("some code")

    mock_embed.assert_called_once()
    assert np.allclose(vector, [0.1, 0.2, 0.3])


def test_embed_texts_async_returns_one_vector_per_input():
    async def fake_embed(model, input):
        await asyncio.sleep(0)
        return {"embeddings": [[float(len(input))] * 3]}

    fake_client = AsyncMock()
    fake_client.embed.side_effect = fake_embed

    with patch("src.embeddings.ollama.AsyncClient", return_value=fake_client):
        result = asyncio.run(embed_texts_async(["a", "bb", "ccc"], concurrency=2))

    assert result.shape == (3, 3)
    assert list(result[:, 0]) == [1.0, 2.0, 3.0]


def test_embed_texts_async_respects_the_concurrency_bound():
    in_flight = 0
    max_in_flight = 0

    async def fake_embed(model, input):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {"embeddings": [[0.0]]}

    fake_client = AsyncMock()
    fake_client.embed.side_effect = fake_embed

    with patch("src.embeddings.ollama.AsyncClient", return_value=fake_client):
        asyncio.run(embed_texts_async([str(i) for i in range(10)], concurrency=3))

    assert max_in_flight <= 3
