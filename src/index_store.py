"""FAISS index construction."""

import faiss


def build_faiss_index(embeddings):
    embeddings = embeddings.astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
