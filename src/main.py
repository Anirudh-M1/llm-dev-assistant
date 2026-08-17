import asyncio

import ollama

from src.chunker import chunk_directory
from src.embeddings import embed_text, embed_texts_async
from src.index_store import build_faiss_index


def create_embeddings(chunks):
    texts = [c.source for c in chunks]
    return asyncio.run(embed_texts_async(texts))


def retrieve_chunks(query, chunks, index, k=3):
    query_vec = embed_text(query).reshape(1, -1)
    _distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]


def explain_code(chunks, query):
    code = "\n\n".join(
        f"# {c.qualified_name} ({c.file_path}:{c.start_line}-{c.end_line})\n{c.source}"
        for c in chunks
    )

    prompt = f"""
    You are a senior software engineer acting as a collaborative co-developer.

    A developer asked the following question about a codebase.

    Question:
    {query}

    Relevant code:
    {code}

    Your response should include:

    1. Direct Answer
    Answer the question clearly and directly.

    2. Relevant Explanation
    Explain the relevant function(s), variables, or logic involved.

    3. Big Picture Context
    Explain how this code fits into the larger system if possible.

    4. Additional Notes
    Mention anything important a developer should notice
    (e.g., recursion, edge cases, design choices).

    Be detailed but stay focused on the question.
    Avoid explaining unrelated parts of the code.
    """

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    chunks = chunk_directory("corpus")

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    while True:
        query = input("Ask about the codebase: ")
        retrieved = retrieve_chunks(query, chunks, index)
        explanation = explain_code(retrieved, query)
        print(explanation)
