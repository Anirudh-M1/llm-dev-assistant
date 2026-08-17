import ollama
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_directory
from src.index_store import build_faiss_index


def create_embeddings(chunks, model):
    texts = [c.source for c in chunks]
    return model.encode(texts)


def retrieve_chunks(query, chunks, index, model, k=3):
    query_vec = model.encode([query]).astype("float32")
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
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = chunk_directory("corpus")

    embeddings = create_embeddings(chunks, model)

    index = build_faiss_index(embeddings)

    while True:
        query = input("Ask about the codebase: ")
        retrieved = retrieve_chunks(query, chunks, index, model)
        explanation = explain_code(retrieved, query)
        print(explanation)
