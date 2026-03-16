import os
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ollama

def create_embeddings(code_files):

    documents = list(code_files.values())
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    return vectorizer, X


def build_faiss_index(X):

    dense_vectors = X.toarray().astype("float32")

    dimension = dense_vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(dense_vectors)

    return index

def retrieve_code(query, vectorizer, index, code_files):
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    
    # Get top 3 files
    distances, indices = index.search(query_vec, k=3)
    
    file_names = list(code_files.keys())
    
    # Combine the code from the top 3 files
    combined_code = ""
    retrieved_files = []
    
    for i in indices[0]:
        combined_code += code_files[file_names[i]] + "\n\n"
        retrieved_files.append(file_names[i])
    
    return retrieved_files, combined_code


def load_code_files(directory):

    code_files = {}

    for root, dirs, files in os.walk(directory):
        for file in files:

            if file.endswith(".py"):

                path = os.path.join(root, file)

                with open(path, "r", encoding="utf-8") as f:
                    code_files[path] = f.read()

    return code_files


def explain_code(code, query):

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

    code_files = load_code_files("corpus")

    vectorizer, X = create_embeddings(code_files)

    index = build_faiss_index(X)

    query = input("Ask about the codebase: ")

    file_names, code = retrieve_code(query, vectorizer, index, code_files)

    print("\nRetrieved files:", file_names)

    explanation = explain_code(code, query)

    print("\nExplanation:\n")
    print(explanation)