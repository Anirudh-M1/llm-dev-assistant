import os
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import ollama
import ast
import numpy as np
DEBUG = False

# old embeddings by word match 
# def create_embeddings(code_files):

#     documents = list(code_files.values())
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(documents)
#     return vectorizer, X


def build_faiss_index(embeddings):

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    print(f"FAISS index contains {index.ntotal} vectors")

    return index

def retrieve_functions(query, functions, index, model, k=3):
    query_vec = model.encode([query]).astype("float32")
    
    distances, indices = index.search(query_vec, k)  # top k
    retrieved = [functions[i] for i in indices[0]]  # get function info
    
    return retrieved

# def retrieve_code(query, vectorizer, index, code_files):
#     query_vec = vectorizer.transform([query]).toarray().astype("float32")
    
#     # Get top 3 files
#     distances, indices = index.search(query_vec, k=3)
    
#     file_names = list(code_files.keys())
    
#     # Combine the code from the top 3 files
#     combined_code = ""
#     retrieved_files = []
    
#     for i in indices[0]:
#         combined_code += code_files[file_names[i]] + "\n\n"
#         retrieved_files.append(file_names[i])
    
#     return retrieved_files, combined_code


def load_code_files(directory):

    code_files = {}

    for root, dirs, files in os.walk(directory):
        for file in files:

            if file.endswith(".py"):

                path = os.path.join(root, file)

                with open(path, "r", encoding="utf-8") as f:
                    code_files[path] = f.read()

    return code_files

def load_code_functions(directory):
    functTrees = {}
    for root, dirs, files in os.walk(directory):
        for file in files:

            if file.endswith(".py"):

                path = os.path.join(root, file)

                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                    functTrees[path] = [ast.parse(code), code]

    functions = []
    for t,c in functTrees.values(): 
        for node in ast.walk(t):
            if isinstance(node, ast.FunctionDef):
                lines = c.splitlines()
                functions.append([node.name, "\n".join(lines[node.lineno - 1: node.end_lineno])])
    if DEBUG:     
        for n,c in functions:
            print(f"**Name: {n}")
            print(f"**Code:\n {c} \n")

    return functions

def create_embeddings(function, model): 
    funcCode = [c for _, c in functions]
    embeddings = model.encode(funcCode)
    #print(embeddings)
    return embeddings

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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    functions = load_code_functions("corpus")

    embeddings = create_embeddings(functions, model)

    index = build_faiss_index(embeddings)
    
    while True:
        query = input("Ask about the codebase: ")
        result = retrieve_functions(query, functions, index, model)
        functions = {}
        for n,c in result:
            functions[n] = c

        explanation = explain_code(functions, query)

        print(explanation)



    # code_files = load_code_files("corpus")


    # vectorizer, X = create_embeddings(code_files)

    # index = build_faiss_index(X)

    # query = input("Ask about the codebase: ")

    # file_names, code = retrieve_code(query, vectorizer, index, code_files)

    # print("\nRetrieved files:", file_names)

    # explanation = explain_code(code, query)

    # print("\nExplanation:\n")
    # print(explanation)