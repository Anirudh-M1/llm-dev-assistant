# LLM Dev Assistant

A Retrieval-Augmented Generation (RAG) assistant for understanding codebases.

## Features

- Code ingestion pipeline
- TF-IDF embeddings
- FAISS vector search
- Llama3 explanation via Ollama

## Architecture

User Query  
↓  
Vector Embedding (TF-IDF)  
↓  
FAISS Retrieval  
↓  
LLM Explanation