# 🚀 LLM-Powered Code Assistant

**Interactive tool to query Python codebases and get LLM-powered explanations.**  
Built to demonstrate retrieval + reasoning pipelines for developer workflows.

---

## 🖼️ Demo

![Demo GIF](assets/demo.gif)  <!-- Replace with actual GIF or screenshot -->
*Type a query about your codebase → retrieves relevant snippets → LLM explains.*

---

## 🧩 How It Works

```mermaid
flowchart LR
    A[Python Corpus] --> B[Embeddings + FAISS Index]
    B --> C[Top-k Retrieval]
    C --> D[LLM Explanation]
