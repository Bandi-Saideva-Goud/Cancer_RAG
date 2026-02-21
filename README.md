

# 🧬 Cancer RAG — Oncology Research Assistant

A **Retrieval-Augmented Generation (RAG)** system designed to provide structured, context-grounded responses to cancer-related research queries using curated medical literature.

This project combines:

* 📚 Vector Database (Chroma)
* 🔎 Semantic Retrieval (HuggingFace Embeddings)
* 🤖 Local LLM via Ollama (Llama 3.2)
* 🌐 Gradio UI
* 🐳 Fully Containerized (Docker + Docker Compose)

---

## 🚀 Project Overview

Cancer RAG is an AI-powered oncology research assistant that:

* Retrieves relevant information from curated cancer literature
* Generates responses strictly grounded in retrieved documents
* Avoids hallucination via controlled prompt design
* Provides structured, research-focused explanations
* Runs fully locally using Ollama (no OpenAI dependency required)

---

## 🏗️ Architecture

```
User (Browser)
        ↓
Gradio Chat UI
        ↓
Retriever (Chroma Vector DB)
        ↓
Top-k Relevant Context
        ↓
Ollama LLM (Llama 3.2)
        ↓
Structured Response
```

---

## 🛠️ Tech Stack

| Component        | Technology                             |
| ---------------- | -------------------------------------- |
| LLM              | Ollama (Llama 3.2:3b)                  |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB        | Chroma                                 |
| Framework        | LangChain                              |
| UI               | Gradio                                 |
| Containerization | Docker + Docker Compose                |

---

## 📂 Project Structure

```
Cancer_RAG/
│
├── app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── data_ingestion/
│   ├── embedder.py
│   ├── vector_db.py
│   ├── web_scraper.py
│
└── cancer_chroma_db/   (generated at runtime)
```

---

## ⚙️ Setup & Installation

### 🔹 1️⃣ Clone the Repository

```bash
git clone https://github.com/Bandi-Saideva-Goud/Cancer_RAG.git
cd Cancer_RAG
```

---

### 🔹 2️⃣ Run with Docker Compose (Recommended)

```bash
docker compose up --build
```

After startup:

Pull the LLM model:

```bash
docker exec -it ollama ollama pull llama3.2:3b
```

Open in browser:

```
http://localhost:8000
```

---

## 🧠 How It Works

1. User submits a cancer-related query.
2. Query is embedded using HuggingFace embeddings.
3. Chroma retrieves top-k semantically similar chunks.
4. Retrieved context is injected into a structured system prompt.
5. Ollama (Llama 3.2) generates a grounded response.
6. Gradio displays the answer.

---

## 🎯 Prompt Guardrails

The system enforces:

* Strict context-based answering
* No hallucinations
* No medical prescriptions
* Educational & research use only
* Structured explanation format

---

## 🔒 Security & Best Practices

* No API keys committed
* Secrets managed via environment variables
* Vector DB excluded from Git
* Docker-based reproducibility

---

## 📌 Example Use Cases

* Cancer treatment explanation research
* Chemotherapy overview
* Oncology educational assistance
* Medical literature contextual querying
* Structured academic Q&A

---

## ⚠️ Disclaimer

This system is for research and educational purposes only.
It does not provide medical advice, diagnosis, or treatment recommendations.

---

## 🧪 Future Improvements

* Streaming token responses
* Citation highlighting
* RAG evaluation framework
* GPU-enabled Ollama deployment
* Advanced hallucination detection
* Multi-document citation tracking

---

## 👨‍💻 Author

**Bandi Saideva Goud**
Data Scientist | AI Engineer
Focused on LLM Systems, RAG Architectures, and Applied AI

---

# 🌟 Why This Project Matters

This project demonstrates:

* End-to-end RAG system design
* Local LLM integration via OpenAI-compatible API
* Production-grade containerization
* Secure ML engineering practices
* Practical healthcare AI application



Just tell me the target audience (recruiter / research / startup / enterprise).
