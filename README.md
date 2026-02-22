# 🧬 Cancer RAG — Conversational Oncology Research Assistant

A **Retrieval-Augmented Generation (RAG)** system designed to provide structured, context-grounded responses to cancer-related research queries using curated medical literature.

This version implements:

* 🌐 Streamlit Chat Interface
* 🔎 Chroma Vector Database
* 🧠 Dual-Stage Retrieval (Small Embedding + Large Rerank)
* 🤖 OpenAI GPT Model for Response Generation
* 💬 Conversational Memory Support
* 🔐 Environment-based API configuration

---

## 🚀 Project Overview

Cancer RAG is a conversational oncology research assistant that:

* Retrieves relevant cancer literature using semantic search
* Reranks results using a higher-quality embedding model
* Generates responses strictly grounded in retrieved documents
* Maintains conversational memory across turns
* Handles greetings and general small talk professionally
* Avoids hallucinations via strict prompt guardrails

This system is designed for **research and educational use only**.

---

## 🏗️ Architecture

```
User (Browser)
        ↓
Streamlit Chat UI
        ↓
Chroma Vector Retrieval (Small Embedding)
        ↓
Top-5 Candidate Chunks
        ↓
Large Embedding Reranking (Similarity Scoring)
        ↓
Final Ordered Context
        ↓
OpenAI GPT Model
        ↓
Structured, Context-Grounded Response
```

---

## 🛠️ Tech Stack

| Component       | Technology                                      |
| --------------- | ----------------------------------------------- |
| LLM             | OpenAI GPT (e.g., gpt-5-nano-2025-08-07)        |
| Retrieval Embed | sentence-transformers OR text-embedding-3-small |
| Rerank Embed    | text-embedding-3-large                          |
| Vector DB       | Chroma                                          |
| Framework       | LangChain                                       |
| UI              | Streamlit                                       |
| Similarity      | Cosine Similarity (NumPy)                       |
| Config          | dotenv (.env)                                   |

---

## 🧠 Retrieval Strategy

This project uses a **two-stage retrieval pipeline**:

### 1️⃣ Fast Candidate Retrieval

* Query embedded using:

  * `sentence-transformers/all-MiniLM-L6-v2`
    **OR**
  * `text-embedding-3-small`
* Chroma retrieves top-5 semantically similar chunks.

### 2️⃣ High-Quality Reranking

* Query embedded using `text-embedding-3-large`
* Each retrieved chunk embedded using `text-embedding-3-large`
* Cosine similarity computed
* Chunks reordered by semantic similarity

This approach improves precision while keeping retrieval efficient.

---

## 💬 Conversational Memory

The assistant maintains chat history using Streamlit session state:

* Previous user and assistant messages are appended to the prompt.
* Context window is dynamically constructed.
* Enables multi-turn follow-up questions.

Example:

> User: What is chemotherapy?
> User: What are its side effects?

The second question uses prior conversation context.

---

## 🎯 Prompt Guardrails

The assistant follows strict behavioral rules:

### ✅ General Conversation

* Responds naturally to greetings and small talk.
* Maintains professional customer-service tone.

### ✅ Cancer-Related Questions

* Answers ONLY using retrieved context.
* No hallucinations.
* Structured explanations.
* No personalized medical advice.
* Includes disclaimers where appropriate.

### 🚫 Out-of-Scope Questions

If unrelated to cancer:

> "I can only assist with cancer-related questions."

---

## 📂 Project Structure

```
Cancer_RAG/
│
├── app.py
├── requirements.txt
│
├── data_ingestion/
│   ├── cancer_chroma_db/   (Persisted Vector Store)
│
└── .env
```

---

## ⚙️ Setup & Installation

### 🔹 1️⃣ Clone Repository

```bash
git clone https://github.com/Bandi-Saideva-Goud/Cancer_RAG.git
cd Cancer_RAG
```

---

### 🔹 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# OR
.venv\Scripts\activate      # Windows
```

---

### 🔹 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 4️⃣ Configure Environment Variables

Create `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small
WEB_LINK='https://jascap.org/cancer-books-pdf/english-books/'
CHROMA_PATH='./cancer_chroma_db'
MAX_WORKERS=4
```

You may also set:

```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

### 🔹 5️⃣ Run Application

```bash
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 Example Use Cases

* Cancer treatment explanation research
* Rare cancer investigation queries
* Oncology literature contextual Q&A
* Multi-turn research discussions

---

## ⚠️ Disclaimer

This system is intended for research and educational purposes only.

It does not provide medical advice, diagnosis, or treatment recommendations.

Always consult qualified healthcare professionals for medical decisions.

---

## 🧠 Future Improvements

* Streaming token responses
* Retrieval score visualization
* RAG evaluation metrics (Recall@k, MRR)
* Cross-encoder reranking
* Context compression
* Token window management
* Hallucination detection layer

---

## 👨‍💻 Author

**Bandi Saideva Goud**
Data Scientist | AI Engineer
Focused on LLM Systems, RAG Architectures, and Applied AI

---

# 🌟 Why This Project Matters

This project demonstrates:

* End-to-end conversational RAG system
* Multi-stage retrieval optimization
* Embedding-based reranking
* Memory-aware prompting
* Secure API-based LLM integration
* Practical healthcare AI application
* Production-oriented modular design

---
