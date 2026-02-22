import os
import streamlit as st
import numpy as np
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import sys

load_dotenv()

MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ==========================
# Utility: Cosine Similarity
# ==========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CancerRAGApp:
    def __init__(
        self,
        chroma_path="data_ingestion/cancer_chroma_db",
        model_name="gpt-5-nano-2025-08-07",
        top_k=5,
    ):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.chroma_path = chroma_path
        self.model_name = model_name
        self.top_k = top_k

        if MODEL == 'sentence-transformers/all-MiniLM-L6-v2':
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=MODEL
            )
        else:
            self.embedding_model = OpenAIEmbeddings(
                model = "text-embedding-3-small"
            )


        self.vector_db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_model,
            # collection_name="cancer_collection"
        )

        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        self.system_prompt = """
        You are a helpful and polite customer service associate specialized in cancer-related medical research.

        Behavior Guidelines:

        1. General Conversation:
        - You may respond naturally to greetings, small talk, and general polite conversation 
        (e.g., "Hello", "How are you?", "Thank you").
        - Keep such responses short, friendly, and professional.

        2. Cancer-Related Questions:
        - Answer ONLY using the retrieved context if it is relevant to the user's question.
        - If the retrieved context is insufficient, say:
        "The provided information is not sufficient to answer your question."
        - Do NOT hallucinate or add information not present in the context.
        - Provide clear and structured explanations.
        - Avoid prescribing treatments or giving personalized medical advice.
        - Add a disclaimer when discussing medical information.

        3. Out-of-Scope Questions:
        - If the question is unrelated to cancer or general conversation, say:
        "I can only assist with cancer-related questions."

        This system is intended for research and educational purposes only.
        """

    # ==========================
    # Retrieval
    # ==========================
    def retrieve_context(self, query):
        return self.retriever.invoke(query)

    # ==========================
    # Reranking
    # ==========================
    def rerank_with_large_embedding(self, query, docs):

        if not docs:
            return []

        # Embed query
        query_embedding = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        ).data[0].embedding

        chunk_texts = [doc.page_content[:8000] for doc in docs]

        chunk_embeddings = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk_texts
        ).data

        scores = []

        for doc, emb_obj in zip(docs, chunk_embeddings):
            chunk_embedding = np.array(emb_obj.embedding)
            score = cosine_similarity(
                np.array(query_embedding),
                chunk_embedding
            )
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scores]
    # ==========================
    # Build Prompt
    # ==========================
    def build_messages(self, query, context, chat_history):

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add previous conversation history
        for msg in chat_history:
            messages.append(msg)

        # Add current user query with RAG context
        user_prompt = f"""
    Use the provided context and Chat History to answer the question.

    Current Question:
    {query}

    Chat History:
    {chat_history}

    Context:
    {context}

    Answer:
    """

        messages.append({"role": "user", "content": user_prompt})

        return messages

    # ==========================
    # LLM Response
    # ==========================
    def generate_response(self, query, context, chat_history):
        messages = self.build_messages(query, context, chat_history)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        return response.choices[0].message.content

    # ==========================
    # 🚀 Streamlit UI as Method
    # ==========================
    def run_streamlit_app(self):
        st.set_page_config(page_title="Cancer RAG Assistant", layout="centered")

        st.title("🧬 Cancer Research RAG Assistant")

        # Session memory
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a cancer-related question...")

        if user_input:
            chat_history = st.session_state.messages.copy()

            with st.chat_message("user"):
                st.markdown(user_input)

            # ------------------------
            # Retrieval + Rerank
            # ------------------------
            # Step 1: Retrieve with small embedding (Chroma already uses small HF model)
            retrieved_docs = self.retrieve_context(user_input)

            # Step 2: Rerank with large embedding only
            reranked_docs = self.rerank_with_large_embedding(
                user_input,
                retrieved_docs
            )

            # Step 3: Build final context
            final_context = "\n\n".join(
                [doc.page_content for doc in reranked_docs]
            )

            # ------------------------
            # LLM Response
            # ------------------------
            answer = self.generate_response(
                user_input,
                final_context,
                chat_history
            )

            # Now update session memory
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

            # ------------------------
            # Show Final Retrieved Chunks
            # ------------------------
            st.divider()
            st.subheader("🔎 Retrieved Top 5 Chunks (After Reranking)")

            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(reranked_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500])
                    st.markdown("---")

# ==========================
# Main Entry
# ==========================
if __name__ == "__main__":
    app = CancerRAGApp()
    app.run_streamlit_app()