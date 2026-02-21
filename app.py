import os
import gradio as gr
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv('EMBEDDING_MODEL')


class CancerRAGApp:
    def __init__(
        self,
        chroma_path="data_ingestion/cancer_chroma_db",
        model_name="llama3.2:3b",
        top_k=5,
    ):

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")

        self.chroma_path = chroma_path
        self.model_name = model_name
        self.top_k = top_k

        # Initialize LLM client (Ollama via OpenAI-compatible API)
        self.llm_client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama"  # Required but unused
        )

        # Initialize Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL,
            # model_kwargs={"device": "mps"}
        )

        # Initialize Vector DB
        self.vector_db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_model
        )

        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # System Prompt
        self.system_prompt = """
        You are an expert oncology assistant specialized in cancer-related medical research.

        Strict Guidelines:
        - Answer ONLY using retrieved context.
        - If answer not found, say: "The provided documents do not contain enough information."
        - Do NOT hallucinate.
        - Provide structured explanations.
        - Avoid prescribing treatment.
        - Add disclaimer when appropriate.

        This system is for research and educational purposes only.
        """

    # ==========================
    # RETRIEVAL
    # ==========================
    def retrieve_context(self, query):
        return self.retriever.invoke(query)

    # ==========================
    # PROMPT BUILDER
    # ==========================
    def build_messages(self, query, retrieved_docs, history):
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        user_prompt = f"""
        Context:
        {context}

        Conversation History:
        {history}

        Current Question:
        {query}

        Answer strictly based on the context above.
        """

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ==========================
    # LLM GENERATION
    # ==========================
    def generate_response(self, query, history):
        retrieved_docs = self.retrieve_context(query)
        messages = self.build_messages(query, retrieved_docs, history)

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        return response.choices[0].message.content

    # ==========================
    # GRADIO CHAT HANDLER
    # ==========================
    def chat_handler(self, message, history):
        history = history or []

        # Generate response
        answer = self.generate_response(message, history)

        # Append user message
        history.append({"role": "user", "content": message})

        # Append assistant message
        history.append({"role": "assistant", "content": answer})

        return answer

    # ==========================
    # BUILD UI
    # ==========================
    def launch(self):
        gr.ChatInterface(self.chat_handler).launch(server_name="0.0.0.0",
                                                    server_port=8000
                                                )
            



# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    app = CancerRAGApp()
    app.launch()