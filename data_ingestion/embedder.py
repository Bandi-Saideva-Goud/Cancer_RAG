from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.getenv('EMBEDDING_MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY', None)

class Embedder:
    def __init__(self):
        if MODEL == 'sentence-transformers/all-MiniLM-L6-v2':
            self.model = HuggingFaceEmbeddings(
                                model_name=MODEL,
                                # model_kwargs={"device": "mps"}
                            )
        else:
            self.model = OpenAIEmbeddings(
                model = "text-embedding-3-small"
            )

