from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.getenv('EMBEDDING_MODEL')

class Embedder:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
                            model_name=MODEL,
                            # model_kwargs={"device": "mps"}
                        )
