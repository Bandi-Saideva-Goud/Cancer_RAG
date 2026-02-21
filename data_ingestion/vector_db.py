import io
import os
import re
import shutil
import pdfplumber
from concurrent.futures import ProcessPoolExecutor, as_completed
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embedder import Embedder
from web_scraper import Web_Parser
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH')
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))

# --- HELPER FUNCTION OUTSIDE THE CLASS ---
# This MUST be outside the class to be "Pickleable" for multiprocessing
def download_and_extract(idx, drive_link, creds_path):
    """Worker function: Downloads and extracts text. No GPU/DB involved here."""
    try:
        # Regex for ID
        match = re.search(r'[-\w]{25,}', drive_link)
        file_id = match.group(0) if match else None
        if not file_id:
            return idx, None, "Invalid Link"

        # Build local service for this process
        creds = service_account.Credentials.from_service_account_file(creds_path)
        service = build('drive', 'v3', credentials=creds, static_discovery=False)

        # Download
        request = service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        file_buffer.seek(0)

        # Extract Text
        text = ""
        with pdfplumber.open(file_buffer) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        file_buffer.close()
        return idx, text, file_id

    except Exception as e:
        return idx, None, str(e)

class Vector_DB:
    def __init__(self):
        self.drive_links = Web_Parser().drive_links
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = Embedder().model
        self.creds_path = 'divine-bonbon-381109-4a6b2845d01f.json'

        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        self.vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embedding_model
        )

        self.vectorize()
        print("\n🎉 All books processed and stored!")

    def vectorize(self):
        print(f"🚀 Parallel Extracting with {MAX_WORKERS} workers...")
        
        # 1. Parallel Text Extraction
        extracted_data = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_and_extract, i, link, self.creds_path) 
                for i, link in enumerate(self.drive_links)
            ]

            for future in as_completed(futures):
                idx, text, result_status = future.result()
                if text:
                    print(f"✅ Extracted Text for Book_{idx+1}")
                    extracted_data.append((idx, text, result_status))
                else:
                    print(f"❌ Failed Book_{idx+1}: {result_status}")

        # 2. Sequential Embedding (The GPU Part)
        # We do this in the main process to avoid CUDA/Multiprocessing errors
        print(f"\n🧠 Embedding and Storing to Vector DB (GPU)...")
        for idx, text, file_id in extracted_data:
            chunks = self.text_splitter.split_text(text)
            metadatas = [{"source": f"Book_{idx+1}", "file_id": file_id} for _ in chunks]
            
            self.vector_db.add_texts(texts=chunks, metadatas=metadatas)
            print(f"💾 Stored {len(chunks)} chunks for Book_{idx+1}")

if __name__ == "__main__":
    pipeline = Vector_DB()