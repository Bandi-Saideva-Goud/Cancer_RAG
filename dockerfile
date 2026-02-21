FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV WEB_LINK=https://jascap.org/cancer-books-pdf/english-books/
ENV CHROMA_PATH=./cancer_chroma_db
ENV MAX_WORKERS=4

CMD ["python", "app.py"]