import os
import json
import time
import requests
import numpy as np
from dotenv import load_dotenv

# ── Load the API key from .env ─────────────────────────────────────────────
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# ── The embedding model we'll use (runs on HF servers) ────────────────────
# all-MiniLM-L6-v2 produces 384-dimensional vectors, fast and accurate
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


# ── STEP 1: Read all .txt files from the docs/ folder ─────────────────────
def load_documents(folder="docs"):
    """Read every .txt file and return a list of (filename, full_text) tuples."""
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append((filename, text))
            print(f"Loaded: {filename} ({len(text)} characters)")
    return documents


# ── STEP 2: Split text into chunks ────────────────────────────────────────
def split_into_chunks(text, chunk_size=150, overlap=30):
    """
    Split a long text into smaller overlapping chunks.

    Why chunks?
      - Embedding models have a token limit (~256-512 tokens)
      - Smaller chunks = more precise retrieval
      - A 2000-word document as one chunk is too vague

    Why overlap?
      - If a sentence is split across two chunks, overlap ensures
        the context isn't lost at the boundary
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward but keep 'overlap' words

    return chunks


# ── STEP 3: Get embeddings from Hugging Face API ───────────────────────────
def get_embedding(text, retries=3):
    """
    Send text to the HF API and get back a vector (list of numbers).

    The model might be loading (cold start) so we retry a few times
    with a wait in between.
    """
    payload = {"inputs": text}

    for attempt in range(retries):
        response = requests.post(EMBEDDING_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            embedding = response.json()
            # HF returns a nested list for sentence-transformers, we want the first item
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            return np.array(embedding)

        elif response.status_code == 503:
            # Model is loading on HF servers — wait and retry
            wait_time = 10 * (attempt + 1)
            print(f"  Model loading, waiting {wait_time}s...")
            time.sleep(wait_time)

        else:
            print(f"  Error {response.status_code}: {response.text}")
            break

    raise Exception("Failed to get embedding after retries")


# ── STEP 4: Build and save the vector store ────────────────────────────────
def build_index(docs_folder="docs", output_file="vector_store.json"):
    """
    Full indexing pipeline:
    1. Load docs → 2. Chunk them → 3. Embed each chunk → 4. Save to file
    """
    print("\n=== Starting Indexing ===\n")
    documents = load_documents(docs_folder)

    all_chunks = []      # the raw text of each chunk
    all_embeddings = []  # the vector for each chunk
    all_sources = []     # which file each chunk came from

    for filename, text in documents:
        print(f"\nProcessing: {filename}")
        chunks = split_into_chunks(text)
        print(f"  Split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            print(f"  Embedding chunk {i+1}/{len(chunks)}...", end=" ")
            embedding = get_embedding(chunk)
            print(f"vector shape: {embedding.shape}")

            all_chunks.append(chunk)
            all_embeddings.append(embedding.tolist())  # numpy → plain list for JSON
            all_sources.append(filename)

    # Save everything to a JSON file
    vector_store = {
        "chunks": all_chunks,
        "embeddings": all_embeddings,
        "sources": all_sources
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vector_store, f)

    print(f"\n=== Done! Indexed {len(all_chunks)} chunks → saved to {output_file} ===")


# ── Run it ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_index()
