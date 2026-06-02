import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

# ── Load API key ───────────────────────────────────────────────────────────
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
MISTRAL_HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}

# ── Models ─────────────────────────────────────────────────────────────────
# Embeddings: HF API (free, works great)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"

# LLM: Mistral's own API (free tier, reliable)
LLM_URL = "https://api.mistral.ai/v1/chat/completions"


# ── STEP 1: Load the vector store we built with indexer.py ────────────────
def load_vector_store(filepath="vector_store.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])  # shape: (num_chunks, 384)
    sources = data["sources"]

    print(f"Loaded vector store: {len(chunks)} chunks ready for search\n")
    return chunks, embeddings, sources


# ── STEP 2: Convert the user's question into a vector ─────────────────────
def get_embedding(text):
    payload = {"inputs": text}
    response = requests.post(EMBEDDING_URL, headers=HF_HEADERS, json=payload)

    if response.status_code == 200:
        embedding = response.json()
        if isinstance(embedding[0], list):
            embedding = embedding[0]
        return np.array(embedding)
    else:
        raise Exception(f"Embedding error {response.status_code}: {response.text}")


# ── STEP 3: Find the most relevant chunks using cosine similarity ──────────
def find_relevant_chunks(question_embedding, embeddings, chunks, sources, top_k=3):
    """
    Cosine similarity measures the angle between two vectors.
    - Score of 1.0 = identical meaning
    - Score of 0.0 = completely unrelated

    We compute the similarity between the question vector and every
    chunk vector, then return the top_k most similar chunks.
    """
    # Normalize vectors to unit length (required for cosine similarity)
    question_norm = question_embedding / np.linalg.norm(question_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Dot product of normalized vectors = cosine similarity
    similarities = np.dot(embeddings_norm, question_norm)

    # Get indices of top_k highest scores
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "source": sources[idx],
            "score": float(similarities[idx])
        })

    return results


# ── STEP 4: Ask Mistral to answer using the retrieved chunks ───────────────
def ask_mistral(question, relevant_chunks):
    # Build context string from retrieved chunks
    context = ""
    for i, item in enumerate(relevant_chunks):
        context += f"--- Excerpt {i+1} from {item['source']} ---\n{item['chunk']}\n\n"

    # System prompt — this is how we reduce hallucination
    system_prompt = """You are an academic study assistant.
Answer questions based on the excerpts provided below.
The answer WILL be in the excerpts — read them carefully.
Be concise and clear. Do not use outside knowledge."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Excerpts from study notes:\n\n{context}\nQuestion: {question}"}
    ]

    payload = {
        "model": "mistral-small-latest",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512
    }

    response = requests.post(LLM_URL, headers=MISTRAL_HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LLM error {response.status_code}: {response.text}")


# ── STEP 5: Main chat loop ─────────────────────────────────────────────────
def chat():
    chunks, embeddings, sources = load_vector_store()

    print("=== Academic Chatbot ===")
    print("Ask questions about your notes. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching notes...")

        # Convert question to vector
        question_embedding = get_embedding(question)

        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(question_embedding, embeddings, chunks, sources, top_k=5)

        # Show which sources were found (transparency)
        print("Relevant sources found:")
        for item in relevant_chunks:
            print(f"  - {item['source']} (similarity: {item['score']:.3f})")

        print("\nAsking Mistral...\n")

        # Get answer from Mistral
        answer = ask_mistral(question, relevant_chunks)

        print(f"Assistant: {answer}\n")
        print("-" * 60 + "\n")


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chat()
