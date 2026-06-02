import os
import json
import requests
import numpy as np
import streamlit as st

# ── API Keys ───────────────────────────────────────────────────────────────
# On Streamlit Cloud, secrets come from st.secrets
# Locally, we fall back to environment variables / .env
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
MISTRAL_HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"
LLM_URL = "https://api.mistral.ai/v1/chat/completions"


# ── Load vector store once and cache it ───────────────────────────────────
@st.cache_resource
def load_vector_store(filepath="vector_store.json"):
    """
    @st.cache_resource means this function only runs ONCE
    even if the user sends multiple messages.
    Streamlit reruns the whole script on every interaction,
    so caching prevents reloading the vector store each time.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])
    sources = data["sources"]
    return chunks, embeddings, sources


# ── Embedding ──────────────────────────────────────────────────────────────
def get_embedding(text):
    response = requests.post(EMBEDDING_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        embedding = response.json()
        if isinstance(embedding[0], list):
            embedding = embedding[0]
        return np.array(embedding)
    raise Exception(f"Embedding error {response.status_code}: {response.text}")


# ── Retrieval ──────────────────────────────────────────────────────────────
def find_relevant_chunks(question_embedding, embeddings, chunks, sources, top_k=5):
    q_norm = question_embedding / np.linalg.norm(question_embedding)
    e_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(e_norm, q_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [{"chunk": chunks[i], "source": sources[i], "score": float(similarities[i])} for i in top_indices]


# ── LLM call ──────────────────────────────────────────────────────────────
def ask_mistral(question, relevant_chunks):
    context = ""
    for i, item in enumerate(relevant_chunks):
        context += f"--- Excerpt {i+1} from {item['source']} ---\n{item['chunk']}\n\n"

    system_prompt = """You are an academic study assistant.
Answer questions based on the excerpts provided below.
The answer WILL be in the excerpts — read them carefully.
Be concise and clear. Do not use outside knowledge."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Excerpts from study notes:\n\n{context}\nQuestion: {question}"}
    ]

    response = requests.post(LLM_URL, headers=MISTRAL_HEADERS, json={
        "model": "mistral-small-latest",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512
    })

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    raise Exception(f"LLM error {response.status_code}: {response.text}")


# ── Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Academic Chatbot", page_icon="📚")

st.title("📚 Academic Chatbot")
st.caption("Ask questions based on your study notes.")

# Load vector store
chunks, embeddings, sources = load_vector_store()

# Chat history — st.session_state persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom
if question := st.chat_input("Ask a question about your notes..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching notes and thinking..."):
            try:
                q_emb = get_embedding(question)
                relevant_chunks = find_relevant_chunks(q_emb, embeddings, chunks, sources)
                answer = ask_mistral(question, relevant_chunks)

                # Show answer
                st.markdown(answer)

                # Show sources in an expander (collapsible)
                with st.expander("📎 Sources used"):
                    for item in relevant_chunks[:3]:
                        st.markdown(f"**{item['source']}** (similarity: `{item['score']:.3f}`)")
                        st.caption(item['chunk'][:200] + "...")
                        st.divider()

                # Save full answer to history
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
