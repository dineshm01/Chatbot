import os
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings
from rapidfuzz import fuzz


# This finds the exact folder where THIS file (rag_utils.py) is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# This joins it with 'vectorstore' so it matches /app/backend/vectorstore
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
def load_vectorstore():
    # Check if the index exists on the persistent disk
    index_file = os.path.join(VECTOR_DIR, "index.faiss")
    if not os.path.exists(index_file):
        print(f"DEBUG: No index found at {index_file}")
        return None

    return FAISS.load_local(
        VECTOR_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def get_retriever():
    vectorstore = load_vectorstore()
    if not vectorstore:
        return None
    return vectorstore.as_retriever(
    search_kwargs={
        "k": 20,
        "score_threshold": 0.1
    }
)

def truncate_docs(docs, max_chars=1500):
    text = ""
    for d in docs:
        if len(text) + len(d.page_content) > max_chars:
            break
        text += d.page_content + "\n"
    return text.strip()

def compute_confidence(docs):
    if not docs:
        return "🔵 Confidence: General knowledge"
    total = sum(len(d.page_content) for d in docs)
    if total >= 800:
        return "🟢 Confidence: Fully covered by notes"
    if total >= 200:
        return "🟡 Confidence: Partially inferred"
    return "🔵 Confidence: General knowledge"


def compute_coverage(docs, answer=None, threshold=70):
    """
    Measures how much of the answer is supported by retrieved docs.
    """
    if not docs or not answer:
        return {
            "grounded": 0,
            "general": 100
        }

    doc_text = " ".join(d.page_content for d in docs).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    if not sentences:
        return {"grounded": 0, "general": 100}

    grounded = 0
    for s in sentences:
        score = fuzz.partial_ratio(s.lower(), doc_text)
        if score >= threshold:
            grounded += 1

    grounded_pct = int((grounded / len(sentences)) * 100)
    general_pct = 100 - grounded_pct

    return {
        "grounded": grounded_pct,
        "general": general_pct
    }







