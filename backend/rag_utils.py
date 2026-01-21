import os
import re
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
        "k": 20
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


def compute_coverage(docs, answer=None, threshold=80):
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # Clean document text to create a standardized search space
    doc_text = " ".join([" ".join(d.page_content.split()) for d in docs]).lower()
    doc_text = doc_text.replace("*", "").replace("#", "")
    
    # Split LLM answer into sentences longer than 30 chars
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 30]

    if not sentences:
        return {"grounded": 0, "general": 100}

    grounded_count = 0
    for s in sentences:
        # Standardize the sentence for comparison
        clean_s = " ".join(s.lower().split()).replace("*", "").replace("#", "")
        
        # Check for direct inclusion or high fuzzy similarity
        if clean_s in doc_text or fuzz.partial_ratio(clean_s, doc_text) >= threshold:
            grounded_count += 1

    grounded_pct = int((grounded_count / len(sentences)) * 100)
    return {"grounded": grounded_pct, "general": 100 - grounded_pct}
