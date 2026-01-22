import os
import re
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings
from rapidfuzz import fuzz

# CRITICAL FIX: This forces the path to match the Railway persistent volume mount
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    """
    Calculates groundedness based on technical fragments to match frontend highlights.
   
    """
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # 1. Standardize the document Source of Truth
    doc_text = " ".join([" ".join(d.page_content.split()) for d in docs]).lower()
    doc_text = doc_text.replace("*", "").replace("#", "").replace("‹#›", "")

    # 2. Extract technical fragments from the AI's answer
    # We split by punctuation and common filler words just like the frontend
    sentences = re.split(r'[.!?\n\-:,;]|\b(?:is|are|was|were|the|an|a|to|for|with|from)\b', answer, flags=re.IGNORECASE)
    
    # Filter for meaningful technical blocks (2+ words and 8+ chars)
    fragments = [s.strip() for s in sentences if len(s.strip()) > 8 and len(s.strip().split()) >= 2]

    if not fragments:
        return {"grounded": 0, "general": 100}

    grounded_count = 0
    for frag in fragments:
        clean_frag = " ".join(frag.lower().split())
        
        # 3. Apply the same matching logic used for highlights
        if clean_frag in doc_text or fuzz.partial_ratio(clean_frag, doc_text) >= threshold:
            grounded_count += 1

    # Calculate final percentage
    grounded_pct = int((grounded_count / len(fragments)) * 100)
    
    # Safety clamp
    grounded_pct = min(100, max(0, grounded_pct))
    
    return {"grounded": grounded_pct, "general": 100 - grounded_pct}

