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
    """
    Verified Confidence Logic for Strict LangChain RAG.
    Ensures the UI accurately reflects retrieved technical data.
    """
    if not docs:
        return "🔴 Confidence: No document context found"
    
    # Measure the depth of information retrieved from the slides
    total_chars = sum(len(d.page_content) for d in docs)
    
    # 1. High Confidence: Technical specifics (like G&D play opposite games) are present
    if total_chars >= 800:
        return "🟢 Confidence: Fully covered by notes"
    
    # 2. Medium Confidence: Some technical fragments found
    if total_chars >= 200:
        return "🟡 Confidence: Partially covered by notes"
        
    # 3. Low Confidence: Stray keywords found but insufficient for a full technical answer
    return "🔵 Confidence: Limited context available"

def compute_coverage(docs, answer=None, threshold=80):
    """
    Calculates groundedness using Shadow Normalization to match frontend highlights.
    """
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # 1. SHADOW NORMALIZATION for the Answer (Removes HTML tags and symbols)
    # This prevents markdown bolding (**word**) from breaking the match
    shadow_answer = re.sub(r'<[^>]*>', '', answer) # Remove HTML tags
    shadow_answer = shadow_answer.lower()
    shadow_answer = re.sub(r'[*_`#‹›()窶]', '', shadow_answer)
    shadow_answer = " ".join(shadow_answer.split())

    # 2. SHADOW NORMALIZATION for the Documents
    doc_text = " ".join([" ".join(d.page_content.split()) for d in docs]).lower()
    doc_text = re.sub(r'[*_`#‹›()窶]', '', doc_text)

    # 3. Extract technical fragments from the cleaned shadow answer
    # We split by punctuation and filler words
    sentences = re.split(r'[.!?\n\-:,;]|\b(?:is|are|was|were|the|an|a|to|for|with|from)\b', shadow_answer, flags=re.IGNORECASE)
    fragments = [s.strip() for s in sentences if len(s.strip()) > 8 and len(s.strip().split()) >= 2]

    if not fragments:
        return {"grounded": 0, "general": 100}

    grounded_count = 0
    for frag in fragments:
        clean_frag = " ".join(frag.split())
        # FIX: Check if the fragment exists in the normalized source text
        if clean_frag in doc_text or fuzz.partial_ratio(clean_frag, doc_text) >= threshold:
            grounded_count += 1

    grounded_pct = int((grounded_count / len(fragments)) * 100)
    return {"grounded": grounded_pct, "general": 100 - grounded_pct}



