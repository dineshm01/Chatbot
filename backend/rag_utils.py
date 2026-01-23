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

def compute_coverage(docs, answer=None, threshold=85):
    """
    SYNCED LOGIC: Uses sliding window matching to align perfectly 
    with the frontend 'Deep Search' highlighter.
    """
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # 1. Clean the Answer (Remove HTML tags like <strong> and artifacts)
    clean_answer = re.sub(r'<[^>]*>', '', answer)
    clean_answer = re.sub(r'[*_`#‹›()窶]', '', clean_answer).lower()
    clean_answer = " ".join(clean_answer.split())

    # 2. Get technical segments from slides (Longer segments only)
    # We split by newlines to get full technical facts from the PPTX
    all_doc_content = "\n".join([d.page_content for d in docs])
    doc_segments = [s.strip().lower() for s in all_doc_content.split('\n') if len(s.strip()) > 10]
    
    # Remove duplicates and artifacts from segments
    doc_segments = list(set([re.sub(r'[*_`#‹›()窶]', '', s) for s in doc_segments]))

    if not doc_segments:
        return {"grounded": 0, "general": 100}

    # 3. MATCHING LOGIC: Does the technical fact from the slide exist in the AI's answer?
    grounded_points = 0
    total_segments_checked = 0

    for segment in doc_segments:
        # We only check segments that the AI actually tried to talk about
        # This prevents the score from being lowered by irrelevant slides
        keywords = segment.split()[:3] # Check first few words
        if any(kw in clean_answer for kw in keywords):
            total_segments_checked += 1
            # Use fuzzy matching for the full technical fact
            if segment in clean_answer or fuzz.partial_ratio(segment, clean_answer) >= threshold:
                grounded_points += 1

    if total_segments_checked == 0:
        return {"grounded": 0, "general": 100}

    grounded_pct = int((grounded_points / total_segments_checked) * 100)
    
    # Cap the groundedness to ensure it doesn't exceed 100
    grounded_pct = min(100, grounded_pct)
    
    return {"grounded": grounded_pct, "general": 100 - grounded_pct}
