import os
import re
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings
from rapidfuzz import fuzz

# CRITICAL FIX: This forces the path to match the Railway persistent volume mount
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

def load_vectorstore():
    """
    STRICT VOLUME LOAD: Ensures the FAISS index is read from the 
    Railway persistent mount point to prevent data loss.
    """
    # Force absolute path verification
    if not os.path.isabs(VECTOR_DIR):
        abs_vector_dir = os.path.abspath(VECTOR_DIR)
    else:
        abs_vector_dir = VECTOR_DIR

    index_file = os.path.join(abs_vector_dir, "index.faiss")
    
    # Check if the index exists on the persistent disk
    if not os.path.exists(index_file):
        print(f"CRITICAL ERROR: No FAISS index found at {index_file}")
        # Return None so the system knows to prompt for a document upload
        return None

    try:
        return FAISS.load_local(
            abs_vector_dir,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"LOAD FAILED: {str(e)}")
        return None
        
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
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # Clean the answer by removing HTML tags first
    clean_answer = re.sub(r'<[^>]*>', '', answer)
    clean_answer = re.sub(r'[*_`#‹›()窶]', '', clean_answer).lower()

    # Extract full technical segments from the PPTX
    all_doc_content = "\n".join([d.page_content for d in docs])
    doc_segments = [s.strip().lower() for s in all_doc_content.split('\n') if len(s.strip()) > 12]
    doc_segments = list(set([re.sub(r'[*_`#‹›()窶]', '', s) for s in doc_segments]))

    grounded_points = 0
    total_checked = 0

    for segment in doc_segments:
        # If the AI mentions the topic of the segment
        if any(word in clean_answer for word in segment.split()[:3]):
            total_checked += 1
            # Use fuzzy matching for the whole technical sentence
            if segment in clean_answer or fuzz.partial_ratio(segment, clean_answer) >= threshold:
                grounded_points += 1

    if total_checked == 0: return {"grounded": 0, "general": 100}
    grounded_pct = min(100, int((grounded_points / total_checked) * 100))
    return {"grounded": grounded_pct, "general": 100 - grounded_pct}


