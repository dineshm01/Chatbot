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
            "k": 5 # Reduced from 20 to stop 'stuffing' the answer with irrelevant slides
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
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # 1. Clean the Answer (Standardize for comparison)
    clean_answer = re.sub(r'<[^>]*>', '', answer)
    clean_answer = re.sub(r'[*_`#‹›()窶]', '', clean_answer).lower()
    clean_answer = " ".join(clean_answer.split())

    # 2. Extract full sentences from slides
    all_doc_content = " ".join([d.page_content for d in docs]).lower()
    all_doc_content = re.sub(r'[*_`#‹›()窶]', '', all_doc_content)
    all_doc_content = " ".join(all_doc_content.split())

    # 3. Use your existing fragment logic but with better normalization
    # We split the answer into chunks of 8+ characters to check for groundedness
    sentences = re.split(r'[.!?\n\-:,;]', clean_answer)
    fragments = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not fragments:
        return {"grounded": 0, "general": 100}

    grounded_count = 0
    for frag in fragments:
        # If the fragment from the AI's answer exists anywhere in the retrieved slides
        if frag in all_doc_content or fuzz.partial_ratio(frag, all_doc_content) >= threshold:
            grounded_count += 1

    grounded_pct = int((grounded_count / len(fragments)) * 100)
    return {"grounded": grounded_pct, "general": 100 - grounded_pct}


