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


def compute_coverage(docs, answer=None, threshold=85):
    """
    Measures groundedness by checking what percentage of the LLM's answer 
    exists word-for-word in the retrieved document chunks.
    """
    if not docs or not answer:
        return {"grounded": 0, "general": 100}

    # 1. Prepare document text (the Source of Truth)
    # FIX: Correctly iterate through 'docs' to build the source text
    doc_text = " ".join([doc.page_content for doc in docs]).lower()
    
    # 2. Split the LLM answer into meaningful sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 30]

    if not sentences:
        return {"grounded": 0, "general": 100}

    grounded_count = 0
    for s in sentences:
        clean_s = s.lower()
        
        # 3. Perform a high-threshold fuzzy match
        score = fuzz.partial_ratio(clean_s, doc_text)
        
        if score >= threshold:
            grounded_count += 1

    # 4. Calculate final percentages
    grounded_pct = int((grounded_count / len(sentences)) * 100)
    general_pct = 100 - grounded_pct

    return {
        "grounded": grounded_pct,
        "general": general_pct
    }

