from huggingface_hub import InferenceClient
import os
from utils.llm import call_llm
from rapidfuzz import fuzz
import re
from pymongo import MongoClient
import os
from rag_utils import (
    load_vectorstore,
    get_retriever,
    truncate_docs,
    compute_confidence,
    compute_coverage
)

client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot"]
raw_docs = db["raw_docs"]


def docs_are_relevant(question, docs, threshold=30):
    if not docs:
        return False

    doc_text = " ".join(d.page_content for d in docs).lower()
    score = fuzz.partial_ratio(question.lower(), doc_text)

    return score >= threshold

def extract_grounded_spans(answer, docs, threshold=0.8):
    """
    Identifies which parts of the AI answer are directly supported by the documents.
    Uses identical cleaning logic to the frontend to ensure blue highlights appear.
    """
    grounded = []
    
    # 1. Build a normalized Source of Truth from the retrieved slides
    # We remove markdown artifacts and standardize whitespace to prevent matching failures.
    doc_text = " ".join([" ".join(d.page_content.split()) for d in docs]).lower()
    doc_text = doc_text.replace("*", "").replace("#", "")

    # 2. Split the AI's rephrased answer into individual sentences for verification
    # We use a 30-character minimum to avoid highlighting generic words like "GANs".
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 30]

    for s in sentences:
        # 3. Deep Normalize the AI's sentence for comparison
        clean_s = " ".join(s.lower().split()).replace("*", "").replace("#", "")
        
        # 4. Check for direct inclusion or high-confidence fuzzy matching
        # Fuzzy matching handles minor rephrasing between your slides and the AI answer.
        if clean_s in doc_text or fuzz.partial_ratio(clean_s, doc_text) >= (threshold * 100):
            grounded.append(s)

    return grounded, [] # Returning empty list for second return value to keep it simple


def generate_answer(question, mode, memory=None, strict=True, user_id=None): 
    # 1. Retrieve data
    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    # 2. Strict Filtering (Internal default)
    filtered_docs = [d for d in docs if not (d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5)]

    if not filtered_docs:
        return {
            "text": "I am sorry, but the provided documents do not contain information to answer this question.",
            "confidence": "No context found",
            "coverage": {"grounded": 0, "general": 0},
            "sources": [], "chunks": [], "raw_retrieval": []
        }

    # 3. Permanent Strict Prompting
    context_text = truncate_docs(filtered_docs)
    # The prompt now forbids general knowledge
    prompt = (
        f"Context: {context_text}\n\n"
        f"Task: Answer the question using ONLY the context provided above. "
        f"If the answer is not in the context, say you do not know. "
        f"Do not use external information. Keep technical keywords exact.\n\n"
        f"Question: {question}\nAnswer:"
    )
    
    answer = call_llm(prompt)
    coverage = compute_coverage(filtered_docs, answer)

    # Standardize retrieval data for the frontend highlighter
    cleaned_retrieval = [" ".join(d.page_content.split()).replace("‹#›", "").strip() for d in filtered_docs]

    return {
        "text": answer.strip(),
        "confidence": compute_confidence(filtered_docs),
        "coverage": coverage,
        "sources": [{"source": os.path.basename(d.metadata.get("source", "Doc")), "page": d.metadata.get("page", "?")} for d in filtered_docs[:3]],
        "chunks": cleaned_retrieval,
        "raw_retrieval": cleaned_retrieval 
    }

