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


def generate_answer(question, mode, memory=None, strict=False, user_id=None): 
    """
    Core RAG engine with existing index query support and fixed highlighting logic.
    """
    # KEEP: Your existing logic for specific index queries
    m = re.search(r"(\d+)(st|nd|rd|th)?\s+question", question.lower())
    if m:
        idx = int(m.group(1))
        cursor = raw_docs.find({"user_id": user_id}, sort=[("index", 1)]).skip(idx - 1).limit(1)
        items = list(cursor)
        if items:
            item = items[0]
            return {
                "text": item["text"],
                "confidence": "Exact match from document",
                "coverage": {"grounded": 100, "general": 0},
                "sources": [{"source": item.get("source"), "page": item.get("page")}],
                "chunks": [item["text"]],
                "raw_retrieval": [item["text"]] # Added for highlighter
            }

    # UPDATE: New optimized RAG flow
    memory = memory or []
    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    # FIX: Per-chunk filtering to capture specific technical terms (e.g., GAN, SRGAN)
    # This prevents the context-dilution bug found in your current docs_are_relevant logic
    filtered_docs = []
    for d in docs:
        if d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5:
            continue
        # Lower threshold for individual chunks ensures specifics like 'Ian Goodfellow' aren't filtered out
        score = fuzz.partial_ratio(question.lower(), d.page_content.lower())
        if score >= 25: 
            filtered_docs.append(d)

    # Handle Strict Mode
    if strict and not filtered_docs:
        return {
            "text": "❌ Strict mode: No relevant technical details found in the slides.",
            "confidence": "Strict mode",
            "coverage": {"grounded": 0, "general": 0},
            "sources": [],
            "chunks": [],
            "raw_retrieval": []
        }

    # Fallback to top docs if not in strict mode
    final_docs = filtered_docs if filtered_docs else docs[:3]

    # Build prompt and generate answer
    context_text = truncate_docs(final_docs)
    memory_text = "\n".join(f"{m['role']}: {m['text']}" for m in memory)
    prompt = f"Conversation:\n{memory_text}\n\nStyle: {mode}\nReference:\n{context_text}\nQuestion: {question}\nAnswer:"
    
    answer = call_llm(prompt)

    # Calculate grounding based on the same docs sent to LLM
    coverage = compute_coverage(final_docs, answer)

    # Inside generate_answer function...
    cleaned_chunks = [d.page_content.replace("‹#›", "").replace("窶ｹ#窶ｺ", "").strip() for d in filtered_docs]

    return {
        "text": answer.strip(),
        "confidence": compute_confidence(filtered_docs),
        "coverage": coverage,
        "sources": [{"source": os.path.basename(d.metadata.get("source", "Doc")), "page": d.metadata.get("page", "?")} for d in filtered_docs[:3]],
        "chunks": cleaned_chunks,
        "raw_retrieval": cleaned_chunks
    }

