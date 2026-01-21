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
    Core RAG engine that retrieves data, generates a style-specific answer, 
    and verifies grounding before returning.
    """
    # 1. Check for specific document index queries (e.g., "What is the 1st question?")
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
                "chunks": [item["text"]]
            }

    # 2. Retrieve relevant slides from the persistent FAISS index
    memory = memory or []
    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    # 3. Filter retrieved content for quality
    filtered_docs = [
        d for d in docs
        if not (d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5)
    ]

    # 4. Handle Strict Mode restrictions
    if strict and not filtered_docs:
        return {
            "text": "❌ Strict mode: No relevant information found in your documents.",
            "confidence": "Strict mode",
            "coverage": {"grounded": 0, "general": 0},
            "sources": [],
            "chunks": []
        }

    # 5. Build the prompt and call the LLM
    context_text = truncate_docs(filtered_docs)
    memory_text = "\n".join(f"{m['role']}: {m['text']}" for m in memory)
    prompt = f"Conversation:\n{memory_text}\n\nStyle: {mode}\nReference:\n{context_text}\nQuestion: {question}\nAnswer:"
    
    answer = call_llm(prompt)

    # 6. Verify grounding using the updated normalization logic
    grounded_sentences, _ = extract_grounded_spans(answer, filtered_docs)
    coverage = compute_coverage(filtered_docs, answer)

    return {
        "text": answer.strip(),
        "confidence": compute_confidence(filtered_docs),
        "coverage": coverage,
        "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in filtered_docs[:3]],
        "chunks": grounded_sentences, # CRITICAL: These trigger blue highlights in App.js
        "debug": {"retrieved_docs": len(filtered_docs)}
    }    
