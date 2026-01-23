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
    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    if not docs:
        return {"text": "No relevant information found in the documents.", "chunks": []}

    context_text = truncate_docs(docs)
    
    prompt = (
        f"SOURCE DATA:\n{context_text}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"STRICT INSTRUCTION:\n"
        f"1. Extract ONLY the information that directly answers the USER QUESTION.\n"
        f"2. DISCARD any text from the SOURCE DATA that is not relevant to the question.\n"
        f"3. Copy the relevant sentences EXACTLY as they appear.\n"
        f"4. If the data is not relevant, do not include it in your response.\n\n"
        f"Focused Technical Answer:"
    )
    
    answer = call_llm(prompt)
    
    # THE FIX: Do not use .split().join() here. 
    # Keep the original raw text so App.js can find the newlines.
    raw_chunks = [d.page_content for d in docs]

    return {
        "text": answer.strip(),
        "confidence": compute_confidence(docs),
        "coverage": compute_coverage(docs, answer),
        "sources": [
            {
                "source": os.path.basename(d.metadata.get("source", "Doc")), 
                "page": d.metadata.get("page", "?")
            } for d in docs
        ],
        "raw_retrieval": raw_chunks, # Pass raw text
        "chunks": raw_chunks        # Pass raw text
    }

