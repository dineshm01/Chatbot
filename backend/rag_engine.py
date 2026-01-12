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

def exact_lookup(question):
    q = question.lower()

    # match patterns like:
    # "what is 35th question"
    # "show 10th item"
    # "get 5th entry"
    m = re.search(r"(?:what is|show|get)?\s*(\d+)(st|nd|rd|th)?\s+(question|item|entry)", q)

    if not m:
        return None

    idx = int(m.group(1))
    return raw_docs.find_one({"index": idx})

def docs_are_relevant(question, docs, threshold=60):
    if not docs:
        return False

    doc_text = " ".join(d.page_content for d in docs).lower()
    score = fuzz.partial_ratio(question.lower(), doc_text)

    return score >= threshold


def extract_grounded_spans(answer, docs, threshold=0.2):
    grounded = []
    debug = []

    doc_text = " ".join(d.page_content.lower() for d in docs)
    doc_tokens = set(re.findall(r"\w+", doc_text))

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 40]

    for s in sentences:
        sent_tokens = set(re.findall(r"\w+", s.lower()))
        if not sent_tokens:
            continue

        overlap = len(sent_tokens & doc_tokens) / len(sent_tokens)
        debug.append({"sentence": s[:120], "overlap": round(overlap, 2)})

        if overlap >= threshold:
            grounded.append(s)

    return grounded, debug
    
def generate_answer(question, mode, memory=None, strict=False):
    exact = exact_lookup(question)
    if exact:
        return {
            "text": exact["text"],
            "confidence": "Exact match from document",
            "coverage": {"grounded": 100, "general": 0},
            "sources": [{"source": exact.get("source"), "page": exact.get("page")}],
            "chunks": [exact["text"]]
        }

    memory = memory or []

    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    filtered_docs = [
        d for d in docs
        if not (d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5)
    ]

    
    if strict:
        if not filtered_docs:
            return {
                "text": "❌ Strict mode: No documents available. Upload material first.",
                "confidence": "Strict mode",
                "coverage": {"grounded": 0, "general": 0},
                "sources": [],
                "chunks": []
            }

        if not docs_are_relevant(question, filtered_docs):
            return {
                "text": "❌ Strict mode: No relevant information found in your documents.",
                "confidence": "Strict mode",
                "coverage": {"grounded": 0, "general": 0},
                "sources": [],
                "chunks": []
            }    
            
    context_text = "" if mode == "Diagram" else truncate_docs(filtered_docs)

    if not filtered_docs and not memory:
        return {
            "text": "I couldn't find this in the uploaded documents. Please try rephrasing or upload relevant material.",
            "confidence": "No documents",
            "coverage": 0,
            "sources": []
        }

    memory_text = "\n".join(f"{m['role']}: {m['text']}" for m in memory)

    prompt = f"""
    Conversation so far:
    {memory_text}

    Style: {mode}

    Reference:
    {context_text}

    Question:
    {question}

    Answer:
    """
    
    if strict:
        answer = call_llm(prompt)
        grounded_sentences, _ = extract_grounded_spans(answer, filtered_docs, threshold=0.6)

        if not grounded_sentences:
            return {
                "text": "❌ Strict mode: Answer could not be grounded in documents.",
                "confidence": "Strict mode",
                "coverage": {"grounded": 0, "general": 0},
                "sources": [],
                "chunks": []
            }
    else:
        answer = call_llm(prompt)


    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in filtered_docs[:3]
    ]

    
    coverage = compute_coverage(docs, answer)

    grounded_sentences, debug = extract_grounded_spans(answer, filtered_docs, threshold=0.55)


    return {
        "text": answer.strip(),
        "confidence": compute_confidence(docs),
        "coverage": coverage,
        "sources": sources,
        "chunks": grounded_sentences,
        "debug": {
            "retrieved_docs": len(filtered_docs),
            "doc_text_length": sum(len(d.page_content) for d in filtered_docs),
            "overlaps": debug
        }
    }
    

