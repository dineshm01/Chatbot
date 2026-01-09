from huggingface_hub import InferenceClient
import os
from utils.llm import call_llm
from rapidfuzz import fuzz
import re
from rag_utils import (
    load_vectorstore,
    get_retriever,
    truncate_docs,
    compute_confidence,
    compute_coverage
)


def extract_grounded_spans(answer, docs, threshold=0.2):
    grounded = []
    debug = []

    doc_text = " ".join(d.page_content.lower() for d in docs)
    doc_tokens = set(re.findall(r"\w+", doc_text))

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]

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
    memory = memory or []

    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    if strict and not docs:
        return {
            "text": "❌ Strict mode: No relevant documents found. Please upload material.",
            "confidence": "Strict mode",
            "coverage": {"grounded": 0, "general": 0},
            "sources": [],
            "chunks": []
        }

    filtered_docs = [
        d for d in docs
        if not (d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5)
    ]

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

    answer = call_llm(prompt)

    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in filtered_docs[:3]
    ]

    

    coverage = compute_coverage(docs, answer)

    grounded_sentences, debug = extract_grounded_spans(answer, filtered_docs, threshold=0.25)



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
















