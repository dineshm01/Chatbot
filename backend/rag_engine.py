from huggingface_hub import InferenceClient
import os
from utils.llm import call_llm
from rag_utils import (
    load_vectorstore,
    get_retriever,
    truncate_docs,
    compute_confidence,
    compute_coverage
)

def extract_grounded_sentences(answer, docs):
    grounded = []
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    for sent in answer_sentences:
        sent_l = sent.lower()
        for d in docs:
            if any(w in d.page_content.lower() for w in sent_l.split() if len(w) > 5):
                grounded.append(sent)
                break
    return grounded

def generate_answer(question, mode, memory=None):
    memory = memory or []

    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

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

    grounded_sentences = extract_grounded_sentences(answer, filtered_docs)

    return {
        "text": answer.strip(),
        "confidence": compute_confidence(docs),
        "coverage": coverage,
        "sources": sources,
        "chunks": grounded_sentences
    }








