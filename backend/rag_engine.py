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
    memory = memory or []

    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    filtered_docs = [
        d for d in docs
        if not (d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5)
    ]

    
    if strict and not docs_are_relevant(question, filtered_docs):
        return {
            "text": "❌ Strict mode: No relevant documents found. Please upload material.",
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

    You must follow the rules for the selected answer style.

    If the selected style is "Concise":
    - Max 2 sentences
    - Direct definition only

    If the selected style is "Detailed":
    - Start with definition
    - Explain step-by-step
    - Teacher style

    If the selected style is "Exam":
    - Bullet points only
    - 2–5 mark answer
    - No explanations

    If the selected style is "ELI5":
    - Very simple language
    - Friendly tone
    - No jargon

    If the selected style is "Compare":
    - Markdown table ONLY
    - First column: Aspect
    - Compare at least two concepts
    - No text outside table

    If the selected style is "Diagram":
    - Explain the diagram step-by-step
    - Use clear headings
    - Explain flow and relationships only as shown
    - Do NOT infer complexity, performance, or internal behavior
    - Do NOT add theory not shown in the diagram
    - Student-friendly explanation

    Selected style: {mode}
    
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

    grounded_sentences, debug = extract_grounded_spans(answer, filtered_docs, threshold=0.55)


    return {
        "text": answer.strip(),
        "display_text": highlighted,
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
























