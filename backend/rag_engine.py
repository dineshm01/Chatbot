from huggingface_hub import InferenceClient
import os
from rag_utils import (
    load_vectorstore,
    get_retriever,
    truncate_docs,
    compute_confidence,
    compute_coverage
)

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

    client = InferenceClient(token=os.getenv("HF_API_KEY"))

    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3
    )

    answer = response.choices[0].message.content

    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in filtered_docs[:3]
    ]

    return {
        "text": answer.strip(),
        "confidence": compute_confidence(docs),
        "coverage": compute_coverage(docs),
        "sources": sources
    }
