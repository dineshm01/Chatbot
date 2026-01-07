from huggingface_hub import InferenceClient
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_utils import (
    load_vectorstore,
    get_retriever,
    truncate_docs,
    compute_confidence,
    compute_coverage
)

def generate_answer(question, mode, memory=None):
    memory = memory or []

    vectorstore = load_vectorstore()
    retriever = get_retriever() if vectorstore else None
    docs = retriever.invoke(question) if retriever else []

    filtered_docs = []
    for d in docs:
        if d.metadata.get("type") == "image" and d.metadata.get("ocr_confidence", 0) < 0.5:
            continue
        filtered_docs.append(d)

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
- Do NOT infer complexity or theory

Selected style: {mode}

Reference:
{context_text}

Question:
{question}

Answer:
"""

    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=os.getenv("HF_API_KEY")
    )

    answer = client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=False
    )

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




