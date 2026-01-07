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
    if vectorstore is None:
        docs = []
    else:
        retriever = get_retriever()
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


    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    prompt = ChatPromptTemplate.from_template(
    """
    
    Conversation so far:
    {memory}

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
    {context}

    Question:
    {question}

    Answer:
    """
    )
    memory_text = ""
    for m in memory:
        memory_text += f"{m['role']}: {m['text']}\n"


    chain = (
        {
            "context": lambda _: context_text,
            "question": RunnablePassthrough(),
            "mode": lambda _: mode,
            "memory": lambda _: memory_text
        }
        | prompt
        | llm
    )

    answer = chain.invoke(question).content
    
    sources = []
    for d in filtered_docs[:3]:
        sources.append({
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page")
        })


    return {
        "text": answer,
        "confidence": compute_confidence(docs),
        "coverage": compute_coverage(docs),
        "sources": sources
    }


