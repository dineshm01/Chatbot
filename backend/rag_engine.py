from huggingface_hub import InferenceClient
import os
from utils.llm import call_llm
from rapidfuzz import fuzz
import re
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
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


# 1. Define the Template externally to avoid logic hardcoding
RAG_STRUCTURE_TEMPLATE = """
CONTEXT FROM TECHNICAL SLIDES:
{context_text}

USER QUESTION: 
{question}

STRICT ARCHITECTURAL RULES:
1. Copy technical facts EXACTLY. Do not rephrase or use your own knowledge.
2. SLIDE ISOLATION: Do not combine facts from different slides into a single sentence.
3. NO BRIDGE SENTENCES: Do not create introductory summaries like "There are two GANs" if that exact phrase is not in the bullets.
4. BULLET-ONLY: If the context is a list of bullet points, your answer must remain a list of bullet points.

Technical Fact-Based Answer:
""".strip()

# 2. Initialize the PromptTemplate
rag_prompt_custom = PromptTemplate(
    input_variables=["context_text", "question"],
    template=RAG_STRUCTURE_TEMPLATE
)

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
    # 1. Retrieval
    docs = retriever.invoke(question) if retriever else []

    if not docs:
        return {"text": "No relevant information found in the documents.", "chunks": []}

    context_text = truncate_docs(docs)
    
    # 3. DYNAMIC INJECTION: No hardcoded string inside the function
    final_prompt = rag_prompt_custom.format(
        context_text=context_text,
        question=question
    )
    
    answer = call_llm(final_prompt)

    # 3. LOGIC CHECK: Auto-reduction of unrelated topics
    # If the LLM still includes too much, we filter by keyword relevance to the question
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    q_keywords = [w.lower() for w in question.split() if len(w) > 3]
    
    # Filter: Keep a sentence only if it shares keywords with the question 
    # OR is part of a direct technical definition.
    filtered_sentences = []
    for s in sentences:
        if any(kw in s.lower() for kw in q_keywords) or len(filtered_sentences) < 2:
            filtered_sentences.append(s)
    
    final_text = " ".join(filtered_sentences)
    
    # 4. Return raw chunks for frontend highlighting sync
    raw_chunks = [d.page_content for d in docs]

    return {
        "text": final_text,
        "confidence": compute_confidence(docs),
        "coverage": compute_coverage(docs, final_text),
        "sources": [
            {
                "source": os.path.basename(d.metadata.get("source", "Doc")), 
                "page": d.metadata.get("page", "?")
            } for d in docs
        ],
        "raw_retrieval": raw_chunks,
        "chunks": raw_chunks 
    }
    






