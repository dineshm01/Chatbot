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

RAG_PERFECTION_TEMPLATE = """
YOU ARE A TECHNICAL EXTRACTION ENGINE.
DOCUMENT CONTEXT:
{context_text}

USER QUESTION: 
{question}

STRICT OPERATING RULES:
1. READ every word of the provided CONTEXT before answering.
2. If the answer involves multiple parts (e.g., 'What are the types?'), you must find and list EVERY part mentioned in the context.
3. DO NOT summarize. Copy the technical definitions exactly as they are written.
4. If a logical connection is not explicitly stated in the context, do not invent one.
5. If the context is insufficient to answer perfectly, say: "The document does not contain enough data for a perfect answer."

TECHNICAL RESPONSE:
""".strip()

rag_prompt_custom = PromptTemplate(
    input_variables=["context_text", "question"],
    template=RAG_PERFECTION_TEMPLATE
)

def generate_answer(question, mode, memory=None, strict=True, user_id=None): 
    # 2. RETRIEVAL (Ensure k=15 is set in rag_utils.py)
    retriever = get_retriever()
    docs = retriever.invoke(question) if retriever else []

    if not docs:
        return {"text": "No relevant information found in the documents.", "chunks": []}

    context_text = truncate_docs(docs)

    # 3. DYNAMIC PROMPT GENERATION
    final_prompt = rag_prompt_custom.format(
        context_text=context_text,
        question=question
    )
    
    # call_llm must use temperature=0.0 to prevent rephrasing
    answer = call_llm(final_prompt)
    
    # 4. LOGIC CHECK: Auto-reduction of unrelated topics
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    q_keywords = [w.lower() for w in question.split() if len(w) > 3]
    
    filtered_sentences = []
    for s in sentences:
        if any(kw in s.lower() for kw in q_keywords) or len(filtered_sentences) < 3:
            filtered_sentences.append(s)
    
    final_text = "\n\n".join(filtered_sentences)
    
    # 5. SYNCHRONIZED RETURN FOR FRONTEND highlighting
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

