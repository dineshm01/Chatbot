import os
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings

VECTOR_DIR = "/app/vectorstore"

def load_vectorstore():
    if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        return None

    return FAISS.load_local(
        VECTOR_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def get_retriever():
    vectorstore = load_vectorstore()
    if not vectorstore:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 6})

def truncate_docs(docs, max_chars=1500):
    text = ""
    for d in docs:
        if len(text) + len(d.page_content) > max_chars:
            break
        text += d.page_content + "\n"
    return text.strip()

def compute_confidence(docs):
    if not docs:
        return "🔵 Confidence: General knowledge"
    total = sum(len(d.page_content) for d in docs)
    if total >= 800:
        return "🟢 Confidence: Fully covered by notes"
    if total >= 200:
        return "🟡 Confidence: Partially inferred"
    return "🔵 Confidence: General knowledge"

def compute_coverage(docs, max_chars=1200):
    if not docs:
        return 0
    total = sum(len(d.page_content) for d in docs)
    return min(100, int((total / max_chars) * 100))
