# backend/rag_utils.py
import os
import sys

# Absolute path of project root: D:\Langchain Project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print("DEBUG PROJECT_ROOT:", PROJECT_ROOT)
print("DEBUG exists:", os.path.exists(PROJECT_ROOT))
print("DEBUG utils exists:", os.path.exists(os.path.join(PROJECT_ROOT, "utils")))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG sys.path:", sys.path)

from utils.embeddings import get_embeddings
from utils.retriever import create_vectorstore
from utils.loaders import load_file
from langchain_community.vectorstores import FAISS

VECTOR_DIR = "vectorstore"

def load_vectorstore():
    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise RuntimeError("Vectorstore not found. Upload a document first.")
    return FAISS.load_local(
        VECTOR_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def get_retriever():
    embeddings = get_embeddings()

    if not os.path.exists("vectorstore/index.faiss"):
        raise RuntimeError("Vectorstore missing. Upload a document first.")


    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

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
    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars >= 800:
        return "🟢 Confidence: Fully covered by notes"
    if total_chars >= 200:
        return "🟡 Confidence: Partially inferred"
    return "🔵 Confidence: General knowledge"

def compute_coverage(docs, max_chars=1200):
    if not docs:
        return 0
    total_chars = sum(len(d.page_content) for d in docs)
    return min(100, int((total_chars / max_chars) * 100))




