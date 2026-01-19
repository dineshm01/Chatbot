from utils.loaders import load_file
from utils.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import os
import re

client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot"]
raw_docs = db["raw_docs"]

VECTOR_DIR = "vectorstore"

def extract_questions(text):
    questions = re.findall(r"\d+\..*?\?", text, flags=re.DOTALL)
    return [q.strip() for q in questions]

def ingest_document(filepath):
    print("Ingesting:", filepath)

    docs = load_file(filepath)

    # Remove empty docs
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        raise RuntimeError("No valid text extracted")

    # Reset raw question store
    raw_docs.delete_many({})

    index = 1
    for d in docs:
        questions = extract_questions(d.page_content)
        for q in questions:
            raw_docs.insert_one({
                "index": index,
                "text": q,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page")
            })
            index += 1

    print("Inserted questions:", index - 1)

    # ✅ CORRECT FAISS CREATION (KEY FIX)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DIR)
