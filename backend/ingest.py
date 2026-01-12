from utils.loaders import load_file
from utils.embeddings import embed_texts, get_embeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import os
import re

client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot"]
raw_docs = db["raw_docs"]


VECTOR_DIR = "vectorstore"


def extract_questions(text):
    questions = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    buffer = ""
    for line in lines:
        # Detect numbered question start like "35.", "35)", "35 -"
        if re.match(r"^\d+[\.\)\-]\s*", line):
            if buffer:
                questions.append(buffer.strip())
            buffer = line
        else:
            buffer += " " + line

    if buffer:
        questions.append(buffer.strip())

    return questions


def ingest_document(filepath):
    docs = load_file(filepath)

    texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
    metadatas = [d.metadata for d in docs if d.page_content and d.page_content.strip()]

    if not texts:
        raise RuntimeError("No valid text extracted from document")

    vectors = embed_texts(texts)

    if len(texts) != len(vectors):
        raise RuntimeError("Mismatch between texts and embeddings")

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

    embeddings = get_embeddings()  # <-- this was missing

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,        # <-- pass embedding wrapper
        metadatas=metadatas
    )

    vectorstore.save_local(VECTOR_DIR)





