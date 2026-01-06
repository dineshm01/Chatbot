import os
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings
from utils.loaders import load_file
from utils.splitter import split_documents

INDEX_DIR = "faiss_index"


def ingest_document(filepath: str):
    """
    Loads a file, splits it into chunks, embeds them and stores into FAISS.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # 1. Load document
    docs = load_file(filepath)
    if not docs:
        raise ValueError("No content loaded from document")

    # 2. Split into chunks
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No chunks created from document")

    # 3. Get embeddings (lazy loaded)
    embeddings = get_embeddings()

    # 4. Create / update FAISS index
    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Save index
    vectorstore.save_local(INDEX_DIR)

    return {"status": "ok", "chunks": len(chunks)}
