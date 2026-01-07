from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from utils.embeddings import get_embeddings
import os

VECTOR_DIR = "vectorstore"

def ingest_document(filepath):
    loader = UnstructuredPowerPointLoader(filepath)

    try:
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"PPTX parsing failed: {e}")

    print("DEBUG docs type:", type(docs))
    print("DEBUG docs count:", len(docs))

    if not docs:
        raise RuntimeError("No documents loaded from file")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print("DEBUG chunks:", len(chunks))

    if not chunks:
        raise RuntimeError("No chunks created")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)

    print("DEBUG vectorstore saved at:", VECTOR_DIR)
