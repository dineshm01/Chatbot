import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPowerPointLoader, PyMuPDFLoader
from utils.embeddings import get_embeddings

VECTOR_DIR = "vectorstore"
os.makedirs(VECTOR_DIR, exist_ok=True)

def ingest_document(filepath: str):
    print(f"INGEST: file = {filepath}")

    if filepath.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
    elif filepath.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(filepath)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    print(f"INGEST: docs = {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"INGEST: chunks = {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks created")

    embeddings = get_embeddings()

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    print(f"INGEST: vectors = {len(vectors)}")

    if not vectors:
        raise RuntimeError("No embeddings generated")

    vectorstore = FAISS.from_texts([c.page_content for c in chunks], embeddings)
    vectorstore.save_local(VECTOR_DIR)

    print("INGEST: vectorstore saved")

