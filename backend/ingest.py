from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from utils.embeddings import get_embeddings
import os

VECTOR_DIR = "vectorstore"

def ingest_document(filepath):
    loader = UnstructuredPowerPointLoader(filepath)
    docs = loader.load()  # This returns List[Document]

    if not docs:
        raise RuntimeError("No documents loaded from file")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("No chunks created")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)
