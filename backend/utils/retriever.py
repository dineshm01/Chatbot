from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings

VECTOR_DIR = "vectorstore"


def create_vectorstore(docs):
    """
    Creates and saves a FAISS vectorstore from documents.
    """
    embeddings = get_embeddings()

    if not docs:
        raise ValueError("No documents to index")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DIR)

    return vectorstore
