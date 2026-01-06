from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings

VECTOR_DIR = "vectorstore"

def load_vectorstore():
    return FAISS.load_local(VECTOR_DIR, get_embeddings(), allow_dangerous_deserialization=True)
