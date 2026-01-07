from langchain_community.vectorstores import FAISS
from utils.embeddings import embed_texts

def load_vectorstore():
    return FAISS.load_local("vectorstore", embed_texts)
