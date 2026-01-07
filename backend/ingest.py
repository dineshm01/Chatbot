from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings

def ingest_document(filepath):
    docs = load_file(filepath)
    chunks = split_docs(docs)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)
