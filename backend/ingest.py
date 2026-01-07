from utils.embeddings import embed_texts
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def ingest_document(filepath):
    docs = load_file(filepath)
    texts = [d.page_content for d in docs if d.page_content.strip()]
    metadatas = [d.metadata for d in docs if d.page_content.strip()]

    vectors = embed_texts(texts)

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        metadatas=metadatas
    )
    vectorstore.save_local("vectorstore")
