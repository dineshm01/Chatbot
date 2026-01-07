from utils.loaders import load_file
from utils.embeddings import embed_texts
from langchain_community.vectorstores import FAISS

VECTOR_DIR = "vectorstore"

def ingest_document(filepath):
    docs = load_file(filepath)

    texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
    metadatas = [d.metadata for d in docs if d.page_content and d.page_content.strip()]

    if not texts:
        raise RuntimeError("No valid text extracted from document")

    vectors = embed_texts(texts)

    if len(texts) != len(vectors):
        raise RuntimeError("Mismatch between texts and embeddings")

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        metadatas=metadatas
    )

    vectorstore.save_local(VECTOR_DIR)
