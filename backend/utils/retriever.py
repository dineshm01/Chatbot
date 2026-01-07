from langchain_community.vectorstores import FAISS

def create_vectorstore(docs, embeddings):
    texts = [d.page_content for d in docs]
    if not texts:
        raise RuntimeError("No text to embed")

    vectors = embeddings.embed_documents(texts)

    if not vectors or len(vectors) != len(texts):
        raise RuntimeError("Embedding failed or returned empty vectors")

    return FAISS.from_texts(texts, embeddings)
