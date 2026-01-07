from langchain_community.vectorstores import FAISS

def create_vectorstore(docs, embeddings):
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)

    return FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embeddings
    )

