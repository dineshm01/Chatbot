from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.loaders import load_file
from utils.embeddings import get_embeddings


def ingest_document(file_path: str):
    print("INGEST: file =", file_path)

    docs = load_file(file_path)
    print("INGEST: docs =", len(docs) if docs else docs)

    if not docs:
        raise ValueError("0")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print("INGEST: chunks =", len(chunks))

    if not chunks:
        raise ValueError("0")

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    print("INGEST: vectors =", len(vectors))

    if not vectors:
        raise ValueError("0")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("INGEST: done")
    return True
