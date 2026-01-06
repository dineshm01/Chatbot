from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.loaders import load_file
from utils.embeddings import get_embeddings


def ingest_document(file_path: str):
    print("INGEST: file =", file_path)

    docs = load_file(file_path)
    print("INGEST: docs =", len(docs) if docs else docs)
    if not docs:
        raise ValueError("No documents loaded")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print("INGEST: chunks =", len(chunks))
    if not chunks:
        raise ValueError("No chunks created")

    embeddings = get_embeddings()

    vectors = []
    for c in chunks:
        v = embeddings.embed_query(c.page_content)
        vectors.append(v)

    print("INGEST: vectors =", len(vectors))

    if len(vectors) != len(chunks):
        print("WARNING: vector count mismatch")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("INGEST: done")
    return True
