from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from utils.embeddings import get_embeddings
import os

VECTOR_DIR = "/app/vectorstore"

def ingest_document(filepath):
    loader = UnstructuredPowerPointLoader(filepath)

    try:
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"PPTX parsing failed: {e}")

    print("DEBUG docs type:", type(docs))
    print("DEBUG docs count:", len(docs))

    if not docs:
        raise RuntimeError("No documents loaded from file")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    # Remove empty chunks
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    print("DEBUG non-empty chunks:", len(chunks))


    if not chunks:
        raise RuntimeError("No chunks created")

    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectors = embeddings.embed_documents(texts)

    if not vectors or not isinstance(vectors[0], list):
        raise RuntimeError("Embedding API returned invalid vectors")

    vectorstore = FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)
    
    print("DEBUG vectorstore saved at:", VECTOR_DIR)
    print("Saved files:", os.listdir(VECTOR_DIR))






