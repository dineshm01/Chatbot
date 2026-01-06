import logging
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.loaders import load_file
from utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

def ingest_document(file_path: str):
    try:
        logger.info(f"INGEST: file = {file_path}")

        docs = load_file(file_path)
        logger.info(f"INGEST: docs = {len(docs)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        logger.info(f"INGEST: chunks = {len(chunks)}")

        embedder = get_embeddings()

        # 👇 Important: manually embed text
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        vectors = embedder.embed_documents(texts)
        logger.info(f"INGEST: vectors = {len(vectors)}")

        vectorstore = FAISS.from_embeddings(vectors, texts, metadatas=metadatas)
        vectorstore.save_local("faiss_index")

        logger.info("INGEST: done successfully")
        return True

    except Exception as e:
        logger.error(f"Upload/Ingest error: {repr(e)}")
        raise
