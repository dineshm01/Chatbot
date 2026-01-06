from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from utils.loaders import load_file
import os

def ingest_document(file_path):
    docs = load_file(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]

    embeddings_client = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = embeddings_client.embed_documents(texts)

    if not embeddings or len(embeddings) == 0:
        raise RuntimeError("HuggingFace returned empty embeddings — check HF_API_KEY or model availability")

    vectorstore = FAISS.from_embeddings(
        texts=texts,
        embeddings=embeddings,
        metadatas=[c.metadata for c in chunks]
    )

    vectorstore.save_local("faiss_index")
    return True
