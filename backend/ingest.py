from utils.loaders import load_file
from utils.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from datetime import datetime, timezone
import os
import re

client = MongoClient(
    os.getenv("MONGO_URI"),
    serverSelectionTimeoutMS=5000, # Wait 5 seconds for DB to wake up
    directConnection=False         # Ensure it looks for the full cluster
)
db = client["chatbot"]
raw_docs = db["raw_docs"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Force the folder to match the persistent volume mount point exactly
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

def extract_questions(text):
    questions = re.findall(r"\d+\..*?\?", text, flags=re.DOTALL)
    return [q.strip() for q in questions]

def ingest_document(file_path, user_id=None):    
    # 1. Load with Slide-Level Precision
    # We must ensure each slide is treated as a separate page to avoid "mashing"
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    raw_docs = loader.load()

    # 2. Optimized Chunking for Technical Content
    # chunk_size=800 is the "sweet spot" to keep a full slide's technical detail together
    # chunk_overlap=80 ensures we don't cut a definition in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # 3. Add Contextual Metadata
    # This ensures the retriever knows exactly which slide a "type" belongs to
    docs = text_splitter.split_documents(raw_docs)
    for doc in docs:
        # Ensure the filename is clean for the UI sources list
        doc.metadata["source"] = os.path.basename(file_path)
        # Preserve slide/page numbers for the 'Perfect Sync' highlighting
        if "page_number" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page_number"]

    # 4. Create Vector Store with MMR-ready Embeddings
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save locally so get_retriever() can find it
    vectorstore.save_local("faiss_index")
    return True




