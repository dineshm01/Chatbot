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

def ingest_document(filepath, user_id): 
    # 1. Load slides one by one via the updated loader
    docs = load_file(filepath)
    
    # 2. THE MEMORY FIX: Generator-based Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "‹#›"]
    )
    
    processed_chunks = []
    
    # Process each slide's Document object individually to save RAM
    for doc in docs:
        # Deep Normalization: Clean slide artifacts
        content = doc.page_content
        # Remove common PPTX artifacts like slide number markers
        content = content.replace("‹#›", "").replace("窶ｹ#窶ｺ", "")
        content = " ".join(content.split()).strip()
        
        # Split only this slide's content
        slide_chunks = text_splitter.split_text(content)
        
        for chunk_text in slide_chunks:
            from langchain_core.documents import Document
            processed_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "user_id": user_id,
                    "ingested_at": datetime.now(timezone.utc).isoformat()
                }
            ))

    # 3. Vector Storage (FAISS)
    # Clear old records for this specific user to keep the DB clean
    raw_docs.delete_many({"user_id": user_id}) 
    
    embeddings = get_embeddings()
    
    # Batch processing for FAISS to prevent memory spikes during embedding
    if processed_chunks:
        vectorstore = FAISS.from_documents(processed_chunks, embeddings)
        # Save to the persistent volume
        vectorstore.save_local(VECTOR_DIR)
