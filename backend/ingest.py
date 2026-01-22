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
    docs = load_file(filepath)
    
    # LANGCHAIN STEP: Recursive Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, # Large enough to keep "Ian Goodfellow (2016)" together
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "‹#›"]
    )
    chunks = text_splitter.split_documents(docs)

    for chunk in chunks:
        # Deep Normalization: Clean artifacts BEFORE embedding
        content = " ".join(chunk.page_content.split())
        chunk.page_content = content.replace("‹#›", "").replace("窶ｹ#窶ｺ", "").strip()

    # LANGCHAIN STEP: Vector Storage (FAISS)
    raw_docs.delete_many({"user_id": user_id}) 
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)
    
