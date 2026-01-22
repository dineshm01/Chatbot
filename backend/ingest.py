from utils.loaders import load_file
from utils.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from datetime import datetime, timezone
import os
import re

client = MongoClient(os.getenv("MONGO_URI"))
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
    
    # 1. Use a specialized splitter for technical slides
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Increased to keep technical phrases like "Ian Goodfellow" together
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "‹#›"] # Added artifact as a separator
    )
    chunks = text_splitter.split_documents(docs)

    for chunk in chunks:
        # 2. DEEP NORMALIZATION: Strip all artifacts found in your PPTX
        content = " ".join(chunk.page_content.split())
        # Remove markdown and PPTX artifacts immediately
        clean_content = content.replace("*", "").replace("#", "").replace("‹#›", "").replace("窶ｹ#窶ｺ", "")
        chunk.page_content = clean_content.strip()

    # 3. Update user metadata
    db["user_metadata"].update_one(
        {"user_id": user_id},
        {"$set": {"last_upload": datetime.now(timezone.utc)}},
        upsert=True
    )

    if not chunks:
        raise RuntimeError("No valid text extracted from document")

    # 4. Clear old user data and save new clean index
    raw_docs.delete_many({"user_id": user_id}) 
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)






