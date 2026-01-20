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

VECTOR_DIR = "vectorstore"

def extract_questions(text):
    questions = re.findall(r"\d+\..*?\?", text, flags=re.DOTALL)
    return [q.strip() for q in questions]

# Change the function signature to accept user_id
def ingest_document(filepath, user_id): 
    docs = load_file(filepath)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Update or Insert user metadata with a timestamp
    db["user_metadata"].update_one(
        {"user_id": user_id},
        {"$set": {"last_upload": datetime.now(timezone.utc)}},
        upsert=True
    )

    if not chunks:
        raise RuntimeError("No valid text extracted")

    # FIX: Only delete documents belonging to THIS user
    raw_docs.delete_many({"user_id": user_id}) 

    # When inserting chunks later (if you add that logic), 
    # ensure you include "user_id": user_id in the document.
    
    # Store chunks in FAISS
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)


