import os
from utils.loaders import load_file
from utils.splitter import split_documents
from utils.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS

PROCESSED_LOG = "data/processed_files.txt"


def ingest_document(file_path):
    os.makedirs("data", exist_ok=True)

    # Read processed files
    processed = set()
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            processed = set(line.strip() for line in f.readlines())

    # Skip if already processed
    if file_path in processed:
        print(f"Skipping already processed file: {file_path}")
        return

    # Load + split
    docs = load_file(file_path)
    chunks = split_documents(docs)

    embeddings = get_embeddings()

    # Load or create FAISS
    if os.path.exists("vectorstore"):
        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("vectorstore")

    # Log processed file
    with open(PROCESSED_LOG, "a") as f:
        f.write(file_path + "\n")
