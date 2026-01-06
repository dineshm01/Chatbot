import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
