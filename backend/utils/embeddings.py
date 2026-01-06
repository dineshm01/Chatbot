from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

