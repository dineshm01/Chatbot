from langchain_community.embeddings import HuggingFaceEndpointEmbeddings
import os

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

