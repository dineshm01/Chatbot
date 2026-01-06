from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

def get_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
