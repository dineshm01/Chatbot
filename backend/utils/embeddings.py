from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

def get_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        HF_API_KEY=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
