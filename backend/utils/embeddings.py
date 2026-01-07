import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

def get_embeddings():
    HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY not set in environment")

    return HuggingFaceInferenceAPIEmbeddings(
        HF_API_KEY=HF_API_KEY,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
