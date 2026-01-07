import os
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

def get_embeddings():
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise RuntimeError("HF_API_KEY environment variable is missing")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
