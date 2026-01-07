from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
import os

def get_embeddings():
    token = os.getenv("HF_API_KEY")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
