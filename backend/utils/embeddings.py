import os
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
import numpy as np  


# Use one consistent name that matches your Railway Variable
HF_API_KEY = os.getenv("HF_API_KEY")

MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
client = InferenceClient(token=HF_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        embeddings = client.feature_extraction(texts, model=MODEL)
        
        # FIX: Immediately convert to a standard list to stop the crash
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
            
        # Ensure the result is a list of lists
        if len(embeddings) > 0 and not isinstance(embeddings[0], list):
            embeddings = [embeddings]
            
        return embeddings
    except Exception as e:
        print(f"HuggingFace Embedding Error: {e}")
        raise e


class HFEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embed_texts(texts)

    def embed_query(self, text):
        res = embed_texts([text])
        if not res:
            return []
        result = res[0]
        if hasattr(result, "tolist"):
            result = result.tolist()
        return result

def get_embeddings():
    return HFEmbeddings()







