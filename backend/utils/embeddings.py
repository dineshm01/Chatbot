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
        # The InferenceClient often returns a numpy array
        embeddings = client.feature_extraction(texts, model=MODEL)
        
        # FIX: Do not use 'if embeddings:' which causes the ambiguous error
        # Instead, check the dimensionality using .ndim if it's a numpy array
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = [embeddings.tolist()]
            else:
                embeddings = embeddings.tolist()
        # Handle case where it might already be a list
        elif isinstance(embeddings, list):
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
        return res[0] if res else []

def get_embeddings():
    return HFEmbeddings()


