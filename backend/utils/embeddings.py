import os
from huggingface_hub import InferenceClient
import time
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.embeddings import Embeddings
import numpy as np  


# Use one consistent name that matches your Railway Variable
HF_API_KEY = os.getenv("HF_API_KEY")

MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
client = InferenceClient(token=HF_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    # Try up to 3 times if the server is busy
    for attempt in range(3):
        try:
            embeddings = client.feature_extraction(texts, model=MODEL)
            
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
                
            if len(embeddings) > 0 and not isinstance(embeddings[0], list):
                embeddings = [embeddings]
                
            return embeddings
        except HfHubHTTPError as e:
            if "504" in str(e) and attempt < 2:
                time.sleep(2) # Wait 2 seconds before retrying
                continue
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








