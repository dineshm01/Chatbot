import os
from huggingface_hub import InferenceClient
import time
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.embeddings import Embeddings

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
client = InferenceClient(token=HF_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    # Pre-process texts to match ingestion normalization
    clean_texts = [" ".join(t.split()).replace("*", "").replace("#", "") for t in texts]
    
    for attempt in range(3):
        try:
            embeddings = client.feature_extraction(clean_texts, model=MODEL)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            if len(embeddings) > 0 and not isinstance(embeddings[0], list):
                embeddings = [embeddings]
            return embeddings
        except HfHubHTTPError as e:
            if "504" in str(e) and attempt < 2:
                time.sleep(2)
                continue
            raise e

class HFEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embed_texts(texts)

    def embed_query(self, text):
        res = embed_texts([text])
        return res[0] if res else []

def get_embeddings():
    return HFEmbeddings()
