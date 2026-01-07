import os
import requests

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = requests.post(URL, headers=HEADERS, json={"inputs": texts})
    response.raise_for_status()
    data = response.json()

    # Normalize shape: wrap single vector into list
    if isinstance(data, list) and data and isinstance(data[0], (float, int)):
        data = [data]

    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise RuntimeError(f"Invalid embedding response: {data}")

    return data


def get_embeddings():
    """
    Compatibility wrapper expected by the rest of the system.
    Returns an object with an embed_documents method.
    """

    class _EmbeddingWrapper:
        def embed_documents(self, texts):
            return embed_texts(texts)

        def embed_query(self, text):
            return embed_texts([text])[0]

    return _EmbeddingWrapper()
