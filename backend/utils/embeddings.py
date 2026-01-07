# backend/utils/embeddings.py

import os
from huggingface_hub import InferenceClient

HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(token=HF_API_KEY)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = client.feature_extraction(
        texts,
        model=MODEL
    )

    if isinstance(embeddings[0], float):
        embeddings = [embeddings]

    return embeddings


def get_embeddings():
    class _Wrapper:
        def embed_documents(self, texts):
            return embed_texts(texts)

        def embed_query(self, text):
            return embed_texts([text])[0]

    return _Wrapper()
