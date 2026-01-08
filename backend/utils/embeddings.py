# utils/embeddings.py

import os
from huggingface_hub import InferenceClient

HF_API_KEY = os.getenv("HF_API_KEY")

# Model that supports feature-extraction
MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

client = InferenceClient(token=HF_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = client.feature_extraction(
        texts,
        model=MODEL
    )

    # Ensure output is always list[list[float]]
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
