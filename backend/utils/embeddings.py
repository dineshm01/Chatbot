from huggingface_hub import InferenceClient
import os

client = InferenceClient(token=os.getenv("HF_API_KEY"))

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    return client.feature_extraction(
        model="sentence-transformers/all-MiniLM-L6-v2",
        inputs=texts
    )

def get_embeddings():
    class _EmbeddingWrapper:
        def embed_documents(self, texts):
            return embed_texts(texts)

        def embed_query(self, text):
            return embed_texts([text])[0]

    return _EmbeddingWrapper()
