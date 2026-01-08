import os
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

HF_API_KEY = os.getenv("HF_API_KEY")

MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
client = InferenceClient(token=HF_API_KEY)

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


class HFEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embed_texts(texts)

    def embed_query(self, text):
        return embed_texts([text])[0]


def get_embeddings():
    return HFEmbeddings()
