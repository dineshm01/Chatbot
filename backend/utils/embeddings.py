import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class WrappedHFEmbeddings:
    def __init__(self):
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise RuntimeError("HF_API_KEY environment variable is missing")

        self.client = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        result = self.client.embed_documents(texts)

        # HF sometimes returns {"embeddings": [...]}
        if isinstance(result, dict):
            result = result.get("embeddings", [])

        return result

    def embed_query(self, text):
        result = self.client.embed_query(text)

        if isinstance(result, dict):
            result = result.get("embedding", result.get("embeddings", []))

        return result


def get_embeddings():
    return WrappedHFEmbeddings()
