import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class SafeHFEmbeddings:
    def __init__(self):
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise RuntimeError("HF_API_KEY is missing")

        self.client = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return [self.client.embed_query(t) for t in texts]

    def embed_query(self, text):
        return self.client.embed_query(text)

def get_embeddings():
    return SafeHFEmbeddings()
