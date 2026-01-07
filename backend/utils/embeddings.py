import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class SafeHFEmbeddings:
    def __init__(self):
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise RuntimeError("HF_API_KEY is missing")

        self.client = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction"
        )

    def embed_documents(self, texts):
        vectors = []
        for t in texts:
            v = self.client.embed_query(t)
            if v and len(v) > 10:
                vectors.append(v)
            else:
                raise RuntimeError("Empty embedding returned from HF API")
        return vectors

    def embed_query(self, text):
        v = self.client.embed_query(text)
        if not v or len(v) < 10:
            raise RuntimeError("Empty embedding returned from HF API")
        return v

def get_embeddings():
    return SafeHFEmbeddings()
