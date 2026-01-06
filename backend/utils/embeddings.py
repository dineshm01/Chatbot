from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
