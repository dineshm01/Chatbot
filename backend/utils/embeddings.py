from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

e = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(e.embed_query("hello world"))
