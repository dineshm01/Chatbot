from langchain_community.embeddings import HuggingFaceEndpointEmbeddings

def get_embeddings():
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise RuntimeError("HF_API_KEY is missing")

    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=api_key,
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )
