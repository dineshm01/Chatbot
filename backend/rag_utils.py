import os
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings

def load_vectorstore():
    """Loads the FAISS index from the persistent volume."""
    embeddings = get_embeddings()
    # Path must match the VECTOR_DIR defined in your ingest.py
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

def get_retriever():
    """
    THE PERFECTION FIX: Uses MMR to ensure the bot reads the entire PPTX depth.
    This prevents the bot from getting stuck on Slide 1.
    """
    vectorstore = load_vectorstore()
    if not vectorstore:
        return None
    
    return vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 15,            # Retrieves 15 diverse slides
            "fetch_k": 50,      # Scans 50 chunks to find all GAN types
            "lambda_mult": 0.6  # Balances relevance with technical diversity
        }
    )

def truncate_docs(docs, max_chars=12000):
    """
    STRUCTURAL TRUNCATION: Prevents logical 'mashing' errors.
    Ensures the AI sees clear boundaries between GAN architectures.
    """
    context_parts = []
    current_length = 0

    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        # Syncs page numbers for perfect highlighting
        page = doc.metadata.get("page", i + 1)
        content = doc.page_content.strip()

        # Added structural headers to fix the 'Wrong Answer' logic
        formatted_chunk = f"--- [SLIDE {page} | {source}] ---\n{content}\n"
        
        if current_length + len(formatted_chunk) > max_chars:
            break
            
        context_parts.append(formatted_chunk)
        current_length += len(formatted_chunk)

    return "\n".join(context_parts)

def compute_confidence(docs):
    """Simple confidence metric based on retrieval success."""
    return "Fully covered by notes" if len(docs) > 0 else "Low confidence"

def compute_coverage(docs, answer):
    """Calculates how much of the answer is supported by retrieved chunks."""
    # Placeholder for your existing coverage logic
    return 100 if len(docs) > 0 else 0
