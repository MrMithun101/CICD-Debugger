from chromadb import EmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def get_embedding_function() -> EmbeddingFunction:
    """Return the embedding function used for all ChromaDB collections.

    Uses all-MiniLM-L6-v2 (384 dims, ~22MB, CPU-friendly).
    To swap models, change only this function — nothing else in the codebase
    needs to know which model is in use.
    """
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
