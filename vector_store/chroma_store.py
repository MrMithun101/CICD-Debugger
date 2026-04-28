import logging
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction

from ingestion.models import ProcessedChunk
from vector_store.embedder import get_embedding_function

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ci_failures"


class ChromaStore:
    def __init__(self, db_path: Path, embedding_function: EmbeddingFunction | None = None):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function or get_embedding_function(),
            # Cosine similarity measures direction (meaning), not magnitude (length).
            # Better than L2 for variable-length text chunks.
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[ProcessedChunk]) -> None:
        """Add or update chunks in the collection. Safe to call repeatedly — idempotent."""
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "repo": c.repo,
                    "workflow_name": c.workflow_name,
                    "run_id": c.run_id,
                    "failure_type": c.failure_type,
                    "source_step": c.source_step,
                    "created_at": c.created_at,
                    "html_url": c.html_url,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )
        logger.debug("Upserted %d chunks", len(chunks))

    def query(
        self,
        text: str,
        n_results: int = 5,
        failure_type: str | None = None,
    ) -> list[dict]:
        """Return the n most similar chunks to the query text.

        Args:
            text: The query — typically a snippet from a new failing log.
            n_results: How many similar chunks to return.
            failure_type: If set, restrict results to this failure category.

        Returns:
            List of dicts, each with chunk fields + "distance" (0 = identical, 1 = opposite).
        """
        total = self.collection.count()
        if total == 0:
            logger.warning("Vector store is empty — run scripts/embed.py first")
            return []

        # Clamp n_results to available documents to avoid ChromaDB errors.
        effective_n = min(n_results, total)
        where = {"failure_type": failure_type} if failure_type else None

        results = self.collection.query(
            query_texts=[text],
            n_results=effective_n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists (outer dimension = number of queries).
        # We always send one query, so we index [0] to flatten.
        return [
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                **results["metadatas"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self.collection.count()
