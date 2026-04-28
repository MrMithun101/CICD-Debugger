"""
Embed all processed log chunks into ChromaDB.

Reads every JSON in data/processed/, loads the chunks, and upserts them into
the vector store. Safe to re-run — upsert is idempotent.

Usage:
    python scripts/embed.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.models import ProcessedChunk
from vector_store.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    processed_dir = data_dir / "processed"
    db_path = data_dir / "chroma_db"

    processed_files = list(processed_dir.glob("*.json"))
    if not processed_files:
        logger.error("No processed files found. Run scripts/ingest.py first.")
        sys.exit(1)

    logger.info("Connecting to ChromaDB at %s", db_path)
    store = ChromaStore(db_path=db_path)
    logger.info("Collection currently holds %d chunks", store.count())

    total_upserted = 0
    for path in processed_files:
        data = json.loads(path.read_text(encoding="utf-8"))

        chunks = [
            ProcessedChunk(
                chunk_id=c["chunk_id"],
                chunk_index=c["chunk_index"],
                source_step=c["source_step"],
                text=c["text"],
                repo=data["repo"],
                workflow_name=data["workflow_name"],
                run_id=data["run_id"],
                failure_type=data["failure_type"],
                created_at=data["created_at"],
                html_url=data["html_url"],
            )
            for c in data["chunks"]
        ]

        store.upsert(chunks)
        total_upserted += len(chunks)
        logger.info("  %s → %d chunks", path.name, len(chunks))

    logger.info("Done. Total chunks in store: %d", store.count())


if __name__ == "__main__":
    main()
