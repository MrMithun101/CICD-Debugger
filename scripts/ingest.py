"""
Run the ingestion pipeline over all scraped raw logs.

Usage:
    python scripts/ingest.py
"""

import logging
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    results = run_pipeline(data_dir)

    if not results:
        logger.warning("No logs processed.")
        return

    type_counts = Counter(r.failure_type for r in results)
    total_chunks = sum(len(r.chunks) for r in results)

    logger.info("--- Ingestion Summary ---")
    logger.info("Runs processed : %d", len(results))
    logger.info("Total chunks   : %d", total_chunks)
    logger.info("Failure types  : %s", dict(type_counts))


if __name__ == "__main__":
    main()
