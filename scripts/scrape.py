"""
Run the GitHub log scraper against all target repos.

Usage:
    python scripts/scrape.py
    python scripts/scrape.py --max-runs 3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.github_client import GitHubLogScraper

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPOS = [
    "facebook/react",
    "microsoft/vscode",
    "django/django",
    "pallets/flask",
    "psf/requests",
    "numpy/numpy",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
]


def main(max_runs: int) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    data_dir = Path(__file__).parent.parent / "data"
    scraper = GitHubLogScraper(token=token, data_dir=data_dir)

    total_saved = 0
    for repo in REPOS:
        logger.info("Scraping %s ...", repo)
        try:
            saved = scraper.scrape_repo(repo, max_runs=max_runs)
            total_saved += len(saved)
            logger.info("  → %d/%d runs saved for %s", len(saved), max_runs, repo)
        except Exception as e:
            logger.warning("  → Failed to scrape %s: %s", repo, e)

    logger.info("Done. Total logs saved: %d", total_saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape GitHub Actions failure logs")
    parser.add_argument("--max-runs", type=int, default=5, help="Runs to fetch per repo")
    args = parser.parse_args()
    main(args.max_runs)
