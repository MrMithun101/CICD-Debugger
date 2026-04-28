"""
End-to-end demo: triage a failing CI/CD log from a file.

Usage:
    python scripts/demo.py <path_to_log_file>
    python scripts/demo.py data/raw_logs/pallets__flask__24927656835.txt
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()
logging.basicConfig(level=logging.WARNING)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo.py <log_file>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: {log_path} does not exist")
        sys.exit(1)

    from vector_store.chroma_store import ChromaStore
    from agent.triage_agent import TriageAgent

    data_dir = Path(__file__).parent.parent / "data"
    store = ChromaStore(db_path=data_dir / "chroma_db")
    agent = TriageAgent(store=store, cerebras_api_key=os.environ["CEREBRAS_API_KEY"])

    log_text = log_path.read_text(encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("  CI/CD Failure Triage Report")
    print(f"  Log: {log_path.name}")
    print(f"{'=' * 60}\n")

    report = agent.triage(log_text)

    print(f"Failure Type : {report.failure_type}")
    print(f"Confidence   : {report.confidence:.0%}")
    print(f"\nRoot Cause:\n  {report.root_cause}")
    print(f"\nSuggested Fix:\n  {report.suggested_fix}")
    print(f"\nExplanation:\n  {report.explanation}")

    if report.similar_cases:
        print(f"\nSimilar Historical Cases ({len(report.similar_cases)}):")
        for c in report.similar_cases:
            print(f"  • {c['repo']}  type={c['failure_type']}  distance={c['distance']:.3f}")
            print(f"    {c['html_url']}")

    print()


if __name__ == "__main__":
    main()
