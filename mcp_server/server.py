"""
MCP server that exposes the CI/CD triage agent as a Claude Code tool.

Transport: stdio (Claude Code launches this as a subprocess)
Protocol:  MCP via FastMCP

To register with Claude Code (run once from the project root):
    claude mcp add --scope project cicd-triage \
        python mcp_server/server.py

Or edit .mcp.json at the project root (already included in this repo).

IMPORTANT: all logging must go to stderr — stdout is the MCP protocol stream.
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Allow imports from the project root when launched as a subprocess.
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time initialization at server startup (not per-call).
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"

logger.info("Initializing vector store...")
from vector_store.chroma_store import ChromaStore
_store = ChromaStore(db_path=_DATA_DIR / "chroma_db")
logger.info("Vector store ready — %d chunks indexed", _store.count())

logger.info("Initializing triage agent...")
from agent.triage_agent import TriageAgent
_agent = TriageAgent(
    store=_store,
    cerebras_api_key=os.environ.get("CEREBRAS_API_KEY"),
)
logger.info("Triage agent ready.")

# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

mcp = FastMCP("cicd-triage")


@mcp.tool()
def triage_cicd_failure(log_text: str, n_similar: int = 3) -> str:
    """Analyze a failing CI/CD log and return a structured triage report.

    Cleans the log, classifies the failure type, retrieves similar historical
    failures from the vector store, and uses Cerebras AI to generate a root
    cause hypothesis, confidence score, and suggested fix.

    Args:
        log_text: Raw text of a failing GitHub Actions workflow log.
        n_similar: Number of similar historical failures to retrieve (default 3).

    Returns:
        JSON string with keys: failure_type, root_cause, confidence (0–1),
        suggested_fix, explanation, similar_cases, log_preview.
    """
    try:
        report = _agent.triage(log_text, n_similar=n_similar)
        return json.dumps(report.to_dict(), indent=2)
    except Exception as e:
        logger.error("Triage failed: %s", e)
        return json.dumps({"error": str(e), "failure_type": "unknown", "confidence": 0.0})


if __name__ == "__main__":
    mcp.run()
