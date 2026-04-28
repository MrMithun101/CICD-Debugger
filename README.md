# cicd-debugger

An AI-powered CI/CD failure triage agent. Give it a failing GitHub Actions log — it classifies the failure, retrieves similar historical failures from a vector store, and uses Cerebras inference to generate a root cause hypothesis, confidence score, and actionable fix.

Exposed as a native Claude Code tool via MCP so you can triage failures without leaving your editor.

---

## Architecture

```
GitHub API
    │
    ▼
┌───────────────┐  raw .txt + .json   ┌──────────────────┐
│    Scraper    │ ──────────────────► │ Ingestion Pipeline│
│ (PyGithub +  │                     │  clean → chunk →  │
│  requests)   │                     │  classify         │
└───────────────┘                     └──────────────────┘
                                               │
                                       processed chunks
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │    ChromaDB    │
                                      │  (local, HNSW  │
                                      │  cosine, 384d) │
                                      └────────────────┘
                                               │
                          ┌────────────────────┘
                          │  retrieve similar failures
                          ▼
new log ──────────► ┌──────────────┐  Cerebras API   ┌──────────────┐
                    │ Triage Agent │ ──────────────► │ TriageReport │
                    │              │  (llama3.1-8b,  │  JSON output │
                    └──────────────┘   JSON mode)    └──────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  MCP Server │ ◄──── Claude Code calls triage_cicd_failure()
                   │  (FastMCP,  │
                   │   stdio)    │
                   └─────────────┘
```

**Data flow:**
1. **Scraper** pulls failed workflow runs from GitHub, downloads log zips, saves `.txt` + `.json` sidecar pairs to `data/raw_logs/`
2. **Ingestion pipeline** cleans (ANSI, timestamps, GHA markers), chunks by step boundary, and labels each run with a failure type
3. **Vector store** embeds chunks with `all-MiniLM-L6-v2` and stores them in a persistent ChromaDB collection
4. **Triage agent** cleans a new log → classifies → retrieves similar historical failures → calls Cerebras → returns a structured report
5. **MCP server** wraps the agent as a Claude Code tool over stdio transport

---

## Tech Stack

| Layer | Technology |
|---|---|
| Log collection | GitHub Actions API via PyGithub |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384d, local) |
| Vector store | ChromaDB (persistent, HNSW cosine index) |
| LLM inference | Cerebras Cloud SDK (`llama3.1-8b`, JSON mode) |
| Tool interface | FastMCP (stdio transport) |
| Testing | pytest + pytest-mock (68 tests, all offline) |

---

## Setup

### 1. Clone and create virtualenv

```bash
git clone https://github.com/MrMithun101/cicd-debugger
cd cicd-debugger
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
#   GITHUB_TOKEN   — https://github.com/settings/tokens (read:repo scope)
#   CEREBRAS_API_KEY — https://cloud.cerebras.ai
```

### 3. Scrape logs, ingest, and embed

```bash
# Step 1: Download failed workflow logs from GitHub (40–80 logs, ~2 min)
python scripts/scrape.py --max-runs 5

# Step 2: Clean, chunk, and classify logs
python scripts/ingest.py

# Step 3: Embed chunks into ChromaDB (downloads model on first run, ~30s)
python scripts/embed.py
```

---

## Usage

### Demo script (terminal)

```bash
python scripts/demo.py data/raw_logs/<any_log_file>.txt
```

**Example output:**
```
============================================================
  CI/CD Failure Triage Report
  Log: pallets__flask__24927656835.txt
============================================================

Failure Type : test_failure
Confidence   : 80%

Root Cause:
  The test suite failed due to a pre-commit hook violation — the ruff
  linter found formatting issues in the changed files.

Suggested Fix:
  Run `ruff format .` locally before pushing, or add a pre-commit
  hook to your local environment with `pre-commit install`.

Similar Historical Cases (3):
  • pallets/flask  type=test_failure  distance=0.000
  • pallets/flask  type=dependency_error  distance=0.480
```

### As a Claude Code tool (MCP)

The `.mcp.json` at the project root registers the server automatically. Open this directory in Claude Code, then ask:

> "Use the triage tool to analyze this log: [paste log text]"

Or from the Claude Code CLI:

```bash
# Verify the tool is registered
claude mcp list

# Claude can now call it natively:
# "Triage this failing CI log for me" → Claude calls triage_cicd_failure()
```

To register manually:
```bash
claude mcp add --scope project cicd-triage python mcp_server/server.py
```

---

## Running Tests

```bash
# All tests — no API keys needed, fully mocked
python -m pytest tests/ -v

# Individual modules
python -m pytest tests/test_scraper.py
python -m pytest tests/test_ingestion.py
python -m pytest tests/test_vector_store.py
python -m pytest tests/test_agent.py
```

68 tests across all components. External APIs (GitHub, Cerebras, ChromaDB) are mocked — tests run offline.

---

## Project Structure

```
cicd-debugger/
│
├── scraper/               # GitHub API client + WorkflowRun model
├── ingestion/             # cleaner, chunker, classifier, pipeline
├── vector_store/          # ChromaDB wrapper + embedding function
├── agent/                 # TriageAgent, TriageReport, prompt templates
├── mcp_server/            # FastMCP server (Claude Code tool)
│
├── scripts/
│   ├── scrape.py          # Fetch logs from GitHub
│   ├── ingest.py          # Run ingestion pipeline
│   ├── embed.py           # Populate ChromaDB
│   └── demo.py            # Terminal demo
│
├── data/
│   ├── raw_logs/          # Downloaded .txt + .json sidecar files
│   ├── processed/         # Cleaned chunk JSON files
│   └── chroma_db/         # Persistent vector store (gitignored)
│
├── tests/                 # pytest test suite (68 tests)
├── .mcp.json              # Project-scope MCP registration
└── .env.example           # Environment variable template
```

---

## Target Repos

40 failed workflow runs scraped from 8 active open-source projects, producing **3,829 embedded chunks**:

| Repo | Language | Why included |
|---|---|---|
| `pallets/flask` | Python | Diverse test/lint failures |
| `psf/requests` | Python | Dependency and compatibility issues |
| `django/django` | Python | Large test suite, matrix builds |
| `numpy/numpy` | Python/C | Build errors, OOM failures |
| `facebook/react` | JavaScript | Jest failures, TypeScript errors |
| `microsoft/vscode` | TypeScript | Build complexity, timeout failures |
| `pytorch/pytorch` | Python/C++ | CUDA OOM, flaky tests |
| `tensorflow/tensorflow` | Python/C++ | Complex dependency errors |

---

## Limitations & Future Improvements

**Current limitations:**
- Heuristic classifier covers ~6 failure categories — exotic or custom CI frameworks fall through to `unknown`
- `all-MiniLM-L6-v2` embeddings are general-purpose; a code/log-specific model would improve retrieval quality
- The agent analyzes one log at a time; batch triage across a PR's matrix builds is not yet supported

**What production would add:**
- Webhook receiver to trigger triage automatically on workflow failure
- Per-repo feedback loop: thumbs up/down on triage quality improves retrieval over time
- Exponential backoff + retry on Cerebras 429s
- Streaming MCP responses for long triage reports
- `text-embedding-3-small` benchmarked against `all-MiniLM-L6-v2` for retrieval quality

---

## Resume Bullets

```
• Built an end-to-end AI CI/CD triage agent using Cerebras LLM inference and ChromaDB
  vector search — scrapes GitHub Actions logs, classifies failure types, retrieves
  similar historical failures via RAG, and generates root cause analysis with confidence
  scores; exposed as a native Claude Code tool via FastMCP (stdio transport)

• Designed a modular ingestion pipeline (log cleaning, structural chunking, heuristic
  classification) that processes logs from 8 open-source repos into 384-dim embeddings
  stored in a persistent HNSW cosine index; 68 automated tests with zero live API
  dependencies

• Implemented an MCP server using FastMCP that registers the triage agent as a project-
  scoped Claude Code tool — demonstrating end-to-end AI tooling integration from data
  collection through LLM inference to IDE-native tool invocation
```

---

## Interview Explanation (30-second version)

> "I built a CI/CD failure triage agent. It scrapes GitHub Actions logs, cleans them and splits them into chunks by step boundary, embeds them with a sentence-transformer model and stores them in ChromaDB. When a new log comes in, the agent classifies the failure type, retrieves the 3 most similar historical failures from the vector store, and passes both to Cerebras to generate a root cause and fix suggestion. The whole thing is exposed as a Claude Code tool via an MCP server, so you can triage failures without leaving your editor. The interesting engineering decisions were around chunking strategy — failures always appear near the end of logs so I reverse-iterate to find the most relevant chunk — and using JSON mode on the Cerebras API to get reliable structured output."
