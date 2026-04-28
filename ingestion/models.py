from dataclasses import dataclass, field


@dataclass
class ProcessedChunk:
    chunk_id: str        # "{repo_safe}__{run_id}__chunk_{index}"
    chunk_index: int
    source_step: str     # step filename from the zip, e.g. "job1/2_run_tests.txt"
    text: str
    # run-level metadata — repeated per chunk so each is self-contained for ChromaDB
    repo: str
    workflow_name: str
    run_id: int
    failure_type: str
    created_at: str
    html_url: str


@dataclass
class ProcessedLog:
    repo: str
    workflow_name: str
    run_id: int
    failure_type: str
    created_at: str
    html_url: str
    chunks: list[ProcessedChunk] = field(default_factory=list)
