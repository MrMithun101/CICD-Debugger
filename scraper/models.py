from dataclasses import dataclass, field


@dataclass
class WorkflowRun:
    repo: str           # "owner/name" e.g. "facebook/react"
    workflow_name: str
    run_id: int
    status: str         # always "completed" for failed runs
    conclusion: str     # "failure", "timed_out", etc.
    created_at: str     # ISO 8601 string — easy to serialize/log
    html_url: str
    log_path: str | None = field(default=None)  # set after download
