import io
import json
import logging
import time
import zipfile
from pathlib import Path

import requests
from github import Github

from scraper.models import WorkflowRun

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubLogScraper:
    def __init__(self, token: str, data_dir: Path):
        self.github = Github(token)
        self.token = token
        self.raw_logs_dir = Path(data_dir) / "raw_logs"
        self.raw_logs_dir.mkdir(parents=True, exist_ok=True)

    def get_failed_runs(self, repo_name: str, max_runs: int = 5) -> list[WorkflowRun]:
        """Fetch the most recent failed workflow runs for a repo."""
        repo = self.github.get_repo(repo_name)
        runs: list[WorkflowRun] = []

        # GitHub filters by status (completed/in_progress), not conclusion.
        # We must fetch completed runs and filter for failure ourselves.
        for run in repo.get_workflow_runs(status="completed"):
            if len(runs) >= max_runs:
                break
            if run.conclusion != "failure":
                continue

            runs.append(WorkflowRun(
                repo=repo_name,
                workflow_name=run.name or "unknown",
                run_id=run.id,
                status=run.status,
                conclusion=run.conclusion,
                created_at=run.created_at.isoformat(),
                html_url=run.html_url,
            ))
            time.sleep(0.1)

        logger.info("Found %d failed runs in %s", len(runs), repo_name)
        return runs

    def download_log(self, run: WorkflowRun) -> str | None:
        """Download the log zip for a run and return its text content.

        GitHub returns a 302 redirect to a presigned S3 zip. We follow the
        redirect, extract every .txt inside, and concatenate them in order.
        Returns None if the log has expired (410) or the request fails.
        """
        url = f"{GITHUB_API}/repos/{run.repo}/actions/runs/{run.run_id}/logs"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        try:
            response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)

            if response.status_code == 410:
                logger.warning("Logs expired (410) for run %d in %s", run.run_id, run.repo)
                return None

            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                parts: list[str] = []
                for name in sorted(zf.namelist()):
                    if name.endswith(".txt"):
                        with zf.open(name) as f:
                            parts.append(f"=== {name} ===")
                            parts.append(f.read().decode("utf-8", errors="replace"))

            return "\n".join(parts)

        except requests.RequestException as e:
            logger.warning("Request failed for run %d: %s", run.run_id, e)
            return None
        except zipfile.BadZipFile:
            logger.warning("Bad zip for run %d in %s", run.run_id, run.repo)
            return None

    def save_run(self, run: WorkflowRun, log_content: str) -> Path:
        """Persist log text and JSON metadata to data/raw_logs/."""
        # Slashes in repo names break file paths — replace with double underscore.
        safe_name = run.repo.replace("/", "__")
        stem = f"{safe_name}__{run.run_id}"

        log_path = self.raw_logs_dir / f"{stem}.txt"
        meta_path = self.raw_logs_dir / f"{stem}.json"

        log_path.write_text(log_content, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "repo": run.repo,
                    "workflow_name": run.workflow_name,
                    "run_id": run.run_id,
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "created_at": run.created_at,
                    "html_url": run.html_url,
                    "log_path": str(log_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return log_path

    def scrape_repo(self, repo_name: str, max_runs: int = 5) -> list[WorkflowRun]:
        """End-to-end: fetch failed runs, download logs, save to disk."""
        runs = self.get_failed_runs(repo_name, max_runs)
        saved: list[WorkflowRun] = []

        for run in runs:
            log_content = self.download_log(run)
            if log_content:
                log_path = self.save_run(run, log_content)
                run.log_path = str(log_path)
                saved.append(run)
                logger.info("Saved run %d from %s → %s", run.run_id, run.repo, log_path.name)

            time.sleep(0.5)

        return saved
