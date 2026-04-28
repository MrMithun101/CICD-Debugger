"""
Tests for the GitHub log scraper.

All tests mock external dependencies (GitHub API, HTTP requests) so they
run offline without consuming API quota.
"""

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scraper.github_client import GitHubLogScraper
from scraper.models import WorkflowRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zip(files: dict[str, str]) -> bytes:
    """Build an in-memory zip with the given filename → content mapping."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _make_mock_run(run_id: int = 12345, conclusion: str = "failure") -> MagicMock:
    """Return a PyGithub WorkflowRun-shaped mock."""
    mock = MagicMock()
    mock.id = run_id
    mock.name = "CI"
    mock.status = "completed"
    mock.conclusion = conclusion
    mock.created_at = datetime(2024, 6, 1, 12, 0, 0)
    mock.html_url = f"https://github.com/test/repo/actions/runs/{run_id}"
    return mock


@pytest.fixture
def scraper(tmp_path: Path) -> GitHubLogScraper:
    """Scraper with a real temp directory but a mocked GitHub client."""
    with patch("scraper.github_client.Github"):
        return GitHubLogScraper(token="fake-token", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# get_failed_runs
# ---------------------------------------------------------------------------

class TestGetFailedRuns:
    def test_returns_only_failure_conclusions(self, scraper: GitHubLogScraper):
        """Non-failure conclusions (cancelled, success) must be excluded."""
        mock_runs = [
            _make_mock_run(1, "failure"),
            _make_mock_run(2, "cancelled"),   # should be skipped
            _make_mock_run(3, "failure"),
        ]
        scraper.github.get_repo.return_value.get_workflow_runs.return_value = mock_runs

        result = scraper.get_failed_runs("test/repo", max_runs=10)

        assert len(result) == 2
        assert all(r.conclusion == "failure" for r in result)

    def test_respects_max_runs_limit(self, scraper: GitHubLogScraper):
        mock_runs = [_make_mock_run(i, "failure") for i in range(20)]
        scraper.github.get_repo.return_value.get_workflow_runs.return_value = mock_runs

        result = scraper.get_failed_runs("test/repo", max_runs=3)

        assert len(result) == 3

    def test_maps_fields_correctly(self, scraper: GitHubLogScraper):
        """WorkflowRun fields must match the PyGithub object attributes."""
        scraper.github.get_repo.return_value.get_workflow_runs.return_value = [
            _make_mock_run(99)
        ]

        result = scraper.get_failed_runs("owner/repo", max_runs=1)
        run = result[0]

        assert run.repo == "owner/repo"
        assert run.run_id == 99
        assert run.workflow_name == "CI"
        assert run.conclusion == "failure"
        assert run.log_path is None   # not set until download


# ---------------------------------------------------------------------------
# download_log
# ---------------------------------------------------------------------------

class TestDownloadLog:
    def _run(self, run_id: int = 42) -> WorkflowRun:
        return WorkflowRun(
            repo="test/repo",
            workflow_name="CI",
            run_id=run_id,
            status="completed",
            conclusion="failure",
            created_at="2024-06-01T12:00:00",
            html_url="https://github.com/test/repo/actions/runs/42",
        )

    def test_extracts_text_from_zip(self, scraper: GitHubLogScraper):
        zip_bytes = _make_zip({
            "job1/1_checkout.txt": "Cloning repo...",
            "job1/2_test.txt":     "FAILED: 3 tests failed",
        })
        mock_resp = MagicMock(status_code=200, content=zip_bytes)
        mock_resp.raise_for_status = MagicMock()

        with patch("scraper.github_client.requests.get", return_value=mock_resp):
            result = scraper.download_log(self._run())

        assert result is not None
        assert "Cloning repo..." in result
        assert "FAILED: 3 tests failed" in result

    def test_returns_none_on_410(self, scraper: GitHubLogScraper):
        """410 means logs have expired — should return None, not raise."""
        mock_resp = MagicMock(status_code=410)
        with patch("scraper.github_client.requests.get", return_value=mock_resp):
            result = scraper.download_log(self._run())
        assert result is None

    def test_returns_none_on_bad_zip(self, scraper: GitHubLogScraper):
        mock_resp = MagicMock(status_code=200, content=b"not a zip")
        mock_resp.raise_for_status = MagicMock()
        with patch("scraper.github_client.requests.get", return_value=mock_resp):
            result = scraper.download_log(self._run())
        assert result is None

    def test_returns_none_on_request_error(self, scraper: GitHubLogScraper):
        import requests as req
        with patch("scraper.github_client.requests.get", side_effect=req.RequestException("timeout")):
            result = scraper.download_log(self._run())
        assert result is None

    def test_log_files_sorted_by_name(self, scraper: GitHubLogScraper):
        """Steps must appear in filename-sorted order, not zip-insertion order."""
        zip_bytes = _make_zip({
            "job1/2_run_tests.txt": "step 2",
            "job1/1_setup.txt":     "step 1",
        })
        mock_resp = MagicMock(status_code=200, content=zip_bytes)
        mock_resp.raise_for_status = MagicMock()

        with patch("scraper.github_client.requests.get", return_value=mock_resp):
            result = scraper.download_log(self._run())

        assert result.index("step 1") < result.index("step 2")


# ---------------------------------------------------------------------------
# save_run
# ---------------------------------------------------------------------------

class TestSaveRun:
    def test_creates_txt_and_json_files(self, scraper: GitHubLogScraper, tmp_path: Path):
        run = WorkflowRun(
            repo="facebook/react",
            workflow_name="CI",
            run_id=777,
            status="completed",
            conclusion="failure",
            created_at="2024-06-01T00:00:00",
            html_url="https://github.com/facebook/react/actions/runs/777",
        )
        log_path = scraper.save_run(run, "build failed: module not found")

        assert log_path.exists()
        assert log_path.read_text() == "build failed: module not found"

        meta_path = log_path.with_suffix(".json")
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == 777
        assert meta["repo"] == "facebook/react"
        assert meta["log_path"] == str(log_path)

    def test_repo_slash_replaced_in_filename(self, scraper: GitHubLogScraper):
        run = WorkflowRun(
            repo="owner/repo-name",
            workflow_name="CI",
            run_id=1,
            status="completed",
            conclusion="failure",
            created_at="2024-01-01T00:00:00",
            html_url="http://example.com",
        )
        log_path = scraper.save_run(run, "log")
        assert "/" not in log_path.name
        assert "owner__repo-name" in log_path.name
