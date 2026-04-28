"""
Tests for the ingestion pipeline — all offline, no filesystem side-effects
except where a tmp_path fixture is used.
"""

import json
from pathlib import Path

import pytest

from ingestion.cleaner import clean_log
from ingestion.chunker import chunk_log, MAX_CHUNK_CHARS
from ingestion.classifier import classify_failure
from ingestion.pipeline import process_log


# ---------------------------------------------------------------------------
# cleaner
# ---------------------------------------------------------------------------

class TestCleaner:
    def test_strips_timestamps(self):
        raw = "2026-04-25T09:20:37.2252504Z Error: something failed"
        assert "2026-04-25" not in clean_log(raw)
        assert "Error: something failed" in clean_log(raw)

    def test_strips_ansi_codes(self):
        raw = "\x1b[31mERROR\x1b[0m: build failed"
        result = clean_log(raw)
        assert "\x1b" not in result
        assert "ERROR" in result
        assert "build failed" in result

    def test_strips_gha_group_commands(self):
        raw = "##[group]Setting up Python\nsome content\n##[endgroup]"
        result = clean_log(raw)
        assert "##[group]" not in result
        assert "##[endgroup]" not in result
        assert "some content" in result

    def test_strips_bom(self):
        raw = "\ufeffFirst line of log"
        assert clean_log(raw).startswith("First line")

    def test_collapses_excess_blank_lines(self):
        raw = "line1\n\n\n\n\nline2"
        result = clean_log(raw)
        assert "\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_empty_input_returns_empty(self):
        assert clean_log("") == ""

    def test_preserves_actual_error_text(self):
        raw = "2026-01-01T00:00:00.000Z FAILED tests/test_app.py::test_login - AssertionError"
        result = clean_log(raw)
        assert "FAILED tests/test_app.py::test_login" in result
        assert "AssertionError" in result


# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------

class TestChunker:
    def _log(self, sections: dict[str, str]) -> str:
        parts = []
        for name, body in sections.items():
            parts.append(f"=== {name} ===\n{body}")
        return "\n".join(parts)

    def test_splits_into_sections(self):
        text = self._log({
            "job/1_setup.txt": "Setting up Python 3.11",
            "job/2_tests.txt": "FAILED: 3 tests failed",
        })
        chunks = chunk_log(text)
        assert len(chunks) == 2
        assert chunks[0]["step"] == "job/1_setup.txt"
        assert chunks[1]["step"] == "job/2_tests.txt"

    def test_skips_empty_sections(self):
        text = "=== job/1_empty.txt ===\n   \n=== job/2_real.txt ===\nsome error"
        chunks = chunk_log(text)
        assert len(chunks) == 1
        assert chunks[0]["step"] == "job/2_real.txt"

    def test_text_without_headers_returns_empty(self):
        assert chunk_log("no headers here at all") == []

    def test_oversized_section_is_split(self):
        # 8 paragraphs of 1000 chars each = ~8000 chars, well over MAX_CHUNK_CHARS (6000).
        # Each paragraph is large enough that the accumulator trips the limit mid-way.
        para = "error: " + "x" * 993   # exactly 1000 chars
        big_body = "\n\n".join([para] * 8)
        assert len(big_body) > MAX_CHUNK_CHARS
        text = f"=== job/big.txt ===\n{big_body}"
        chunks = chunk_log(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c["text"]) <= MAX_CHUNK_CHARS

    def test_chunk_text_preserves_content(self):
        text = self._log({"job/1.txt": "error: module not found\npip failed"})
        chunks = chunk_log(text)
        assert "error: module not found" in chunks[0]["text"]
        assert "pip failed" in chunks[0]["text"]


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_detects_test_failure(self):
        assert classify_failure("FAILED tests/test_views.py::test_home - AssertionError") == "test_failure"

    def test_detects_build_error(self):
        assert classify_failure("error TS2304: Cannot find name 'React'") == "build_error"

    def test_detects_dependency_error(self):
        assert classify_failure("ERROR: Could not find a version that satisfies the requirement numpy==99.0") == "dependency_error"

    def test_detects_lint_error(self):
        assert classify_failure("ruff: Found 12 errors.") == "lint_error"

    def test_detects_timeout(self):
        assert classify_failure("The runner has received a shutdown signal") == "timeout"

    def test_detects_oom(self):
        assert classify_failure("MemoryError: Unable to allocate array") == "oom"

    def test_unknown_fallback(self):
        assert classify_failure("Everything seems fine but exited anyway") == "unknown"

    def test_oom_takes_priority_over_test_failure(self):
        # OOM is listed before test_failure in _PATTERNS — must win
        mixed = "FAILED test_foo.py - MemoryError"
        assert classify_failure(mixed) == "oom"

    def test_case_insensitive(self):
        assert classify_failure("MEMORYERROR in worker") == "oom"


# ---------------------------------------------------------------------------
# pipeline (integration)
# ---------------------------------------------------------------------------

class TestPipeline:
    def _meta(self, run_id: int = 42) -> dict:
        return {
            "repo": "owner/repo",
            "workflow_name": "CI",
            "run_id": run_id,
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2024-06-01T12:00:00",
            "html_url": f"https://github.com/owner/repo/actions/runs/{run_id}",
            "log_path": f"data/raw_logs/owner__repo__{run_id}.txt",
        }

    def test_produces_correct_chunk_ids(self, tmp_path: Path):
        raw = (
            "2026-01-01T00:00:00.000Z ==> setup\n"
            "=== job/1_setup.txt ===\n"
            "Setting up env\n"
            "=== job/2_test.txt ===\n"
            "FAILED tests/test_foo.py - AssertionError\n"
        )
        result = process_log(raw, self._meta(99), tmp_path)
        ids = [c.chunk_id for c in result.chunks]
        assert "owner__repo__99__chunk_0" in ids
        assert "owner__repo__99__chunk_1" in ids

    def test_failure_type_propagates_to_all_chunks(self, tmp_path: Path):
        raw = (
            "=== job/1.txt ===\nsome setup\n"
            "=== job/2.txt ===\nFAILED test_foo.py - AssertionError\n"
        )
        result = process_log(raw, self._meta(1), tmp_path)
        assert result.failure_type == "test_failure"
        assert all(c.failure_type == "test_failure" for c in result.chunks)

    def test_saves_json_to_disk(self, tmp_path: Path):
        raw = "=== job/1.txt ===\nsome content\n"
        process_log(raw, self._meta(55), tmp_path)
        out = tmp_path / "owner__repo__55.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["run_id"] == 55
        assert len(data["chunks"]) == 1

    def test_empty_log_produces_no_chunks(self, tmp_path: Path):
        result = process_log("", self._meta(0), tmp_path)
        assert result.chunks == []
