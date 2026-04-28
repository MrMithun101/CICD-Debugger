"""
Tests for the triage agent — all mocked, no live Cerebras or ChromaDB calls.
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from agent.triage_agent import TriageAgent, TriageReport
from agent.prompts import build_system_prompt, build_user_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_LOG = """=== job/1_setup.txt ===
Setting up Python 3.11

=== job/2_tests.txt ===
FAILED tests/test_views.py::test_index - AssertionError: expected 200, got 500
Process completed with exit code 1
"""

_SIMILAR_CASE = {
    "chunk_id": "pallets__flask__1__0",
    "text": "FAILED tests/test_app.py::test_home - AssertionError",
    "repo": "pallets/flask",
    "failure_type": "test_failure",
    "distance": 0.22,
    "html_url": "https://github.com/pallets/flask/actions/runs/1",
    "workflow_name": "CI",
    "run_id": 1,
    "source_step": "job/2_test.txt",
    "created_at": "2024-06-01",
    "chunk_index": 0,
}

_LLM_RESPONSE = {
    "root_cause": "The view function raises an unhandled exception causing a 500 response",
    "confidence": 0.85,
    "suggested_fix": "Add error handling in the view or check the application factory config",
    "explanation": "The test expected HTTP 200 but received 500, indicating a server error",
}


def _make_cerebras_response(data: dict) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = json.dumps(data)
    return mock


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.query.return_value = [_SIMILAR_CASE]
    return store


@pytest.fixture
def agent(mock_store: MagicMock) -> TriageAgent:
    with patch("agent.triage_agent.Cerebras") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_cerebras_response(_LLM_RESPONSE)
        a = TriageAgent(store=mock_store, cerebras_api_key="fake-key")
        return a


# ---------------------------------------------------------------------------
# TriageReport
# ---------------------------------------------------------------------------

class TestTriageReport:
    def test_to_dict_is_json_serializable(self):
        report = TriageReport(
            failure_type="test_failure",
            root_cause="assertion error",
            confidence=0.9,
            suggested_fix="fix the test",
            explanation="the test failed",
            similar_cases=[],
            log_preview="FAILED tests/...",
        )
        d = report.to_dict()
        dumped = json.dumps(d)   # must not raise
        assert json.loads(dumped)["failure_type"] == "test_failure"

    def test_to_dict_includes_all_fields(self):
        report = TriageReport(
            failure_type="build_error",
            root_cause="missing symbol",
            confidence=0.7,
            suggested_fix="install dep",
            explanation="...",
            similar_cases=[{"repo": "a/b"}],
            log_preview="error TS2304",
        )
        d = report.to_dict()
        assert set(d.keys()) == {
            "failure_type", "root_cause", "confidence",
            "suggested_fix", "explanation", "similar_cases", "log_preview",
        }


# ---------------------------------------------------------------------------
# triage() — happy path
# ---------------------------------------------------------------------------

class TestTriage:
    def test_returns_triage_report(self, agent: TriageAgent):
        report = agent.triage(_FAKE_LOG)
        assert isinstance(report, TriageReport)

    def test_classifies_failure_type_correctly(self, agent: TriageAgent):
        report = agent.triage(_FAKE_LOG)
        assert report.failure_type == "test_failure"

    def test_root_cause_from_llm(self, agent: TriageAgent):
        report = agent.triage(_FAKE_LOG)
        assert report.root_cause == _LLM_RESPONSE["root_cause"]

    def test_confidence_from_llm(self, agent: TriageAgent):
        report = agent.triage(_FAKE_LOG)
        assert report.confidence == pytest.approx(0.85)

    def test_similar_cases_from_store(self, agent: TriageAgent, mock_store: MagicMock):
        report = agent.triage(_FAKE_LOG)
        assert len(report.similar_cases) == 1
        assert report.similar_cases[0]["repo"] == "pallets/flask"

    def test_log_preview_is_cleaned(self, agent: TriageAgent):
        """Log preview should not contain raw timestamps."""
        report = agent.triage("2026-01-01T00:00:00.000Z FAILED test_foo - AssertionError")
        assert "2026-01-01T" not in report.log_preview

    def test_calls_store_with_failure_type_filter(self, agent: TriageAgent, mock_store: MagicMock):
        agent.triage(_FAKE_LOG)
        first_call = mock_store.query.call_args_list[0]
        assert first_call.kwargs.get("failure_type") == "test_failure"

    def test_falls_back_to_unfiltered_query_when_empty(
        self, mock_store: MagicMock
    ):
        """If filtered query returns nothing, agent should retry without filter."""
        mock_store.query.side_effect = [[], [_SIMILAR_CASE]]
        with patch("agent.triage_agent.Cerebras") as mock_cls:
            mock_cls.return_value.chat.completions.create.return_value = (
                _make_cerebras_response(_LLM_RESPONSE)
            )
            a = TriageAgent(store=mock_store, cerebras_api_key="fake-key")
            a.triage(_FAKE_LOG)

        assert mock_store.query.call_count == 2
        # second call should have no failure_type filter
        second_call = mock_store.query.call_args_list[1]
        assert second_call.kwargs.get("failure_type") is None


# ---------------------------------------------------------------------------
# triage() — error handling
# ---------------------------------------------------------------------------

class TestTriageErrorHandling:
    def test_cerebras_api_error_returns_report_with_zero_confidence(
        self, mock_store: MagicMock
    ):
        with patch("agent.triage_agent.Cerebras") as mock_cls:
            mock_cls.return_value.chat.completions.create.side_effect = RuntimeError("API down")
            a = TriageAgent(store=mock_store, cerebras_api_key="fake-key")
            report = a.triage(_FAKE_LOG)

        assert report.confidence == 0.0
        assert "API error" in report.root_cause

    def test_malformed_json_from_cerebras_returns_fallback(
        self, mock_store: MagicMock
    ):
        bad_response = MagicMock()
        bad_response.choices[0].message.content = "not valid json {"
        with patch("agent.triage_agent.Cerebras") as mock_cls:
            mock_cls.return_value.chat.completions.create.return_value = bad_response
            a = TriageAgent(store=mock_store, cerebras_api_key="fake-key")
            report = a.triage(_FAKE_LOG)

        assert report.confidence == 0.0
        assert report.root_cause != ""   # should have a fallback message


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_system_prompt_mentions_json(self):
        assert "JSON" in build_system_prompt()

    def test_user_prompt_includes_failure_type(self):
        prompt = build_user_prompt("some log", [], "build_error")
        assert "build_error" in prompt

    def test_user_prompt_includes_log_snippet(self):
        prompt = build_user_prompt("ImportError: cannot import X", [], "dependency_error")
        assert "ImportError" in prompt

    def test_user_prompt_includes_similar_cases(self):
        prompt = build_user_prompt("some log", [_SIMILAR_CASE], "test_failure")
        assert "pallets/flask" in prompt
        assert "Similar Historical" in prompt

    def test_user_prompt_with_no_similar_cases(self):
        prompt = build_user_prompt("some log", [], "test_failure")
        assert "Similar Historical" not in prompt

    def test_user_prompt_truncates_long_log(self):
        long_log = "x" * 10_000
        prompt = build_user_prompt(long_log, [], "unknown")
        # The log snippet in the prompt should be capped at 3000 chars
        assert len(prompt) < 10_000
