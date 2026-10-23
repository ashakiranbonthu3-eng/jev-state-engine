import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import httpx

from jev.engine import JevEngine, node_gate
from jev.ledger import SqliteLedger, get_jev_home
from jev.models import State, Subgoal, ValidationVerdict, MechanicalCheckResult
from jev.qwen_postmortem import (
    DEFAULT_QWEN_MAX_DIFF_CHARS,
    DEFAULT_QWEN_TIMEOUT,
    DEFAULT_QWEN_URL,
    QwenClient,
    QwenConfig,
    format_postmortem_prompt,
    parse_qwen_response,
)
from jev.workspace import Workspace


# ==============================================================================
# 1. Configuration & Precedence Tests
# ==============================================================================

def test_qwen_config_defaults():
    """Default-off behavior: when no CLI flags or env vars are set, all defaults apply."""
    config = QwenConfig.from_env_and_args()
    assert config.enabled is False
    assert config.url == DEFAULT_QWEN_URL
    assert config.model is None
    assert config.timeout == DEFAULT_QWEN_TIMEOUT
    assert config.max_diff_chars == DEFAULT_QWEN_MAX_DIFF_CHARS
    assert config.feedback is False


def test_qwen_config_env_vars(monkeypatch):
    """Env vars take effect when CLI flags are not provided."""
    monkeypatch.setenv("JEV_QWEN_POSTMORTEM", "on")
    monkeypatch.setenv("JEV_QWEN_URL", "http://custom-llama:8080")
    monkeypatch.setenv("JEV_QWEN_MODEL", "qwen-2.5-coder")
    monkeypatch.setenv("JEV_QWEN_TIMEOUT", "60")
    monkeypatch.setenv("JEV_QWEN_MAX_DIFF_CHARS", "12000")
    monkeypatch.setenv("JEV_QWEN_FEEDBACK", "on")

    config = QwenConfig.from_env_and_args()
    assert config.enabled is True
    assert config.url == "http://custom-llama:8080"
    assert config.model == "qwen-2.5-coder"
    assert config.timeout == 60.0
    assert config.max_diff_chars == 12000
    assert config.feedback is True


def test_qwen_config_cli_precedence_over_env(monkeypatch):
    """CLI flags must win over env vars."""
    monkeypatch.setenv("JEV_QWEN_POSTMORTEM", "off")
    monkeypatch.setenv("JEV_QWEN_URL", "http://env-llama:8080")
    monkeypatch.setenv("JEV_QWEN_MODEL", "env-model")
    monkeypatch.setenv("JEV_QWEN_TIMEOUT", "90")
    monkeypatch.setenv("JEV_QWEN_MAX_DIFF_CHARS", "8000")
    monkeypatch.setenv("JEV_QWEN_FEEDBACK", "off")

    config = QwenConfig.from_env_and_args(
        cli_postmortem="on",
        cli_url="http://cli-llama:9000",
        cli_model="cli-model",
        cli_timeout=45,
        cli_max_diff_chars=16000,
        cli_feedback="on",
    )
    assert config.enabled is True
    assert config.url == "http://cli-llama:9000"
    assert config.model == "cli-model"
    assert config.timeout == 45.0
    assert config.max_diff_chars == 16000
    assert config.feedback is True


# ==============================================================================
# 2. Server Health Check & Model Discovery Tests
# ==============================================================================

def test_qwen_health_check_auto_detects_model():
    """Health check queries /v1/models and auto-detects first model if none configured."""
    config = QwenConfig(enabled=True, url="http://127.0.0.1:8080", model=None)
    client = QwenClient(config=config)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"data": [{"id": "qwen2.5-coder-7b-instruct.gguf"}]}

    with patch("httpx.Client.get", return_value=fake_resp):
        assert client.check_health() is True
        assert client.model == "qwen2.5-coder-7b-instruct.gguf"
        assert client.enabled is True


def test_qwen_health_check_unreachable_disables_for_run(caplog):
    """If server is unreachable, health check warns once and disables for run."""
    config = QwenConfig(enabled=True, url="http://127.0.0.1:9999")
    client = QwenClient(config=config)

    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        assert client.check_health() is False
        assert client.enabled is False

    # Second call should return cached status without making calls
    assert client.check_health() is False


def test_qwen_health_check_http_error_disables_for_run():
    """If server returns non-200 on health check, disables for run."""
    config = QwenConfig(enabled=True, url="http://127.0.0.1:8080")
    client = QwenClient(config=config)

    fake_resp = MagicMock()
    fake_resp.status_code = 503
    with patch("httpx.Client.get", return_value=fake_resp):
        assert client.check_health() is False
        assert client.enabled is False


# ==============================================================================
# 3. Prompt Construction & Truncation Tests
# ==============================================================================

def test_prompt_construction_contents():
    """Prompt must contain ticket, rejecting source, reason, Jev output, and diff."""
    prompt = format_postmortem_prompt(
        ticket="Add OAuth2 token refresh endpoint",
        rejecting_source="Tier 0 mechanical check: tests",
        rejection_reason="AssertionError: Token expired",
        jev_verdict=None,
        diff="--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n+def refresh(): pass",
        strike_number="1 of 3",
        previous_strikes=None,
    )
    assert "Add OAuth2 token refresh endpoint" in prompt
    assert "Tier 0 mechanical check: tests" in prompt
    assert "AssertionError: Token expired" in prompt
    assert "Verdict: Not evaluated" in prompt
    assert "no reason provided" in prompt
    assert "+def refresh(): pass" in prompt
    assert "1 of 3" in prompt


def test_prompt_jev_null_reason_displays_no_reason_provided():
    """When Jev's reason is null, prompt must explicitly output 'no reason provided'."""
    verdict_null_reason = ValidationVerdict(valid=False, reason=None)
    prompt = format_postmortem_prompt(
        ticket="Task X",
        rejecting_source="Tier 1 Jev semantic validation",
        rejection_reason="Semantic failure",
        jev_verdict=verdict_null_reason,
        diff="+x = 1",
        strike_number="2 of 3",
    )
    assert "Verdict: Invalid" in prompt
    assert "Reason: no reason provided" in prompt


def test_diff_truncation_marker_present_when_exceeded():
    """Requirement: The diff truncation marker [TRUNCATED] is present when the limit is exceeded."""
    huge_diff = "+" + ("a" * 10000)
    prompt = format_postmortem_prompt(
        ticket="Task Y",
        rejecting_source="Tier 0",
        rejection_reason="Error",
        jev_verdict=None,
        diff=huge_diff,
        strike_number="1 of 3",
        max_diff_chars=100,
    )
    assert "[TRUNCATED]" in prompt
    assert huge_diff not in prompt
    assert len(prompt) < len(huge_diff)


# ==============================================================================
# 4. JSON Parsing & Fallback Tests
# ==============================================================================

def test_parse_qwen_response_valid_json():
    raw_json = json.dumps({
        "chain_of_thought": "Worker missed boundary check.",
        "where_it_went_wrong": "src/pagination.py: line 15",
        "root_cause": "Off-by-one error in page indexing",
        "suggested_fix": "Add 1 to page index before division",
        "confidence": "high",
    })
    parsed, is_fallback = parse_qwen_response(raw_json)
    assert not is_fallback
    assert parsed["confidence"] == "high"
    assert parsed["root_cause"] == "Off-by-one error in page indexing"
    assert parsed["where_it_went_wrong"] == "src/pagination.py: line 15"


def test_parse_qwen_response_markdown_fences():
    fenced = """```json
{
  "chain_of_thought": "Reasoning...",
  "where_it_went_wrong": "File X",
  "root_cause": "Typo",
  "suggested_fix": "Fix typo",
  "confidence": "medium"
}
```"""
    parsed, is_fallback = parse_qwen_response(fenced)
    assert not is_fallback
    assert parsed["confidence"] == "medium"
    assert parsed["root_cause"] == "Typo"


def test_parse_qwen_response_garbage_fallback():
    garbage = "I think the code failed because the test was expecting something else."
    parsed, is_fallback = parse_qwen_response(garbage)
    assert is_fallback
    assert parsed["confidence"] == "low"
    assert parsed["chain_of_thought"] == garbage
    assert garbage in parsed["root_cause"]


# ==============================================================================
# 5. Fail-Open Paths & Mechanical Isolation Tests
# ==============================================================================

class FakeWorkspaceForGate:
    def __init__(self, repo_dir: Path, diff: str = "+x = 1\n", mech_passed: bool = False, failed_check: str = "tests", detail: str = "Test failed"):
        self.repo_dir = repo_dir
        self._diff = diff
        self._mech_passed = mech_passed
        self._failed_check = failed_check
        self._detail = detail

    def get_staged_diff(self) -> str:
        return self._diff

    def run_mechanical_checks(self, subgoal, **kwargs):
        return MechanicalCheckResult(
            passed=self._mech_passed,
            failed_check=self._failed_check if not self._mech_passed else None,
            detail=self._detail,
            checks_run=["compile", "tests"],
            checks={"compile": {"ran": True, "passed": True}, "tests": {"ran": True, "passed": self._mech_passed}},
        )

    def discard_subgoal_worktree(self, *args, **kwargs):
        pass


def test_qwen_failure_timeout_unreachable_does_not_change_run_outcome(tmp_path):
    """(a) Qwen failure, timeout, unreachable server, or garbage output does not change the run outcome compared to Qwen off."""
    base_state_qwen_off: State = {
        "ticket": "Implement feature A",
        "thread_id": "test-fail-open",
        "run_id": "test-fail-open",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
        "mechanical_strike_count": 0,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    ws = FakeWorkspaceForGate(tmp_path, mech_passed=False, failed_check="tests", detail="pytest failed")

    # 1. Run with Qwen off
    state_off = node_gate(dict(base_state_qwen_off), workspace=ws, gatekeeper=None, qwen_client=None)

    # 2. Run with Qwen on but server throws timeout
    qwen_config = QwenConfig(enabled=True, feedback=False)
    failing_client = QwenClient(config=qwen_config)
    failing_client.run_postmortem = MagicMock(return_value=None)  # Simulate fail open / exception

    base_state_qwen_on = dict(base_state_qwen_off)
    state_on = node_gate(base_state_qwen_on, workspace=ws, gatekeeper=None, qwen_client=failing_client)

    # Outcomes must be identical
    assert state_on["gate_status"] == state_off["gate_status"] == "mechanical_failure"
    assert state_on["mechanical_strike_count"] == state_off["mechanical_strike_count"] == 1
    assert state_on["semantic_strike_count"] == state_off["semantic_strike_count"] == 0


def test_byte_identical_retry_prompt_when_feedback_off(tmp_path):
    """(b) With --qwen-feedback off, the worker's retry prompt is byte-identical to a run with Qwen off."""
    base_state: State = {
        "ticket": "Implement feature B",
        "thread_id": "test-byte-identical",
        "run_id": "test-byte-identical",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["b.py"]),
        "mechanical_strike_count": 0,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    ws = FakeWorkspaceForGate(tmp_path, mech_passed=False, failed_check="compile", detail="SyntaxError: invalid syntax")

    # Run 1: Qwen off
    out_off = node_gate(dict(base_state), workspace=ws, gatekeeper=None, qwen_client=None)

    # Run 2: Qwen on with feedback off and active postmortem result
    mock_postmortem = MagicMock()
    mock_postmortem.to_json.return_value = '{"root_cause": "syntax error"}'
    mock_postmortem.latency_ms = 120.0
    mock_postmortem.error = None
    mock_postmortem.root_cause = "Syntax error"
    mock_postmortem.where_it_went_wrong = "line 1"
    mock_postmortem.suggested_fix = "add colon"

    qwen_config = QwenConfig(enabled=True, feedback=False)
    client_feedback_off = QwenClient(config=qwen_config)
    client_feedback_off.run_postmortem = MagicMock(return_value=mock_postmortem)

    out_fb_off = node_gate(dict(base_state), workspace=ws, gatekeeper=None, qwen_client=client_feedback_off)

    # Byte-identical verification:
    assert out_off["last_feedback"] == out_fb_off["last_feedback"]
    assert out_fb_off["last_feedback"].encode("utf-8") == out_off["last_feedback"].encode("utf-8")


def test_feedback_on_leaves_strike_counting_and_verdict_identical(tmp_path):
    """(d) With --qwen-feedback on, strike counting and the verdict are still identical to a run with Qwen off."""
    base_state: State = {
        "ticket": "Implement feature C",
        "thread_id": "test-fb-on",
        "run_id": "test-fb-on",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["c.py"]),
        "mechanical_strike_count": 1,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    ws = FakeWorkspaceForGate(tmp_path, mech_passed=False, failed_check="tests", detail="1 test failed")

    # Run 1: Qwen off
    out_off = node_gate(dict(base_state), workspace=ws, gatekeeper=None, qwen_client=None)

    # Run 2: Qwen on with feedback ON
    mock_postmortem = MagicMock()
    mock_postmortem.to_json.return_value = '{"root_cause": "test assertion"}'
    mock_postmortem.latency_ms = 150.0
    mock_postmortem.error = None
    mock_postmortem.root_cause = "Test assertion"
    mock_postmortem.where_it_went_wrong = "tests/test_c.py"
    mock_postmortem.suggested_fix = "fix return value"

    qwen_config = QwenConfig(enabled=True, feedback=True)
    client_feedback_on = QwenClient(config=qwen_config)
    client_feedback_on.run_postmortem = MagicMock(return_value=mock_postmortem)

    out_fb_on = node_gate(dict(base_state), workspace=ws, gatekeeper=None, qwen_client=client_feedback_on)

    # Strikes and gate status MUST BE IDENTICAL
    assert out_fb_on["gate_status"] == out_off["gate_status"] == "mechanical_failure"
    assert out_fb_on["mechanical_strike_count"] == out_off["mechanical_strike_count"] == 2
    assert out_fb_on["semantic_strike_count"] == out_off["semantic_strike_count"] == 0

    # But feedback has advisory block appended
    assert "[Advisory analysis, may be wrong]" in out_fb_on["last_feedback"]
    assert "Where it went wrong: tests/test_c.py" in out_fb_on["last_feedback"]
    assert "Suggested fix: fix return value" in out_fb_on["last_feedback"]


# ==============================================================================
# 6. Post-Mortem Markdown Persistence & Ledger Storage Tests
# ==============================================================================

def test_postmortem_markdown_file_and_ledger_written(tmp_path):
    """Post-mortem writes strike markdown under ~/.jev/postmortems/<thread_id>/strike-<n>.md

    and records qwen_postmortem, qwen_latency_ms, qwen_error in ledger and trajectory.
    """
    thread_id = "test-postmortem-storage"
    ledger = SqliteLedger(db_path=tmp_path / "ledger.db")

    fake_chat_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "chain_of_thought": "Analysis of failure.",
                        "where_it_went_wrong": "file.py:20",
                        "root_cause": "Missing argument in function call",
                        "suggested_fix": "Pass argument foo to bar()",
                        "confidence": "high",
                    })
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_chat_response

    qwen_config = QwenConfig(enabled=True, feedback=False)
    qwen_client = QwenClient(config=qwen_config)

    state: State = {
        "ticket": "Fix argument bug",
        "thread_id": thread_id,
        "run_id": thread_id,
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["file.py"]),
        "mechanical_strike_count": 0,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    ws = FakeWorkspaceForGate(tmp_path, mech_passed=False, failed_check="tests", detail="Missing arg error")

    with patch("httpx.Client.post", return_value=mock_resp):
        out_state = node_gate(state, workspace=ws, gatekeeper=None, recorder=ledger, qwen_client=qwen_client)

    # 1. Assert markdown file written under ~/.jev/postmortems/<thread_id>/strike-1.md
    expected_md = get_jev_home() / "postmortems" / thread_id / "strike-1.md"
    assert expected_md.exists(), f"Expected postmortem file at {expected_md}"
    md_content = expected_md.read_text(encoding="utf-8")
    assert "Qwen Advisory Post-Mortem: Strike 1" in md_content
    assert "Missing argument in function call" in md_content
    assert "Pass argument foo to bar()" in md_content

    # 2. Assert trajectory contains qwen fields
    traj = out_state["trajectory"]
    assert len(traj) == 1
    assert "qwen_postmortem" in traj[0]
    assert traj[0]["qwen_postmortem"] is not None
    assert "Missing argument in function call" in traj[0]["qwen_postmortem"]
    assert traj[0]["qwen_latency_ms"] is not None
    assert traj[0]["qwen_error"] is None

    # 3. Assert ledger attempts table recorded qwen fields
    with ledger._get_connection() as conn:
        row = conn.execute("SELECT qwen_postmortem, qwen_latency_ms, qwen_error FROM attempts WHERE run_id = ?", (thread_id,)).fetchone()
        assert row is not None
        assert "Missing argument in function call" in row[0]
        assert row[1] is not None  # latency
        assert row[2] is None      # error
