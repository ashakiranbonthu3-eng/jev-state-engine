import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from jev.engine import node_gate
from jev.ledger import SqliteLedger
from jev.laya import (
    DEFAULT_LAYA_MODEL,
    LayaConfig,
    LayaResult,
    LayaShadowJudge,
)
from jev.models import State, Subgoal, ValidationVerdict, MechanicalCheckResult
from jev.qwen_postmortem import QwenClient, QwenConfig


class FakeWorkspaceForGate:
    """Mock workspace for testing node_gate isolation."""

    def __init__(
        self,
        repo_dir: Path,
        mech_passed: bool = True,
        failed_check: Optional[str] = None,
        detail: str = "",
        diff: str = "diff --git a/test.py b/test.py\n+print('hello')",
    ):
        self.repo_dir = repo_dir
        self.mech_passed = mech_passed
        self.failed_check = failed_check
        self.detail = detail
        self.diff = diff
        self.committed = False
        self.rolled_back = False

    def get_staged_diff(self) -> str:
        return self.diff

    def run_mechanical_checks(self, subgoal: Subgoal, **kwargs) -> MechanicalCheckResult:
        return MechanicalCheckResult(
            passed=self.mech_passed,
            failed_check=self.failed_check,
            detail=self.detail,
            checks_run=["build", "compile", "tests", "scope"],
            checks={
                "build": {"ran": True, "passed": True},
                "compile": {"ran": True, "passed": self.failed_check != "compile"},
                "tests": {"ran": True, "passed": self.failed_check != "tests"},
                "scope": {"ran": True, "passed": self.failed_check != "scope"},
            },
        )

    def commit_subgoal(self):
        self.committed = True

    def rollback_subgoal(self):
        self.rolled_back = True

    def _get_head_commit(self) -> str:
        return "fake-sha-head"

    def discard_subgoal_worktree(self, *args, **kwargs):
        pass


class FakeGatekeeper:
    def __init__(self, valid: bool = True, reason: str = "", probability: float = 1.0):
        self.valid = valid
        self.reason = reason
        self.probability = probability
        self.last_subgoal = None
        self.last_diff = None

    def validate_subgoal(self, subgoal, diff, **kwargs):
        self.last_subgoal = subgoal
        self.last_diff = diff
        return ValidationVerdict(
            valid=self.valid,
            reason=self.reason,
            probability=self.probability,
        )


# ==============================================================================
# 1. Configuration & Precedence Tests
# ==============================================================================

def test_laya_config_defaults():
    """Default-off behavior: when no CLI flags or env vars are set, Laya is off."""
    config = LayaConfig.from_env_and_args()
    assert config.enabled is False
    assert config.model == DEFAULT_LAYA_MODEL


def test_laya_config_env_vars(monkeypatch):
    """Env vars take effect when CLI flags are not provided."""
    monkeypatch.setenv("JEV_LAYA", "on")
    monkeypatch.setenv("JEV_LAYA_MODEL", "custom/laya-judge")

    config = LayaConfig.from_env_and_args()
    assert config.enabled is True
    assert config.model == "custom/laya-judge"

    # Also test '1' and 'true'
    monkeypatch.setenv("JEV_LAYA", "1")
    assert LayaConfig.from_env_and_args().enabled is True

    monkeypatch.setenv("JEV_LAYA", "true")
    assert LayaConfig.from_env_and_args().enabled is True

    monkeypatch.setenv("JEV_LAYA", "off")
    assert LayaConfig.from_env_and_args().enabled is False


def test_laya_config_cli_precedence_over_env(monkeypatch):
    """CLI flag strictly wins over environment variables."""
    # CLI off overrides ENV on
    monkeypatch.setenv("JEV_LAYA", "on")
    config = LayaConfig.from_env_and_args(cli_laya="off")
    assert config.enabled is False

    # CLI on overrides ENV off
    monkeypatch.setenv("JEV_LAYA", "off")
    config = LayaConfig.from_env_and_args(cli_laya="on")
    assert config.enabled is True

    # CLI model overrides ENV model
    monkeypatch.setenv("JEV_LAYA_MODEL", "env-model")
    config = LayaConfig.from_env_and_args(cli_model="cli-model")
    assert config.model == "cli-model"


# ==============================================================================
# 2. LayaShadowJudge Evaluation & Unanchored Input Tests
# ==============================================================================

def test_laya_shadow_judge_disabled_returns_empty():
    """When disabled, judge returns an empty result without invoking anything."""
    judge = LayaShadowJudge(LayaConfig(enabled=False))
    res = judge.evaluate(
        subgoal_description="Test subgoal",
        subgoal_scope=["test.py"],
        diff="fake diff",
    )
    assert res.verdict is None
    assert res.probability is None
    assert res.error is None


def test_laya_shadow_judge_unanchored_input_guarantee():
    """Verify Laya receives subgoal, diff, notes, and delta WITHOUT seeing Jev's verdict."""
    mock_agent = MagicMock()
    mock_agent.predict.return_value = {
        "answers": {
            "valid": {
                "noul": 0.92,
                "explanation": "Implementation matches scope exactly.",
            }
        }
    }

    judge = LayaShadowJudge(LayaConfig(enabled=True))
    with patch.object(judge, "_get_agent", return_value=mock_agent):
        res = judge.evaluate(
            subgoal_description="Fix payment rounding",
            subgoal_scope=["billing.py"],
            diff="+ rounded = round(val, 2)",
            investigation_notes="Billing uses banker's rounding.",
            mechanical_detail="All 5 tests pass.",
            dependency_delta={"added": []},
        )

    assert mock_agent.predict.called
    call_kwargs = mock_agent.predict.call_args[1]
    passed_state = call_kwargs["state"]

    # Verify input components are present
    assert passed_state["subgoal"] == "Fix payment rounding"
    assert passed_state["description"] == "Fix payment rounding"
    assert passed_state["scope"] == ["billing.py"]
    assert passed_state["diff"] == "+ rounded = round(val, 2)"
    assert passed_state["investigation_notes"] == "Billing uses banker's rounding."
    assert passed_state["mechanical_detail"] == "All 5 tests pass."
    assert passed_state["dependency_delta"] == {"added": []}

    # CRITICAL: Verify NO anchoring to Jev verdict
    for key in passed_state:
        assert "jev" not in key.lower()
        assert "verdict" not in key.lower()

    # Result fields parsed correctly
    assert res.verdict is True
    assert res.probability == 0.92
    assert res.reason == "Implementation matches scope exactly."
    assert res.latency_ms is not None
    assert res.error is None


def test_laya_shadow_judge_parses_noul_thresholds():
    """Noul probability >= 0.5 is True, < 0.5 is False."""
    mock_agent = MagicMock()
    judge = LayaShadowJudge(LayaConfig(enabled=True))

    with patch.object(judge, "_get_agent", return_value=mock_agent):
        # Case A: >= 0.5 -> True
        mock_agent.predict.return_value = {"answers": {"valid": {"noul": 0.50}}}
        res_a = judge.evaluate("sg", ["a.py"], "diff")
        assert res_a.verdict is True
        assert res_a.probability == 0.50

        # Case B: < 0.5 -> False
        mock_agent.predict.return_value = {"answers": {"valid": {"noul": 0.49, "reason": "Missing edge case"}}}
        res_b = judge.evaluate("sg", ["a.py"], "diff")
        assert res_b.verdict is False
        assert res_b.probability == 0.49
        assert res_b.reason == "Missing edge case"

        # Case C: Malformed (missing noul)
        mock_agent.predict.return_value = {"answers": {"valid": {}}}
        res_c = judge.evaluate("sg", ["a.py"], "diff")
        assert res_c.verdict is None
        assert "missing noul probability" in res_c.error


def test_laya_shadow_judge_fail_open_on_exception():
    """If laya import, load, or predict fails, judge fails open without crashing."""
    # 1. Load failure
    judge_load_fail = LayaShadowJudge(LayaConfig(enabled=True))
    with patch("builtins.__import__", side_effect=ImportError("No module named laya")):
        agent = judge_load_fail._get_agent()
        assert agent is None
        assert judge_load_fail.enabled is False
        res = judge_load_fail.evaluate("sg", ["a.py"], "diff")
        assert res.error == "Laya agent unavailable"

    # 2. Predict runtime exception
    mock_agent = MagicMock()
    mock_agent.predict.side_effect = RuntimeError("GPU out of memory")
    judge_pred_fail = LayaShadowJudge(LayaConfig(enabled=True))
    with patch.object(judge_pred_fail, "_get_agent", return_value=mock_agent):
        res = judge_pred_fail.evaluate("sg", ["a.py"], "diff")
        assert res.verdict is None
        assert "GPU out of memory" in res.error
        assert res.latency_ms is not None


# ==============================================================================
# 3. Fail-Open Resilience in node_gate Tests
# ==============================================================================

def test_node_gate_fail_open_laya_error_does_not_alter_run_outcome(tmp_path):
    """When Jev passes, but Laya throws an error/timeout, run outcome is identical to Laya off."""
    ws = FakeWorkspaceForGate(tmp_path, mech_passed=True)
    gatekeeper = FakeGatekeeper(valid=True, reason="Looks great")

    base_state: State = {
        "ticket": "Implement feature",
        "thread_id": "test-fail-open",
        "run_id": "test-fail-open",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
        "mechanical_strike_count": 0,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    # Run 1: Laya off
    state_off = node_gate(dict(base_state), workspace=ws, gatekeeper=gatekeeper, laya_client=None)

    # Run 2: Laya on but raises exception during evaluation
    mock_failing_laya = MagicMock()
    mock_failing_laya.enabled = True
    mock_failing_laya.evaluate.side_effect = TimeoutError("Laya model timed out after 30s")

    state_on = node_gate(dict(base_state), workspace=ws, gatekeeper=gatekeeper, laya_client=mock_failing_laya)

    # Both must pass cleanly with strikes reset to 0
    assert state_on["gate_status"] == state_off["gate_status"] == "passed"
    assert state_on["semantic_strike_count"] == state_off["semantic_strike_count"] == 0
    assert state_on["mechanical_strike_count"] == state_off["mechanical_strike_count"] == 0
    assert state_on["current_subgoal"].status == "committed"

    # Trajectory captures error without failing
    traj = state_on["trajectory"][-1]
    assert traj["laya_verdict"] is None
    assert "Laya model timed out" in traj["laya_error"]
    assert traj["judges_agree"] is None


def test_node_gate_disagreement_jev_valid_laya_invalid(tmp_path):
    """Jev is True, Laya is False: Jev authority governs, commit succeeds, disagreement recorded."""
    ws = FakeWorkspaceForGate(tmp_path, mech_passed=True)
    gatekeeper = FakeGatekeeper(valid=True, reason="Jev says valid")

    mock_laya = MagicMock()
    mock_laya.enabled = True
    mock_laya.evaluate.return_value = LayaResult(
        verdict=False,
        probability=0.25,
        reason="Laya claims scope creep",
        latency_ms=120.0,
    )

    state: State = {
        "ticket": "Implement feature",
        "thread_id": "test-disagree-1",
        "run_id": "test-disagree-1",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
        "mechanical_strike_count": 0,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    out = node_gate(state, workspace=ws, gatekeeper=gatekeeper, laya_client=mock_laya)

    # Jev authority governs: PASSED
    assert out["gate_status"] == "passed"
    assert out["current_subgoal"].status == "committed"
    assert ws.committed is True

    # Telemetry records disagreement
    traj = out["trajectory"][-1]
    assert traj["jev_verdict"]["valid"] is True
    assert traj["laya_verdict"] is False
    assert traj["laya_reason"] == "Laya claims scope creep"
    assert traj["judges_agree"] is False


def test_node_gate_disagreement_jev_invalid_laya_valid(tmp_path):
    """Jev is False, Laya is True: Jev authority governs, strike incremented, laya_opinion passed to Qwen."""
    ws = FakeWorkspaceForGate(tmp_path, mech_passed=True)
    gatekeeper = FakeGatekeeper(valid=False, reason="Jev says invalid syntax and missing tests")

    mock_laya = MagicMock()
    mock_laya.enabled = True
    mock_laya.evaluate.return_value = LayaResult(
        verdict=True,
        probability=0.91,
        reason="Laya says code is correct",
        latency_ms=110.0,
    )

    mock_qwen = MagicMock()
    mock_qwen.enabled = True
    mock_qwen.run_postmortem.return_value = None

    state: State = {
        "ticket": "Implement feature",
        "thread_id": "test-disagree-2",
        "run_id": "test-disagree-2",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
        "mechanical_strike_count": 0,
        "semantic_strike_count": 0,
        "trajectory": [],
        "last_feedback": None,
    }

    out = node_gate(state, workspace=ws, gatekeeper=gatekeeper, laya_client=mock_laya, qwen_client=mock_qwen)

    # Jev authority governs: REJECTED
    assert out["gate_status"] == "semantic_failure"
    assert out["semantic_strike_count"] == 1
    assert ws.committed is False

    # Disagreement recorded
    traj = out["trajectory"][-1]
    assert traj["jev_verdict"]["valid"] is False
    assert traj["laya_verdict"] is True
    assert traj["judges_agree"] is False

    # Qwen received laya opinion!
    assert mock_qwen.run_postmortem.called
    qwen_call_kwargs = mock_qwen.run_postmortem.call_args[1]
    assert qwen_call_kwargs["laya_opinion"] == {
        "verdict": True,
        "reason": "Laya says code is correct",
    }


def test_node_gate_agreement_both_valid_and_both_invalid(tmp_path):
    """When both agree (both True or both False), judges_agree is True."""
    ws = FakeWorkspaceForGate(tmp_path, mech_passed=True)

    # 1. Both valid
    gk_valid = FakeGatekeeper(valid=True)
    laya_valid = MagicMock(enabled=True)
    laya_valid.evaluate.return_value = LayaResult(verdict=True, probability=0.9, latency_ms=50.0)

    state1: State = {
        "ticket": "Agree test",
        "thread_id": "agree-1",
        "run_id": "agree-1",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
        "trajectory": [],
    }
    out1 = node_gate(state1, workspace=ws, gatekeeper=gk_valid, laya_client=laya_valid)
    assert out1["trajectory"][-1]["judges_agree"] is True

    # 2. Both invalid
    gk_invalid = FakeGatekeeper(valid=False, reason="Bad diff")
    laya_invalid = MagicMock(enabled=True)
    laya_invalid.evaluate.return_value = LayaResult(verdict=False, probability=0.1, latency_ms=50.0)

    state2: State = {
        "ticket": "Agree test",
        "thread_id": "agree-2",
        "run_id": "agree-2",
        "current_subgoal": Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
        "trajectory": [],
    }
    out2 = node_gate(state2, workspace=ws, gatekeeper=gk_invalid, laya_client=laya_invalid)
    assert out2["trajectory"][-1]["judges_agree"] is True


# ==============================================================================
# 4. SqliteLedger Persistence & Reporting Tests
# ==============================================================================

def test_sqlite_ledger_records_and_retrieves_laya_fields(tmp_path):
    """SqliteLedger stores laya_verdict, laya_reason, laya_latency_ms, laya_error, and judges_agree."""
    db_path = tmp_path / "test_ledger.db"
    ledger = SqliteLedger(db_path=db_path)

    run_id = "test-laya-ledger"
    ledger.record_run_start(run_id=run_id, thread_id=run_id, ticket="Test ticket")

    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="semantic",
        outcome="passed",
        diff="+ code",
        diff_sha="abc123",
        jev_valid=True,
        jev_probability=0.95,
        laya_verdict=True,
        laya_reason="Correctly implemented",
        laya_latency_ms=145.5,
        laya_error=None,
        judges_agree=True,
    )

    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-2",
        tier="semantic",
        outcome="semantic_failure",
        diff="+ code2",
        diff_sha="def456",
        jev_valid=False,
        jev_probability=0.20,
        laya_verdict=True,
        laya_reason="Looks fine to me",
        laya_latency_ms=210.0,
        laya_error=None,
        judges_agree=False,
    )

    with ledger._get_connection() as conn:
        rows = conn.execute(
            "SELECT subgoal_id, jev_valid, laya_verdict, laya_reason, laya_latency_ms, laya_error, judges_agree "
            "FROM attempts WHERE run_id = ? ORDER BY subgoal_id",
            (run_id,),
        ).fetchall()

        assert len(rows) == 2

        # Row 1 (Agreed)
        assert rows[0][0] == "sg-1"
        assert rows[0][1] == 1
        assert rows[0][2] == 1
        assert rows[0][3] == "Correctly implemented"
        assert rows[0][4] == 145.5
        assert rows[0][5] is None
        assert rows[0][6] == 1

        # Row 2 (Disagreed)
        assert rows[1][0] == "sg-2"
        assert rows[1][1] == 0
        assert rows[1][2] == 1
        assert rows[1][3] == "Looks fine to me"
        assert rows[1][4] == 210.0
        assert rows[1][5] is None
        assert rows[1][6] == 0


def test_ledger_report_agreement_calculation(tmp_path, capsys):
    """scripts/ledger_report.py calculates agreement rate and lists discrepancies."""
    from scripts.ledger_report import main as report_main

    db_path = tmp_path / "ledger.db"
    ledger = SqliteLedger(db_path=db_path)
    run_id = "test-report-run"

    ledger.record_run_start(run_id=run_id, thread_id=run_id, ticket="Report ticket")
    ledger.record_subgoals(
        run_id=run_id,
        subgoals=[
            Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"]),
            Subgoal(id="sg-2", description="Subgoal 2", scope=["b.py"]),
        ],
    )

    # 1 Agreed attempt
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="semantic",
        outcome="passed",
        diff="diff1",
        diff_sha="sha1",
        jev_valid=True,
        laya_verdict=True,
        laya_reason="Agreed ok",
        judges_agree=True,
    )

    # 1 Disagreed attempt
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-2",
        tier="semantic",
        outcome="semantic_failure",
        diff="diff2",
        diff_sha="sha2",
        jev_valid=False,
        laya_verdict=True,
        laya_reason="Laya approves",
        judges_agree=False,
    )

    exit_code = report_main(["--db-path", str(db_path), "--run-id", run_id])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "LAYA SHADOW AGREEMENT" in captured
    assert "Total Evaluations:   2" in captured
    assert "Agreed:              1" in captured
    assert "Disagreed:           1" in captured
    assert "Agreement Rate:      50.0%" in captured
    assert "Discrepancies:" in captured
    assert "[sg-2 - Attempt 1] Jev Valid: 0 | Laya: 1 | Reason: Laya approves" in captured


def test_export_finetuning_dataset_exports_laya_fields(tmp_path):
    """scripts/export_finetuning_dataset.py includes shadow_judge in exported records."""
    from scripts.export_finetuning_dataset import export_dataset

    db_path = tmp_path / "ledger.db"
    jsonl_path = tmp_path / "dataset.jsonl"
    ledger = SqliteLedger(db_path=db_path)
    run_id = "test-export-run"

    ledger.record_run_start(run_id=run_id, thread_id=run_id, ticket="Export ticket")
    ledger.record_subgoals(
        run_id=run_id,
        subgoals=[Subgoal(id="sg-1", description="Subgoal 1", scope=["a.py"])],
    )
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="semantic",
        outcome="passed",
        diff="diff1",
        diff_sha="sha1",
        jev_valid=True,
        laya_verdict=True,
        laya_reason="Looks good",
        laya_latency_ms=130.0,
        judges_agree=True,
    )

    ret = export_dataset(db_path, jsonl_path)
    assert ret == 0
    assert jsonl_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert "shadow_judge" in record
    shadow = record["shadow_judge"]
    assert shadow["verdict"] == 1 or shadow["verdict"] is True
    assert shadow["reason"] == "Looks good"
    assert shadow["latency_ms"] == 130.0
    assert shadow["judges_agree"] is True
