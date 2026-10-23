#!/usr/bin/env python3
"""Simulate a full end-to-end run in the local ledger.

Executes attempts through Jev Gatekeeper and Laya Shadow Judge,
recording attempts, agreements, and discrepancies into the repository ledger
so that scripts/ledger_report.py and scripts/export_finetuning_dataset.py
can be inspected immediately without external API keys.
"""

import sys
import time
from pathlib import Path

# Ensure repo's src directory is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from jev.laya import LayaConfig, LayaShadowJudge
from jev.ledger import SqliteLedger, compute_repo_key, get_jev_home
from jev.models import Subgoal


def main() -> int:
    repo_dir = Path(".").resolve()
    repo_key = compute_repo_key(repo_dir)
    db_path = get_jev_home() / "repos" / repo_key / "ledger.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SIMULATING END-TO-END RUN WITH LAYA SHADOW JUDGE")
    print("=" * 65)
    print(f"Target Repository: {repo_dir}")
    print(f"Repo Key:          {repo_key}")
    print(f"Ledger DB Path:    {db_path}")
    print()

    ledger = SqliteLedger(db_path=db_path)
    run_id = f"demo-run-{int(time.time())}"
    ticket = "Refactor user authentication and validate session expiration handling."

    print(f"1. Recording Run Start: {run_id}...")
    ledger.record_run_start(
        run_id=run_id,
        thread_id=run_id,
        ticket=ticket,
        worker_model="gemini-3.5-flash-lite",
        test_policy="auto",
    )

    subgoals = [
        Subgoal(
            id="sg-1",
            description="Implement token expiration validation in src/auth/session.py",
            scope=["src/auth/session.py"],
            expects_tests=True,
        ),
        Subgoal(
            id="sg-2",
            description="Add unit tests for expired session rejection in tests/test_session.py",
            scope=["tests/test_session.py"],
            expects_tests=True,
        ),
    ]
    ledger.record_subgoals(run_id=run_id, subgoals=subgoals)

    # Initialize Laya Shadow Judge
    laya = LayaShadowJudge(LayaConfig(enabled=True))

    print("2. Simulating Subgoal 1 (Agreement: Both Jev & Laya approve)...")
    sg1_diff = (
        "--- a/src/auth/session.py\n"
        "+++ b/src/auth/session.py\n"
        "@@ -10,3 +10,6 @@\n"
        " def is_session_valid(session: dict) -> bool:\n"
        "+    if time.time() > session.get('expires_at', 0):\n"
        "+        return False\n"
        "     return session.get('active', False)\n"
    )

    t0 = time.perf_counter()
    laya_res1 = laya.evaluate(
        subgoal_description=subgoals[0].description,
        subgoal_scope=subgoals[0].scope,
        diff=sg1_diff,
        investigation_notes="Project uses timestamp-based expiration.",
        mechanical_detail="All unit tests passed.",
    )
    lat1 = laya_res1.latency_ms or round((time.perf_counter() - t0) * 1000, 2)

    # Simulated Jev decision for sg-1: Valid (Both agree)
    jev_valid_1 = True
    laya_verdict_1 = laya_res1.verdict if laya_res1.verdict is not None else True
    agree_1 = (jev_valid_1 == laya_verdict_1)

    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="semantic",
        outcome="passed",
        diff=sg1_diff,
        diff_sha="a1b2c3d4e5",
        files=["src/auth/session.py"],
        tests_ran=True,
        test_outcome="PASSED",
        jev_valid=jev_valid_1,
        jev_probability=0.96,
        jev_model="jev-system-one-latest",
        laya_verdict=laya_verdict_1,
        laya_reason=laya_res1.reason or "Implementation correctly matches subgoal and declared scope.",
        laya_latency_ms=lat1,
        laya_error=laya_res1.error,
        judges_agree=agree_1,
    )
    ledger.record_subgoal_committed(run_id=run_id, subgoal_id="sg-1", commit_sha="commit-sha-1")
    print(f"   • Jev: valid={jev_valid_1} | Laya: valid={laya_verdict_1} | Agree: {agree_1}")

    print("3. Simulating Subgoal 2 (Disagreement: Jev rejects, Laya approves)...")
    sg2_diff = (
        "--- a/tests/test_session.py\n"
        "+++ b/tests/test_session.py\n"
        "@@ -1,3 +1,6 @@\n"
        "+def test_session_expired():\n"
        "+    assert not is_session_valid({'expires_at': 0})\n"
    )

    t0 = time.perf_counter()
    laya_res2 = laya.evaluate(
        subgoal_description=subgoals[1].description,
        subgoal_scope=subgoals[1].scope,
        diff=sg2_diff,
        investigation_notes="Tests must test boundary conditions.",
        mechanical_detail="Unit tests ran.",
    )
    lat2 = laya_res2.latency_ms or round((time.perf_counter() - t0) * 1000, 2)

    # Simulated Jev decision for sg-2: Jev rejects for missing boundary assertion, Laya passes
    jev_valid_2 = False
    laya_verdict_2 = True
    agree_2 = False

    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-2",
        tier="semantic",
        outcome="semantic_failure",
        failed_check="jev_rejection",
        diff=sg2_diff,
        diff_sha="f6g7h8i9j0",
        files=["tests/test_session.py"],
        tests_ran=True,
        test_outcome="PASSED",
        jev_valid=jev_valid_2,
        jev_probability=0.35,
        jev_model="jev-system-one-latest",
        feedback_sent="Diff omitted boundary test condition for exact expiration timestamp.",
        laya_verdict=laya_verdict_2,
        laya_reason="Test verifies basic expiration behavior adequately.",
        laya_latency_ms=lat2,
        laya_error=laya_res2.error,
        judges_agree=agree_2,
    )
    print(f"   • Jev: valid={jev_valid_2} | Laya: valid={laya_verdict_2} | Agree: {agree_2}")

    ledger.record_run_end(
        run_id=run_id,
        final_status="completed",
        final_commit="final-commit-sha-2",
    )

    print()
    print("=" * 65)
    print("Run simulated successfully! Database populated at:")
    print(f"  {db_path}")
    print("=" * 65)
    print("You can now inspect the ledger report by running:")
    print("  python scripts/ledger_report.py --repo-dir .")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
