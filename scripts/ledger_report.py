#!/usr/bin/env python3
"""Read-only viewer for Jev State Engine run ledgers."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repo's src directory is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from jev.ledger import compute_repo_key, get_jev_home


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print plain-text report of runs, subgoals, and attempts from Jev ledger"
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Path to repository to inspect",
    )
    parser.add_argument(
        "--repo-key",
        type=str,
        default=None,
        help="16-hex repo key to inspect",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Direct path to SQLite ledger database",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Run ID to report on (defaults to latest)",
    )
    return parser


def resolve_db_path(args: argparse.Namespace) -> Path:
    if args.db_path:
        return Path(args.db_path).resolve()
    repo_key = args.repo_key
    if not repo_key and args.repo_dir:
        repo_key = compute_repo_key(args.repo_dir)
    if not repo_key:
        repo_key = compute_repo_key(Path.cwd())
    return get_jev_home() / "repos" / repo_key / "ledger.db"


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = resolve_db_path(args)
    if not db_path.exists():
        print(f"Error: Ledger database not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    run_id = args.run_id
    if run_id == "latest":
        row = cursor.execute("SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        if not row:
            print("No runs found in ledger database.")
            return 0
        run_id = row["run_id"]

    run = cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if not run:
        print(f"Error: Run '{run_id}' not found in ledger.")
        return 1

    print("=" * 70)
    print(f"RUN: {run['run_id']}")
    print("=" * 70)
    print(f"Ticket:             {run['ticket']}")
    print(f"Thread ID:          {run['thread_id']}")
    print(f"Status:             {run['status']}")
    print(f"Base Commit:        {run['base_commit']}")
    print(f"Final Commit:       {run['final_commit']}")
    print(f"Engine Commit:      {run['engine_commit']}")
    print(f"Worker Model:       {run['worker_model']}")
    print(f"Test Policy:        {run['test_policy']}")
    print(f"Started At:         {run['started_at']}")
    print(f"Ended At:           {run['ended_at']}")
    print(f"Notes Chars:        {run['notes_chars']}")
    print()

    print("-" * 70)
    print("SUBGOALS")
    print("-" * 70)
    subgoals = cursor.execute(
        "SELECT * FROM subgoals WHERE run_id = ? ORDER BY ordinal", (run_id,)
    ).fetchall()
    if not subgoals:
        print("  (No subgoals recorded)")
    for sg in subgoals:
        print(f"[{sg['subgoal_id']}] ({sg['status']}) {sg['description']}")
        print(f"  Scope:            {sg['scope_json']}")
        print(f"  Parent SHA:       {sg['parent_sha']}")
        print(f"  Commit SHA:       {sg['commit_sha']}")
        print()

    print("-" * 70)
    print("ATTEMPTS")
    print("-" * 70)
    attempts = cursor.execute(
        "SELECT * FROM attempts WHERE run_id = ? ORDER BY subgoal_id, attempt_no",
        (run_id,),
    ).fetchall()
    if not attempts:
        print("  (No attempts recorded)")
    for att in attempts:
        diff_str = att["diff"] or ""
        diff_size = len(diff_str)
        print(
            f"[{att['subgoal_id']} - Attempt {att['attempt_no']}] Tier: {att['tier']} | "
            f"Outcome: {att['outcome']} | Failed Check: {att['failed_check']}"
        )
        print(f"  Diff Size:        {diff_size} bytes (SHA: {att['diff_sha']})")
        print(f"  Files:            {att['files_json']}")
        print(
            f"  Jev Valid:        {att['jev_valid']} | Probability: {att['jev_probability']} | Model: {att['jev_model']}"
        )
        print(f"  Tests Ran:        {bool(att['tests_ran'])} | Test Outcome: {att['test_outcome']}")
        if att["feedback_sent"]:
            preview = att["feedback_sent"].strip().replace("\n", " ")[:120]
            print(f"  Feedback:         {preview}")
        if "qwen_postmortem" in att.keys() and att["qwen_postmortem"]:
            pm_data = None
            try:
                pm_data = json.loads(att["qwen_postmortem"])
            except Exception:
                pass
            if isinstance(pm_data, dict):
                conf = pm_data.get("confidence", "unknown")
                rc = pm_data.get("root_cause", "")
                lat = att["qwen_latency_ms"] if "qwen_latency_ms" in att.keys() and att["qwen_latency_ms"] is not None else pm_data.get("latency_ms")
                lat_str = f" ({lat:.1f}ms)" if lat is not None else ""
                print(f"  Qwen Post-Mortem: Confidence: {conf}{lat_str} | Root Cause: {rc}")
            else:
                raw_preview = str(att["qwen_postmortem"]).strip().replace("\n", " ")[:120]
                print(f"  Qwen Post-Mortem: {raw_preview}")
        if "qwen_error" in att.keys() and att["qwen_error"]:
            print(f"  Qwen Error:       {att['qwen_error']}")
        print()

    print("-" * 70)
    print("POST-MORTEMS")
    print("-" * 70)
    postmortems = [att for att in attempts if "qwen_postmortem" in att.keys() and att["qwen_postmortem"]]
    if not postmortems:
        print("  (No post-mortems recorded)")
    for att in postmortems:
        pm_data = None
        try:
            pm_data = json.loads(att["qwen_postmortem"])
        except Exception:
            pass
        if isinstance(pm_data, dict):
            conf = pm_data.get("confidence", "unknown")
            rc = pm_data.get("root_cause", "")
            lat = att["qwen_latency_ms"] if "qwen_latency_ms" in att.keys() and att["qwen_latency_ms"] is not None else pm_data.get("latency_ms")
            lat_str = f" ({lat:.1f}ms)" if lat is not None else ""
            print(f"[{att['subgoal_id']} - Attempt {att['attempt_no']}] Confidence: {conf}{lat_str} | Root Cause: {rc}")
        else:
            raw_preview = str(att["qwen_postmortem"]).strip().replace("\n", " ")[:120]
            print(f"[{att['subgoal_id']} - Attempt {att['attempt_no']}] {raw_preview}")
    print()

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
