#!/usr/bin/env python3
"""Export all worker, judge, and advisory model data from the SQLite run ledger
into JSONL dataset format for model fine-tuning and evaluation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jev.ledger import compute_repo_key, get_jev_home


def export_dataset(
    db_path: Path,
    output_path: Path,
    include_advisory_only: bool = False,
) -> int:
    """Read attempts joined with runs and subgoals, and export as JSONL."""
    if not db_path.exists():
        print(f"Error: Ledger database not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT 
            r.run_id,
            r.thread_id,
            r.ticket,
            r.worker_model,
            r.test_policy,
            r.started_at,
            r.ended_at,
            r.status,
            s.subgoal_id,
            s.description AS subgoal_description,
            s.scope_json AS subgoal_scope,
            s.expects_tests,
            a.attempt_no,
            a.tier,
            a.outcome,
            a.failed_check,
            a.diff AS diff_patch,
            a.diff_sha,
            a.files_json,
            a.compile_errors_json,
            a.tests_ran,
            a.test_outcome,
            a.jev_valid,
            a.jev_probability,
            a.jev_model,
            a.jev_raw,
            a.jev_tokens_in,
            a.jev_tokens_out,
            a.feedback_sent,
            a.tokens_in AS worker_tokens_in,
            a.tokens_out AS worker_tokens_out,
            a.qwen_postmortem,
            a.qwen_latency_ms,
            a.qwen_error,
            a.created_at AS attempt_created_at
        FROM attempts a
        LEFT JOIN subgoals s ON a.run_id = s.run_id AND a.subgoal_id = s.subgoal_id
        LEFT JOIN runs r ON a.run_id = r.run_id
        ORDER BY a.run_id, a.subgoal_id, a.attempt_no
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            qwen_raw = row["qwen_postmortem"]
            if include_advisory_only and not qwen_raw:
                continue

            qwen_parsed = None
            if qwen_raw:
                try:
                    qwen_parsed = json.loads(qwen_raw)
                except Exception:
                    qwen_parsed = qwen_raw

            scope_list = []
            if row["subgoal_scope"]:
                try:
                    scope_list = json.loads(row["subgoal_scope"])
                except Exception:
                    scope_list = [row["subgoal_scope"]]

            diff_str = row["diff_patch"] or ""
            diff_size = len(diff_str.encode("utf-8"))

            record: Dict[str, Any] = {
                "metadata": {
                    "run_id": row["run_id"],
                    "thread_id": row["thread_id"],
                    "ticket": row["ticket"],
                    "subgoal_id": row["subgoal_id"],
                    "subgoal_description": row["subgoal_description"],
                    "subgoal_scope": scope_list,
                    "attempt_number": row["attempt_no"],
                    "timestamp": row["attempt_created_at"],
                },
                "worker_model": {
                    "model": row["worker_model"],
                    "diff_patch": diff_str,
                    "diff_size_bytes": diff_size,
                    "diff_sha256": row["diff_sha"],
                    "tokens_in": row["worker_tokens_in"],
                    "tokens_out": row["worker_tokens_out"],
                    "feedback_received": row["feedback_sent"],
                },
                "judge_model": {
                    "tier": row["tier"],
                    "outcome": row["outcome"],
                    "failed_check": row["failed_check"],
                    "compile_errors": row["compile_errors_json"],
                    "tests_ran": bool(row["tests_ran"]),
                    "test_outcome": row["test_outcome"],
                    "jev_verdict": bool(row["jev_valid"]) if row["jev_valid"] is not None else None,
                    "jev_probability": row["jev_probability"],
                    "jev_model": row["jev_model"],
                    "jev_raw": row["jev_raw"],
                    "jev_tokens_in": row["jev_tokens_in"],
                    "jev_tokens_out": row["jev_tokens_out"],
                },
                "advisory_model": {
                    "postmortem": qwen_parsed,
                    "latency_ms": row["qwen_latency_ms"],
                    "error": row["qwen_error"],
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Exported {count} model interaction records to {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export worker, judge, and advisory data from SQLite ledger for fine-tuning."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="Target repository directory (to locate default ledger.db).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Direct path to ledger.db (overrides --repo-dir).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("finetuning_dataset.jsonl"),
        help="Output JSONL file path (defaults to finetuning_dataset.jsonl).",
    )
    parser.add_argument(
        "--advisory-only",
        action="store_true",
        help="Export only attempts that contain advisory Qwen post-mortems.",
    )

    args = parser.parse_args()

    if args.db_path:
        db_path = args.db_path
    else:
        repo_key = compute_repo_key(args.repo_dir)
        db_path = get_jev_home() / "repos" / repo_key / "ledger.db"

    return export_dataset(db_path, args.output, include_advisory_only=args.advisory_only)


if __name__ == "__main__":
    sys.exit(main())
