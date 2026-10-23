#!/usr/bin/env python3
"""Smoke test script for Laya Secondary (Shadow) Judge.

Evaluates a synthetic subgoal and diff using the configured Laya model
and prints the verdict, probability, reason, and latency.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repo's src directory is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from jev.laya import DEFAULT_LAYA_MODEL, LayaConfig, LayaShadowJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for Laya secondary (shadow) judge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Laya model identifier (defaults to JEV_LAYA_MODEL or {DEFAULT_LAYA_MODEL})",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = LayaConfig.from_env_and_args(
        cli_laya="on",
        cli_model=args.model,
    )

    print(f"Initializing Laya Shadow Judge with model: {config.model}")
    judge = LayaShadowJudge(config=config)

    print("Evaluating synthetic subgoal diff...")
    subgoal_description = "Add basic input validation to calculate_total function"
    subgoal_scope = ["src/calc.py"]
    diff = (
        "--- a/src/calc.py\n"
        "+++ b/src/calc.py\n"
        "@@ -5,3 +5,5 @@\n"
        " def calculate_total(items: list[float]) -> float:\n"
        "+    if not items:\n"
        "+        return 0.0\n"
        "     return sum(items)\n"
    )
    notes = "Project uses Python 3.11 with strict typing."
    mech_detail = "Syntax OK, all 3 unit tests passed."

    t0 = time.perf_counter()
    result = judge.evaluate(
        subgoal_description=subgoal_description,
        subgoal_scope=subgoal_scope,
        diff=diff,
        investigation_notes=notes,
        mechanical_detail=mech_detail,
    )
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    print("=" * 60)
    print("LAYA SHADOW EVALUATION RESULT")
    print("=" * 60)
    print(f"Verdict:       {result.verdict}")
    print(f"Probability:   {result.probability}")
    print(f"Reason:        {result.reason}")
    print(f"Latency:       {result.latency_ms or duration_ms:.1f} ms")
    if result.error:
        print(f"Error:         {result.error}")
    print("=" * 60)

    if result.error:
        print(f"Note: Completed with warning/error (fail-open): {result.error}")
        return 0

    print("Laya smoke test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
