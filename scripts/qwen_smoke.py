#!/usr/bin/env python3
"""Live smoke test script for Qwen post-mortem via local llama-server.

Sends a fake strike rejection to a running llama-server instance
and prints the structured advisory response.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo's src directory is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from jev.qwen_postmortem import QwenClient, QwenConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test running llama-server with Qwen advisory post-mortem"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="llama-server URL (defaults to JEV_QWEN_URL or http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name sent in request (defaults to JEV_QWEN_MODEL or server-reported model)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout seconds (defaults to JEV_QWEN_TIMEOUT or 120.0)",
    )
    parser.add_argument(
        "--feedback",
        choices=["on", "off"],
        default="off",
        help="Test advisory feedback formatting (on, off; defaults to off)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = QwenConfig.from_env_and_args(
        cli_postmortem="on",
        cli_url=args.url,
        cli_model=args.model,
        cli_timeout=args.timeout,
        cli_feedback=args.feedback,
    )

    print(f"Connecting to llama-server at: {config.url}")
    client = QwenClient(config=config)

    print("Checking server health...")
    healthy = client.check_health()
    if not healthy:
        print(
            f"ERROR: Server at {config.url} is unreachable or not healthy. "
            "Please ensure llama-server is running.",
            file=sys.stderr,
        )
        return 1

    print(f"Health check passed! Active model: {client.model}")
    print("-" * 60)
    print("Sending synthetic rejection to Qwen...")

    ticket = "Fix pagination calculation in user listing API."
    rejecting_source = "Tier 0 mechanical check: tests"
    rejection_reason = (
        "FAILED tests/test_pagination.py::test_total_pages - AssertionError: 9 != 10\n"
        "E       AssertionError: Expected 10 pages for 100 items with page size 10, got 9."
    )
    diff = (
        "--- a/src/pagination.py\n"
        "+++ b/src/pagination.py\n"
        "@@ -15,3 +15,3 @@\n"
        " def get_total_pages(total_items: int, page_size: int) -> int:\n"
        "-    return (total_items + page_size - 1) // page_size\n"
        "+    return (total_items // page_size) - 1\n"
    )

    result = client.run_postmortem(
        ticket=ticket,
        rejecting_source=rejecting_source,
        rejection_reason=rejection_reason,
        jev_verdict=None,
        diff=diff,
        strike_number="1 of 3",
        previous_strikes=None,
        thread_id="smoke-test-thread",
        subgoal_id="sg-smoke",
        strike_idx=1,
    )

    if not result:
        print("ERROR: Post-mortem returned None (client disabled).", file=sys.stderr)
        return 1

    if result.error:
        print(f"WARNING: Qwen completed with error: {result.error}", file=sys.stderr)

    print("=" * 60)
    print(f"Confidence:       {result.confidence}")
    print(f"Latency:          {result.latency_ms:.1f} ms")
    print(f"Fallback Mode:    {result.parse_fallback}")
    print("-" * 60)
    print(f"Root Cause:\n  {result.root_cause}")
    print("-" * 60)
    print(f"Where It Went Wrong:\n  {result.where_it_went_wrong}")
    print("-" * 60)
    print(f"Suggested Fix:\n  {result.suggested_fix}")
    print("-" * 60)
    print(f"Chain of Thought:\n  {result.chain_of_thought}")

    if client.feedback:
        print("-" * 60)
        feedback_block = client.format_advisory_feedback(result)
        print(f"Advisory Feedback Block Appended to Worker:\n{feedback_block}")

    print("=" * 60)
    print("Smoke test successfully executed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
