import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import dotenv_values

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from jev.dependency_delta import format_dependency_summary
from jev.engine import JevEngine
from jev.evidence import format_evidence_summary
from jev.gatekeeper import Gatekeeper
from jev.ledger import NullRecorder, SqliteLedger
from jev.qwen_postmortem import QwenClient, QwenConfig
from jev.workspace import Workspace


import uuid
from datetime import datetime, timezone


DEFAULT_WORKER_MODEL = "gemini-3.5-flash-lite"


def generate_default_thread_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rnd = uuid.uuid4().hex[:4]
    return f"run-{ts}-{rnd}"


def positive_int(value: str) -> int:
    try:
        val = int(value)
        if val <= 0:
            raise ValueError()
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"'{value}' is not a positive integer")
    return val


def non_empty_str(value: str) -> str:
    if value is None or not value.strip():
        raise argparse.ArgumentTypeError("Model name cannot be empty or whitespace-only")
    return value.strip()


def resolve_worker_model(cli_model: Optional[str] = None) -> str:
    if cli_model is not None:
        if not cli_model.strip():
            raise argparse.ArgumentTypeError("Model name cannot be empty or whitespace-only")
        return cli_model.strip()
    env_model = os.environ.get("JEV_WORKER_MODEL")
    if env_model is not None:
        if not env_model.strip():
            raise argparse.ArgumentTypeError("JEV_WORKER_MODEL cannot be empty or whitespace-only")
        return env_model.strip()
    return DEFAULT_WORKER_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jev State Engine: Deterministic State Machine with Two-Tier Gating"
    )
    parser.add_argument(
        "ticket",
        type=str,
        help="The ticket or task description for the engine to execute",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Path to the repository to execute within (defaults to current working directory)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="checkpoints.db",
        help="Path to the SQLite checkpoint database file (defaults to checkpoints.db)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=generate_default_thread_id(),
        help="Execution thread identifier for checkpoint persistence (defaults to run-<utc-timestamp>-<4 hex>)",
    )
    parser.add_argument(
        "--test-policy",
        type=str,
        choices=["auto", "verify-only", "never", "always"],
        default="auto",
        help="Execution policy for repository test suites (auto, verify-only, never, always; defaults to auto)",
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        default=False,
        help="Disable recording to SQLite ledger",
    )
    parser.add_argument(
        "--retry-intel",
        choices=["on", "off"],
        default="on",
        help="Enable or disable retry intelligence (dead-end block, duplicate-diff guard, typed semantic feedback). Default: on.",
    )
    parser.add_argument(
        "--repo-profile",
        choices=["on", "off"],
        default="off",
        help="Inject repository profile into plan and implement prompts (on, off; defaults to off).",
    )
    parser.add_argument(
        "--test-timeout",
        type=positive_int,
        default=None,
        help="Timeout in seconds for test suite execution (positive integer; defaults to JEV_TEST_TIMEOUT or 600).",
    )
    parser.add_argument(
        "--dep-delta-jev",
        choices=["on", "off"],
        default="off",
        help="Send dependency delta to Jev as state evidence (on, off; defaults to off).",
    )
    parser.add_argument(
        "--model",
        type=non_empty_str,
        default=None,
        help="Worker LLM model name (defaults to JEV_WORKER_MODEL or gemini-3.5-flash-lite).",
    )
    parser.add_argument(
        "--qwen-postmortem",
        choices=["on", "off"],
        default=None,
        help="Enable Qwen advisory post-mortem on strikes (on, off; defaults to JEV_QWEN_POSTMORTEM or off).",
    )
    parser.add_argument(
        "--qwen-url",
        type=str,
        default=None,
        help="llama-server URL (defaults to JEV_QWEN_URL or http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=None,
        help="Model name sent in request (defaults to JEV_QWEN_MODEL or whatever server reports).",
    )
    parser.add_argument(
        "--qwen-timeout",
        type=positive_int,
        default=None,
        help="Timeout in seconds for Qwen calls (defaults to JEV_QWEN_TIMEOUT or 120).",
    )
    parser.add_argument(
        "--qwen-max-diff-chars",
        type=positive_int,
        default=None,
        help="Max diff chars sent to Qwen (defaults to JEV_QWEN_MAX_DIFF_CHARS or 24000).",
    )
    parser.add_argument(
        "--qwen-feedback",
        choices=["on", "off"],
        default=None,
        help="Feed Qwen advice back into worker retry prompt (on, off; defaults to JEV_QWEN_FEEDBACK or off).",
    )
    return parser



def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model = resolve_worker_model(args.model)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    repo_dir = Path(args.repo_dir).resolve() if args.repo_dir else Path.cwd().resolve()

    # Wire Dependency Injection
    workspace = Workspace(repo_dir=repo_dir, test_timeout=args.test_timeout)
    engine_env = Path.cwd() / ".env"
    if not engine_env.exists():
        engine_env = Path(__file__).resolve().parent.parent / ".env"
    env_file = engine_env if engine_env.exists() else ((repo_dir / ".env") if (repo_dir / ".env").exists() else None)
    gatekeeper = Gatekeeper(env_file=env_file)

    if args.no_ledger:
        recorder = NullRecorder()
    else:
        try:
            recorder = SqliteLedger(repo_dir=repo_dir)
        except Exception as e:
            print(f"ledger disabled: {e}", file=sys.stderr, flush=True)
            recorder = NullRecorder()

    # Wire Qwen advisory post-mortem client
    qwen_config = QwenConfig.from_env_and_args(
        cli_postmortem=args.qwen_postmortem,
        cli_url=args.qwen_url,
        cli_model=args.qwen_model,
        cli_timeout=args.qwen_timeout,
        cli_max_diff_chars=args.qwen_max_diff_chars,
        cli_feedback=args.qwen_feedback,
    )
    qwen_client = QwenClient(config=qwen_config)
    if qwen_client.enabled:
        qwen_client.check_health()

    # Wire Gemini LLM via langchain-google-genai
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key and env_file and env_file.exists():
        parsed_env = dotenv_values(env_file)
        google_api_key = parsed_env.get("GOOGLE_API_KEY")

    llm = None
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_api_key,
            )
        except Exception as e:
            print(f"[WARNING] Could not initialize ChatGoogleGenerativeAI: {e}")
            llm = None

    engine = JevEngine(
        workspace=workspace,
        gatekeeper=gatekeeper,
        llm=llm,
        recorder=recorder,
        db_path=args.db_path,
        test_policy=args.test_policy,
        retry_intel=args.retry_intel,
        repo_profile=args.repo_profile,
        dep_delta_jev=args.dep_delta_jev,
        qwen_client=qwen_client,
    )


    final_state = engine.execute(ticket=args.ticket, thread_id=args.thread_id)

    status = final_state.get("status")
    gate_status = final_state.get("gate_status")

    target_run_id = final_state.get("run_id") or args.thread_id

    ev_level = None
    if hasattr(recorder, "get_run_evidence_level"):
        ev_level = recorder.get_run_evidence_level(target_run_id)
    print(format_evidence_summary(ev_level, test_policy=args.test_policy))

    dep_delta_json = None
    if hasattr(recorder, "get_run_dependency_delta"):
        dep_delta_json = recorder.get_run_dependency_delta(target_run_id)
    print(format_dependency_summary(dep_delta_json))

    if status == "escalated":
        print(f"[ESCALATION] Engine halted. Triggering status: {gate_status}")
        return 1
    elif status != "completed":
        print(f"[FAILURE] Engine halted with status: {status}, gate_status: {gate_status}")
        return 1

    print(f"[SUCCESS] Ticket executed successfully. Final status: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
