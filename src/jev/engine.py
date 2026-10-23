import hashlib
import inspect
import json
import logging
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from jev.dependency_delta import compute_dependency_delta
from jev.evidence import evidence_level
from jev.ledger import NullRecorder, Recorder

from jev.models import (
    CompileOutcome,
    MechanicalCheckResult,
    State,
    Subgoal,
    TestOutcome,
    TestPolicy,
    ValidationVerdict,
)
from jev.workspace import (
    MAX_READ_CHARS,
    MainDivergedError,
    TrackedModificationsError,
    Workspace,
)

MAX_IMPLEMENT_FILE_CHARS = 80_000


def _cap_read_text(text: str) -> str:
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n[truncated: showing first {MAX_READ_CHARS} of {len(text)} characters]"
    return text

logger = logging.getLogger(__name__)


STATE_CUSTOM_TYPES = [
    Subgoal,
    MechanicalCheckResult,
    ValidationVerdict,
    TestOutcome,
    CompileOutcome,
    TestPolicy,
]


class SafeRecorder(NullRecorder):
    """Fail-safe proxy wrapper around a Recorder.

    Recording must never change a run's outcome. Wrap every ledger call.
    On any error, log one warning per run ("ledger disabled for this run: <error>") and continue.
    """

    def __init__(self, target: Any):
        self.target = target
        self.disabled = False

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self.disabled or self.target is None:
            return None
        try:
            fn = getattr(self.target, name, None)
            if fn:
                return fn(*args, **kwargs)
        except Exception as exc:
            self.disabled = True
            msg = f"ledger disabled for this run: {exc}"
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)
            return None

    def record_run_start(self, *args: Any, **kwargs: Any) -> None:
        self._call("record_run_start", *args, **kwargs)

    def record_run_end(self, *args: Any, **kwargs: Any) -> None:
        self._call("record_run_end", *args, **kwargs)

    def record_subgoals(self, *args: Any, **kwargs: Any) -> None:
        self._call("record_subgoals", *args, **kwargs)

    def record_subgoal_committed(self, *args: Any, **kwargs: Any) -> None:
        self._call("record_subgoal_committed", *args, **kwargs)

    def record_subgoal_status(self, *args: Any, **kwargs: Any) -> None:
        self._call("record_subgoal_status", *args, **kwargs)

    def record_attempt(self, *args: Any, **kwargs: Any) -> None:
        self._call("record_attempt", *args, **kwargs)

    def has_run(self, *args: Any, **kwargs: Any) -> bool:
        res = self._call("has_run", *args, **kwargs)
        return bool(res)

    def get_run_evidence_level(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return self._call("get_run_evidence_level", *args, **kwargs)

    def get_run_dependency_delta(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return self._call("get_run_dependency_delta", *args, **kwargs)


def _get_engine_commit() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def _normalize_compile_errors(compile_info: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not compile_info or not isinstance(compile_info, dict):
        return []
    new_errors = compile_info.get("new_errors") or []
    if not new_errors:
        return []
    normalized: List[Dict[str, Any]] = []
    try:
        diag_objs = Workspace.parse_diagnostics("\n".join(str(e) for e in new_errors))
        for d in diag_objs:
            normalized.append({"file": d.file, "code": d.code, "message": d.message})
    except Exception:
        pass
    if not normalized:
        for err in new_errors:
            normalized.append({"file": "", "code": "", "message": str(err)})
    return normalized


def _extract_diff_excerpt(diff: Any, max_lines: int = 40, max_chars: int = 3000) -> str:
    if not isinstance(diff, str) or not diff:
        return ""
    lines = diff.splitlines(keepends=True)
    excerpt_lines = lines[:max_lines]
    excerpt = "".join(excerpt_lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    return excerpt


def _record_failed_attempt_in_state(
    state: State,
    tier: str,
    outcome: str,
    failed_check: Optional[str],
    diff: Any,
    compile_info: Optional[Dict[str, Any]] = None,
) -> None:
    if "failed_attempts" not in state or state["failed_attempts"] is None:
        state["failed_attempts"] = []

    prev_attempts = state["failed_attempts"]
    attempt_no = (prev_attempts[-1].get("attempt_no", 0) + 1) if prev_attempts else 1
    if isinstance(diff, str):
        diff_bytes = diff.encode("utf-8")
    elif isinstance(diff, bytes):
        diff_bytes = diff
    else:
        diff_bytes = b""
    diff_sha = hashlib.sha256(diff_bytes).hexdigest()
    diff_excerpt = _extract_diff_excerpt(diff)

    norm_errors = _normalize_compile_errors(compile_info)
    norm_error_keys: List[str] = []
    for err in norm_errors:
        key = err.get("code") or err.get("message") or ""
        if err.get("file"):
            key = f"{err['file']}:{key}"
        norm_error_keys.append(key)

    entry = {
        "attempt_no": attempt_no,
        "tier": tier,
        "failed_check": failed_check,
        "outcome": outcome,
        "normalized_error_keys": norm_error_keys,
        "diff_sha": diff_sha,
        "diff_excerpt": diff_excerpt,
    }

    state["failed_attempts"].append(entry)
    if len(state["failed_attempts"]) > 2:
        state["failed_attempts"] = state["failed_attempts"][-2:]


def _format_typed_semantic_feedback(
    verdict: ValidationVerdict,
    subgoal: Subgoal,
    diff: str,
    tests_ran: bool,
) -> str:
    lines = ["Semantic validation rejected by Gatekeeper."]
    if verdict.kind:
        if verdict.kind == "fine":
            prob_str = f" (validity probability: {verdict.probability:.2f}, pass needs 0.50)" if getattr(verdict, "probability", None) is not None else ""
            lines.append(f"No specific issue identified by the judge{prob_str}")
        else:
            conf_str = f" (confidence: {verdict.kind_confidence:.2f})" if verdict.kind_confidence is not None else ""
            lines.append(f"Main issue: {verdict.kind}{conf_str}")

    flagged = []
    if verdict.incomplete is not None and verdict.incomplete > 0.5:
        flagged.append(f"Incomplete implementation (probability: {verdict.incomplete:.2f})")
    if verdict.wrong_behavior is not None and verdict.wrong_behavior > 0.5:
        flagged.append(f"Wrong behavior contradicting subgoal or ticket (probability: {verdict.wrong_behavior:.2f})")
    if verdict.conflicts_with_notes is not None and verdict.conflicts_with_notes > 0.5:
        flagged.append(f"Conflicts with investigation notes or conventions (probability: {verdict.conflicts_with_notes:.2f})")
    if flagged:
        lines.append("Flagged issues:")
        for item in flagged:
            lines.append(f"- {item}")

    # Deterministic facts
    files = []
    if diff:
        try:
            files = sorted(list(Workspace._extract_files_from_diff(diff)))
        except Exception:
            files = []
    scope_str = ", ".join(subgoal.scope) if subgoal.scope else "none"
    files_str = ", ".join(files) if files else "none"
    lines.append(f"Files touched: {files_str} (Declared scope: {scope_str})")

    added = 0
    removed = 0
    if diff:
        for dline in diff.splitlines():
            if dline.startswith("+") and not dline.startswith("+++"):
                added += 1
            elif dline.startswith("-") and not dline.startswith("---"):
                removed += 1
    lines.append(f"Lines changed: +{added}, -{removed}")
    lines.append(f"Tests ran: {'Yes' if tests_ran else 'No'}")

    return "\n".join(lines)




class SqliteSaver(BaseCheckpointSaver):
    """SQLite-backed LangGraph checkpointer for persisting FSM state and strike counters."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        if serde is None:
            serde = JsonPlusSerializer(
                allowed_msgpack_modules=STATE_CUSTOM_TYPES,
                allowed_json_modules=STATE_CUSTOM_TYPES,
            )
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = threading.RLock()
        self._setup()

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        with self.lock:
            cur = self.conn.cursor()
            try:
                yield cur
                if transaction:
                    self.conn.commit()
            except Exception:
                if transaction:
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                try:
                    cur.close()
                except Exception:
                    pass

    def _setup(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode = WAL")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT,
                        checkpoint_ns TEXT,
                        checkpoint_id TEXT,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint TEXT,
                        metadata TEXT,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoint_writes (
                        thread_id TEXT,
                        checkpoint_ns TEXT,
                        checkpoint_id TEXT,
                        task_id TEXT,
                        idx INTEGER,
                        channel TEXT,
                        type TEXT,
                        blob TEXT,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                """)
                self.conn.commit()
            finally:
                cur.close()

    @classmethod
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> "SqliteSaver":
        if conn_string != ":memory:" and not conn_string.startswith("file:"):
            Path(conn_string).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(conn_string, check_same_thread=False)
        return cls(conn, serde=serde)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns") or ""
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id:
            return None

        with self.cursor(transaction=False) as cursor:
            if checkpoint_id:
                cursor.execute(
                    """
                    SELECT parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (str(thread_id), str(checkpoint_ns), str(checkpoint_id)),
                )
            else:
                cursor.execute(
                    """
                    SELECT parent_checkpoint_id, type, checkpoint, metadata, checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (str(thread_id), str(checkpoint_ns)),
                )

            row = cursor.fetchone()
            if not row:
                return None

            if checkpoint_id:
                parent_id, cp_type, cp_data, md_data = row
                ret_id = checkpoint_id
            else:
                parent_id, cp_type, cp_data, md_data, ret_id = row

            raw_bytes = cp_data if isinstance(cp_data, bytes) else (cp_data.encode("utf-8") if isinstance(cp_data, str) else cp_data)
            checkpoint = self.serde.loads_typed((cp_type, raw_bytes))
            metadata = json.loads(md_data) if md_data else {}

            cursor.execute(
                """
                SELECT task_id, channel, type, blob
                FROM checkpoint_writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ORDER BY task_id, idx
                """,
                (str(thread_id), str(checkpoint_ns), str(ret_id)),
            )
            writes = []
            for task_id, channel, w_type, w_blob in cursor.fetchall():
                w_bytes = w_blob if isinstance(w_blob, bytes) else (w_blob.encode("utf-8") if isinstance(w_blob, str) else w_blob)
                val = self.serde.loads_typed((w_type, w_bytes))
                writes.append((task_id, channel, val))

        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": ret_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=writes,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        if checkpoint_ns is None and config and "checkpoint_ns" in configurable:
            checkpoint_ns = ""
        query = "SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints WHERE 1=1"
        params: List[Any] = []
        if thread_id:
            query += " AND thread_id = ?"
            params.append(str(thread_id))
        if checkpoint_ns is not None:
            query += " AND checkpoint_ns = ?"
            params.append(str(checkpoint_ns))
        if before and before.get("configurable", {}).get("checkpoint_id"):
            query += " AND checkpoint_id < ?"
            params.append(str(before["configurable"]["checkpoint_id"]))
        query += " ORDER BY rowid DESC"
        if limit:
            query += f" LIMIT {limit}"

        with self.cursor(transaction=False) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        for t_id, c_ns, c_id in rows:
            tup = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": t_id,
                        "checkpoint_ns": c_ns,
                        "checkpoint_id": c_id,
                    }
                }
            )
            if tup:
                if filter:
                    if not all(tup.metadata.get(k) == v for k, v in filter.items()):
                        continue
                yield tup

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns") or ""
        parent_id = configurable.get("checkpoint_id")
        checkpoint_id = checkpoint["id"]

        cp_type, cp_blob = self.serde.dumps_typed(checkpoint)
        blob_bytes = bytes(cp_blob) if isinstance(cp_blob, (bytes, bytearray, memoryview)) else (cp_blob.encode("utf-8") if isinstance(cp_blob, str) else cp_blob)
        md_str = json.dumps(metadata)

        with self.cursor(transaction=True) as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(thread_id),
                    str(checkpoint_ns),
                    str(checkpoint_id),
                    str(parent_id) if parent_id else None,
                    str(cp_type),
                    blob_bytes,
                    md_str,
                ),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns") or ""
        checkpoint_id = configurable.get("checkpoint_id", "")

        query = (
            """
            INSERT OR REPLACE INTO checkpoint_writes
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else """
            INSERT OR IGNORE INTO checkpoint_writes
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        params = []
        for idx, (channel, val) in enumerate(writes):
            w_type, w_blob = self.serde.dumps_typed(val)
            blob_bytes = bytes(w_blob) if isinstance(w_blob, (bytes, bytearray, memoryview)) else (w_blob.encode("utf-8") if isinstance(w_blob, str) else w_blob)
            idx_val = WRITES_IDX_MAP.get(channel, idx)
            params.append(
                (
                    str(thread_id),
                    str(checkpoint_ns),
                    str(checkpoint_id),
                    str(task_id),
                    int(idx_val),
                    str(channel),
                    str(w_type),
                    blob_bytes,
                )
            )

        with self.cursor(transaction=True) as cur:
            cur.executemany(query, params)


def _accepts_param(fn: Any, param_name: str) -> bool:
    try:
        target = getattr(fn, "side_effect", None)
        if target is None or not callable(target):
            target = fn
        sig = inspect.signature(target)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        return (param_name in sig.parameters) or has_kwargs
    except Exception:
        return False


def _safe_escalate_deadlock(
    gatekeeper: Any,
    trajectory: List[Dict[str, Any]],
    triggering_tier: str,
    integration_branch: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> None:
    if gatekeeper is None or not hasattr(gatekeeper, "escalate_deadlock"):
        return
    from jev.gatekeeper import Gatekeeper
    if isinstance(gatekeeper, Gatekeeper):
        eff_thread_id = thread_id
        if not eff_thread_id and integration_branch:
            eff_thread_id = integration_branch.replace("jev-ticket-", "")
        gatekeeper.escalate_deadlock(
            trajectory=trajectory,
            triggering_tier=triggering_tier,
            thread_id=eff_thread_id,
            integration_branch=integration_branch,
        )
    else:
        kwargs: Dict[str, Any] = {}
        target = getattr(gatekeeper.escalate_deadlock, "side_effect", None) or gatekeeper.escalate_deadlock
        try:
            sig = inspect.signature(target)
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kwargs and "integration_branch" in sig.parameters:
                if integration_branch:
                    kwargs["integration_branch"] = integration_branch
            elif has_var_kwargs and integration_branch:
                kwargs["integration_branch"] = integration_branch
        except Exception:
            if integration_branch:
                kwargs["integration_branch"] = integration_branch
        gatekeeper.escalate_deadlock(
            trajectory=trajectory,
            triggering_tier=triggering_tier,
            **kwargs,
        )


def _extract_content_text(content: Any) -> str:
    """Normalize message content to a string, handling lists of text blocks if present."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def node_investigate(
    state: State,
    workspace: Optional[Any] = None,
    llm: Optional[Any] = None,
    max_investigation_turns: int = 20,
) -> State:
    """State 1: INVESTIGATION (Read-Only access to codebase)."""
    if "trajectory" not in state or state["trajectory"] is None:
        state["trajectory"] = []

    # If no LLM is provided, maintain backwards compatibility with static fallback
    if llm is None:
        notes = ""
        if workspace is not None and hasattr(workspace, "run_read_tool"):
            git_status = workspace.run_read_tool("git", ["status", "--porcelain"])
            notes = f"Workspace status:\n{git_status}"
        else:
            notes = "No workspace read tool bound."
        state["last_feedback"] = notes
        state["investigation_notes"] = notes
        state["trajectory"].append({"node": "investigate", "notes": notes})
        return state

    finished = False
    investigation_summary = ""
    executed_tools: List[Dict[str, Any]] = []
    raw_worktree = getattr(workspace, "worktree_dir", None) or getattr(workspace, "repo_dir", None)
    workspace_root: Optional[Path] = Path(raw_worktree).resolve() if raw_worktree else None

    # Bind strictly READ-ONLY tools (State 1 constraint)
    @tool
    def list_dir(path: str = ".") -> str:
        """List files and subdirectories at the given path relative to repository root.

        Args:
            path: Directory path to list, defaults to "." (repository root).
        """
        if hasattr(workspace, "list_dir"):
            try:
                return str(workspace.list_dir(path))
            except Exception as e:
                return f"Error listing directory {path}: {e}"

        worktree = getattr(workspace, "worktree_dir", getattr(workspace, "repo_dir", Path.cwd()))
        base = Path(worktree) if worktree else Path.cwd()
        target = (base / (path or ".")).resolve()
        try:
            target.relative_to(base.resolve())
        except ValueError:
            return f"Error: Access denied for path outside workspace: {path}"

        if not target.exists():
            return f"Directory not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"

        ignore_names = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".mypy_cache"}
        entries = []
        for child in sorted(target.iterdir()):
            if child.name in ignore_names:
                continue
            entries.append(f"{child.name}/" if child.is_dir() else child.name)
        return "\n".join(entries) if entries else "(empty directory)"

    @tool
    def grep(query: str, path: Optional[str] = None) -> str:
        """Search for string patterns or regex matches across files in the workspace.

        Args:
            query: String or regex pattern to search for.
            path: Optional subdirectory or file path to limit the search.
        """
        if not query:
            return "Error: query parameter is required for grep."

        worktree = getattr(workspace, "worktree_dir", getattr(workspace, "repo_dir", Path.cwd()))
        base = Path(worktree).resolve() if worktree else Path.cwd().resolve()
        safe_rel_path = None
        search_target = base

        if path:
            target = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
            try:
                safe_rel_path = target.relative_to(base).as_posix()
            except ValueError:
                return f"Error: Access denied for path outside workspace: {path}"
            search_target = target

        if hasattr(workspace, "grep"):
            try:
                return str(workspace.grep(query, path))
            except Exception as e:
                return f"Error running grep: {e}"

        if workspace is not None and hasattr(workspace, "run_read_tool"):
            args = ["grep", "-n", "-I", "--untracked", query]
            if safe_rel_path:
                args.extend(["--", safe_rel_path])
            try:
                res = workspace.run_read_tool("git", args)
                if "fatal: not a git repository" not in res and "Command not found" not in res:
                    return res.strip() if res.strip() else "No matches found."
            except Exception:
                pass

        # Python regex search fallback
        import re
        if not search_target.exists():
            return f"Path not found: {path}"

        try:
            pattern = re.compile(query, re.IGNORECASE)
        except Exception:
            pattern = re.compile(re.escape(query))

        ignore_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".mypy_cache"}
        target_files = [search_target] if search_target.is_file() else [
            p for p in search_target.rglob("*")
            if p.is_file() and not any(part in ignore_dirs for part in p.parts)
        ]
        matches = []
        for f in target_files:
            try:
                rel = f.relative_to(base)
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
                for idx, line in enumerate(lines, 1):
                    if pattern.search(line):
                        matches.append(f"{rel}:{idx}:{line}")
                        if len(matches) >= 50:
                            break
            except Exception:
                continue
            if len(matches) >= 50:
                break
        return "\n".join(matches) if matches else "No matches found."

    @tool
    def read_file(path: str) -> str:
        """Read the full content of a file in the workspace. Backed by workspace.run_read_tool().

        Args:
            path: Relative path to the file from repository root.
        """
        if not path:
            return "Error: path is required for read_file."

        worktree = getattr(workspace, "worktree_dir", getattr(workspace, "repo_dir", None))
        if worktree:
            base = Path(worktree).resolve()
            target = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                return f"Error: Access denied for path outside workspace: {path}"

        if workspace is not None and hasattr(workspace, "run_read_tool"):
            try:
                output = workspace.run_read_tool("cat", [path])
                if output is not None and "Command not found: cat" not in output:
                    return _cap_read_text(output)
            except Exception:
                pass

        if worktree:
            if target.exists():
                if target.is_dir():
                    return f"Error: {path} is a directory, not a file. Use list_dir instead."
                try:
                    return _cap_read_text(target.read_text(encoding="utf-8", errors="replace"))
                except Exception as e:
                    return f"Error reading file {path}: {e}"
            return f"File not found: {path}"
        return f"Workspace unavailable; could not read {path}"

    @tool
    def finish_investigation(summary: str = "") -> str:
        """Signal that investigation is complete and provide a comprehensive summary of findings to inform planning.

        Args:
            summary: Comprehensive summary of code patterns, architecture, relevant files, and implementation findings.
        """
        nonlocal finished, investigation_summary
        finished = True
        investigation_summary = summary
        return "Investigation finished."

    # Bind strictly READ-ONLY tools (no write tools)
    tools = [list_dir, grep, read_file, finish_investigation]
    investigated_dirs: set[str] = set()

    ticket = state.get("ticket", "")
    prompt_lines = [
        f"Task Ticket: {ticket}",
        "",
        "Instructions:",
        "1. You have read-only access to investigate the codebase using the available tools: `list_dir`, `grep`, and `read_file`.",
        "2. Explore the file structure, find relevant files, and understand existing patterns and architecture.",
        "3. Identify the application's product identity, business domain, and core entity concepts from root metadata (e.g. package.json, layout files, README) and document them in your summary.",
        "4. When you have gathered enough context to plan the implementation, call `finish_investigation` with a comprehensive summary of your findings.",
    ]
    prompt_text = "\n".join(prompt_lines)

    messages: List[Any] = [HumanMessage(content=prompt_text)]
    bound_llm = llm.bind_tools(tools)

    max_turns = max_investigation_turns
    turn = 0

    while turn < max_turns and not finished:
        turn += 1
        response = None
        for api_attempt in range(5):
            try:
                response = bound_llm.invoke(messages)
                break
            except Exception as e:
                if api_attempt == 4:
                    raise
                sleep_time = 25 if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else (5 * (api_attempt + 1))
                time.sleep(sleep_time)

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls and isinstance(response, dict):
            tool_calls = response.get("tool_calls")

        if not tool_calls:
            if not investigation_summary:
                investigation_summary = _extract_content_text(getattr(response, "content", "")).strip()
            break

        for tool_call in tool_calls:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", f"call_{turn}")

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            executed_tools.append({"name": name, "args": args})

            if name == "finish_investigation":
                finished = True
                extracted = (
                    args.get("summary")
                    or args.get("notes")
                    or args.get("investigation_notes")
                    or args.get("findings")
                    or args.get("result")
                    or args.get("analysis")
                    or args.get("summary_text")
                    or ""
                )
                if extracted:
                    investigation_summary = _extract_content_text(extracted)
                else:
                    investigation_summary = _extract_content_text(getattr(response, "content", "")).strip()
                result = "Investigation finished."
            elif name == "read_file":
                actual_path = args.get("path") or args.get("file_path") or args.get("filename") or ""
                try:
                    result = read_file.invoke({"path": actual_path}) if hasattr(read_file, "invoke") else read_file(actual_path)
                except Exception as e:
                    result = f"Error reading file {actual_path}: {e}"

                if workspace_root is not None and actual_path:
                    try:
                        p = Path(actual_path)
                        target = (workspace_root / p).resolve() if not p.is_absolute() else p.resolve()
                        target.relative_to(workspace_root)
                        if target.is_file():
                            parent = target.parent.relative_to(workspace_root).as_posix().lstrip("./")
                            investigated_dirs.add(parent if parent else ".")
                    except Exception:
                        pass
            elif name == "list_dir":
                actual_path = args.get("path") or args.get("directory") or args.get("dir_path") or "."
                try:
                    result = list_dir.invoke({"path": actual_path}) if hasattr(list_dir, "invoke") else list_dir(actual_path)
                except Exception as e:
                    result = f"Error listing directory {actual_path}: {e}"

                if workspace_root is not None:
                    try:
                        p = Path(actual_path)
                        target = (workspace_root / p).resolve() if not p.is_absolute() else p.resolve()
                        target.relative_to(workspace_root)
                        if target.is_dir():
                            clean_dir = target.relative_to(workspace_root).as_posix().lstrip("./")
                            investigated_dirs.add(clean_dir if clean_dir else ".")
                    except Exception:
                        pass
            elif name == "grep":
                actual_query = args.get("query") or args.get("pattern") or args.get("search_term") or ""
                actual_path = args.get("path") or args.get("directory") or args.get("file_path")
                call_args = {"query": actual_query}
                if actual_path:
                    call_args["path"] = actual_path
                try:
                    result = grep.invoke(call_args) if hasattr(grep, "invoke") else grep(actual_query, actual_path)
                except Exception as e:
                    result = f"Error running grep: {e}"

                if workspace_root is not None and actual_path:
                    try:
                        p = Path(actual_path)
                        target = (workspace_root / p).resolve() if not p.is_absolute() else p.resolve()
                        target.relative_to(workspace_root)
                        if target.is_dir():
                            clean_dir = target.relative_to(workspace_root).as_posix().lstrip("./")
                            investigated_dirs.add(clean_dir if clean_dir else ".")
                        elif target.is_file():
                            parent = target.parent.relative_to(workspace_root).as_posix().lstrip("./")
                            investigated_dirs.add(parent if parent else ".")
                    except Exception:
                        pass
            else:
                result = f"Error: Tool '{name}' is not permitted in investigation state. Only read-only tools (list_dir, grep, read_file, finish_investigation) are available."

            messages.append(
                ToolMessage(
                    content=str(result),
                    name=name,
                    tool_call_id=str(call_id or f"call_{turn}"),
                )
            )

        if finished:
            break

    if not investigation_summary:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                text = _extract_content_text(getattr(msg, "content", "")).strip()
                if text:
                    investigation_summary = text
                    break
        if not investigation_summary:
            investigation_summary = f"Investigation concluded after reaching maximum turns ({max_turns})."

    state["investigation_notes"] = investigation_summary
    state["last_feedback"] = investigation_summary
    state["investigated_directories"] = sorted(list(investigated_dirs))
    state["investigation_incomplete"] = not finished

    traj_entry: Dict[str, Any] = {
        "node": "investigate",
        "notes": investigation_summary,
        "investigation_incomplete": state["investigation_incomplete"],
    }
    if executed_tools:
        traj_entry["tool_calls"] = executed_tools
    state["trajectory"].append(traj_entry)
    return state



class PlanSubgoalModel(BaseModel):
    """Strict schema for validating LLM-generated Subgoal outputs."""

    model_config = ConfigDict(strict=True)

    description: str = Field(..., min_length=1)
    scope: List[str] = Field(..., min_length=1)
    expects_tests: bool = True

    @field_validator("scope")
    @classmethod
    def validate_scope_items(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Subgoal scope must not be empty.")
        cleaned_list: List[str] = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Scope items must be non-empty strings.")
            cleaned_item = item.strip().replace("\\", "/")
            path_obj = Path(cleaned_item)
            is_abs = (
                path_obj.is_absolute()
                or bool(path_obj.drive)
                or cleaned_item.startswith("/")
                or cleaned_item.startswith("\\")
            )
            if is_abs or ".." in path_obj.parts:
                raise ValueError(f"Scope item '{item}' must be a relative path within the repository.")
            if cleaned_item not in cleaned_list:
                cleaned_list.append(cleaned_item)
        return cleaned_list

    @field_validator("description")
    @classmethod
    def validate_description_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Subgoal description must be non-empty.")
        return v.strip()


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()

    # 1. Look specifically for a ```json ... ``` code fence (case-insensitive)
    match_json = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match_json:
        return match_json.group(1).strip()

    # 2. Look for any generic code fence ``` ... ```
    match_generic = re.search(r"```\s*([\s\S]*?)\s*```", cleaned)
    if match_generic:
        return match_generic.group(1).strip()

    # 3. Check for raw array bracket [ ... ] if root is not an object dict { ... }
    start_bracket = cleaned.find("[")
    end_bracket = cleaned.rfind("]")
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        start_brace = cleaned.find("{")
        if start_brace == -1 or start_brace > start_bracket:
            return cleaned[start_bracket : end_bracket + 1].strip()

    return cleaned


def _parse_and_validate_plan(raw_text: str) -> List[Subgoal]:
    """Parse raw LLM output as JSON and validate strictly against Subgoal schema."""
    cleaned = _clean_json_text(raw_text)
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError(f"Plan output must be a JSON array of Subgoal objects, got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("Plan output must contain at least one subgoal, got empty array")

    subgoals: List[Subgoal] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Subgoal at index {idx} must be a JSON object, got {type(item).__name__}")
        validated = PlanSubgoalModel.model_validate(item)
        subgoals.append(
            Subgoal(
                description=validated.description,
                scope=validated.scope,
                expects_tests=validated.expects_tests,
            )
        )
    return subgoals


def _extract_candidate_files(
    investigation_notes: str,
    base_dir: Optional[Path] = None,
) -> List[str]:
    """Extract candidate source files identified during investigation."""
    if not investigation_notes:
        return []

    candidates: List[str] = []
    non_source_extensions = {
        "md", "txt", "json", "yaml", "yml", "toml", "lock", "csv",
        "png", "jpg", "jpeg", "svg", "gif", "ico", "gitignore", "env",
    }

    # 1. Look for explicit candidate sections if present (e.g. "Candidate Files:", "Relevant Files:")
    candidate_section_match = re.search(
        r"(?:candidate|relevant|target)\s+files?[:\n](.*?)(?:\n\s*\n|\Z)",
        investigation_notes,
        re.IGNORECASE | re.DOTALL,
    )
    section_text = candidate_section_match.group(1) if candidate_section_match else ""

    # Path pattern:
    # A) relative paths with directory separators: e.g. src/lib/auth.ts
    # B) single filenames with recognized code extensions: e.g. auth.ts, core.py
    path_pattern = re.compile(
        r"(?:[a-zA-Z0-9_\-\.]+[/\\])+[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9_-]+"
        r"|\b[a-zA-Z0-9_\-]+\.(?:ts|tsx|js|jsx|py|go|rs|java|c|cpp|h|hpp|rb|php|cs|swift|kt|pyi)\b"
    )

    text_to_search = section_text if section_text.strip() else investigation_notes
    matches = path_pattern.findall(text_to_search)

    for m in matches:
        cleaned = m.strip().replace("\\", "/").rstrip(".,:;)>]'\"")
        if not cleaned:
            continue
        p = Path(cleaned)
        ext = p.suffix.lower().lstrip(".")
        if ext in non_source_extensions:
            continue

        if cleaned not in candidates:
            candidates.append(cleaned)

    return candidates


def _matches_candidate(scope_path: str, candidates: List[str], base_dir: Optional[Path] = None) -> bool:
    """Check if a scope path matches any candidate file identified during investigation."""
    cleaned = scope_path.strip().replace("\\", "/")
    p = Path(cleaned)
    p_name = p.name.lower()

    for c in candidates:
        c_cleaned = c.strip().replace("\\", "/")
        c_p = Path(c_cleaned)
        c_name = c_p.name.lower()

        # Exact match
        if cleaned == c_cleaned:
            return True
        # Resolved path match
        if base_dir is not None:
            try:
                if (base_dir / cleaned).resolve() == (base_dir / c_cleaned).resolve():
                    return True
            except Exception:
                pass
        # Relative path suffix match (e.g. "auth.ts" matches "src/lib/auth.ts")
        if cleaned.endswith(c_cleaned) or c_cleaned.endswith(cleaned):
            return True
        # Filename match (e.g. auth.ts matches auth.ts)
        if p_name == c_name:
            return True

    return False


def _segment_in_notes_path(segment: str, notes: str) -> bool:
    """Check if a directory segment appears in investigation notes as part of a path string

    (a token containing '/'), not just as an ordinary word in prose.
    """
    if not notes or not segment:
        return False
    seg_lower = segment.lower()
    tokens = re.findall(r'[^\s`"\'()<>\[\]{}]+', notes)
    for raw_token in tokens:
        cleaned = raw_token.strip(".,:;!?'\"`").replace("\\", "/")
        if "/" in cleaned:
            parts = [p.lower() for p in cleaned.split("/") if p]
            if seg_lower in parts:
                return True
    return False


def _check_single_subgoal_grounding(
    subgoal: Subgoal,
    workspace: Optional[Any],
    ticket: str,
    investigation_notes: str,
    investigated_directories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ticket_lower = ticket.lower()
    notes_lower = (investigation_notes or "").lower()
    inv_dirs = set(investigated_directories or [])

    # Determine if ticket specifically targets existing code
    targets_existing = bool(
        re.search(r"\b(existing|undocumented|current)\b", ticket_lower)
    ) or bool(
        re.search(r"\badd\s+to\b", ticket_lower)
    )

    # Determine if ticket requests creating new files or implementing new features
    creates_new = bool(
        re.search(r"\b(create|new\s+file|scaffold|implement|build|generate|add)\b", ticket_lower)
    )

    # Resolve workspace root directory if available
    base_dir: Optional[Path] = None
    if workspace is not None:
        raw_base = getattr(workspace, "worktree_dir", None) or getattr(workspace, "repo_dir", None)
        if raw_base is not None:
            base_dir = Path(raw_base).resolve()

    # Extract candidate files identified during investigation
    candidate_files = _extract_candidate_files(investigation_notes, base_dir)

    # Extract all file extensions mentioned in investigation_notes (e.g. .ts, .py, .go, .rs, .js)
    notes_extensions = set(re.findall(r"\.([a-zA-Z0-9_-]+)\b", notes_lower))

    non_source_extensions = {
        "md", "txt", "json", "yaml", "yml", "toml", "lock", "csv",
        "png", "jpg", "jpeg", "svg", "gif", "ico", "gitignore", "env",
    }

    subgoal_desc_lower = subgoal.description.lower()
    subgoal_targets_existing = targets_existing or bool(
        re.search(r"\b(existing|undocumented|current)\b", subgoal_desc_lower)
    ) or bool(
        re.search(r"\badd\s+to\b", subgoal_desc_lower)
    )

    for scope_item in subgoal.scope:
        cleaned_path = scope_item.strip().replace("\\", "/")
        path_obj = Path(cleaned_path)
        file_name = path_obj.name.lower()
        file_stem = path_obj.stem.lower()
        ext = path_obj.suffix.lower().lstrip(".")

        # Check if file path, filename, or meaningful stem is explicitly present in notes or ticket
        stem_in_notes = len(file_stem) >= 3 and bool(re.search(r"\b" + re.escape(file_stem) + r"\b", notes_lower))
        stem_in_ticket = len(file_stem) >= 3 and bool(re.search(r"\b" + re.escape(file_stem) + r"\b", ticket_lower))
        in_notes = (cleaned_path.lower() in notes_lower) or (file_name in notes_lower) or stem_in_notes
        in_ticket = (cleaned_path.lower() in ticket_lower) or (file_name in ticket_lower) or stem_in_ticket

        # 1. Check if the file exists on disk in the workspace
        exists_on_disk = False
        if base_dir is not None:
            try:
                full_target = (base_dir / cleaned_path).resolve()
                exists_on_disk = full_target.exists() and (full_target.is_file() or full_target.is_dir())
            except Exception:
                exists_on_disk = False

        # Tightened candidate check for tickets/subgoals targeting existing functionality
        if subgoal_targets_existing:
            # When investigation notes identified specific candidate files, scope MUST match one of them
            if candidate_files:
                if not _matches_candidate(cleaned_path, candidate_files, base_dir) and not in_ticket:
                    if not exists_on_disk:
                        return {
                            "status": "rejected",
                            "reason": (
                                f"Scope file '{scope_item}' does not exist in repository and was not found during investigation for ticket modifying existing code: "
                                f"scope must reference one of the files investigation identified as containing the target functionality: {sorted(candidate_files)}"
                            ),
                        }
                    else:
                        return {
                            "status": "rejected",
                            "reason": (
                                f"Scope file '{scope_item}' is invalid: for tickets targeting existing functionality, "
                                f"scope must reference one of the files investigation identified as containing the target functionality: {sorted(candidate_files)}"
                            ),
                        }

            # Non-source files (README.md, config files) cannot satisfy tickets targeting existing functions
            if ext in non_source_extensions and not in_ticket:
                return {
                    "status": "rejected",
                    "reason": f"Scope file '{scope_item}' is a non-source file ({ext}) and cannot satisfy a ticket targeting existing functions or code.",
                }

        if exists_on_disk:
            continue

        # If investigated_directories was recorded, verify architectural consistency for new files
        if inv_dirs:
            parent_dir = path_obj.parent.as_posix().lstrip("./")
            if not parent_dir:
                parent_dir = "."

            parent_dir_allowed = False
            if parent_dir == ".":
                if "." in inv_dirs or "" in inv_dirs:
                    parent_dir_allowed = True
            else:
                if parent_dir in inv_dirs:
                    parent_dir_allowed = True
                elif any(
                    parent_dir == d or parent_dir.startswith(d + "/")
                    for d in inv_dirs
                    if d not in ("", ".")
                ):
                    parent_dir_allowed = True

            if not parent_dir_allowed:
                if (
                    parent_dir.lower() in ticket_lower
                    or cleaned_path.lower() in ticket_lower
                    or _segment_in_notes_path(parent_dir, investigation_notes)
                ):
                    parent_dir_allowed = True

            if not parent_dir_allowed:
                return {
                    "status": "rejected",
                    "reason": f"Scope proposes creating files under '{parent_dir}/', but investigation only found {sorted(list(inv_dirs))} -- this repository does not use a {parent_dir} directory structure",
                }

            # New rule for new files (files that do not exist on disk), applied to every directory segment that does not already exist on disk:
            # - The new directory segment(s) must appear in the ticket text as a word, OR appear in the investigation notes as part of a path string (a token containing "/"), not just as an ordinary word in prose.
            # - Otherwise return status "rejected" with a reason naming the new directory.
            # - Keep it ecosystem-neutral: no special-casing of Next.js, "pages", "app" or any framework.
            if parent_dir and parent_dir != ".":
                segments = [seg for seg in parent_dir.split("/") if seg and seg != "."]
                for i in range(len(segments)):
                    current_prefix_path = "/".join(segments[: i + 1])
                    current_segment = segments[i]

                    # Check if this prefix path already exists on disk
                    prefix_already_exists = False
                    if base_dir is not None:
                        try:
                            target_dir = (base_dir / current_prefix_path).resolve()
                            prefix_already_exists = target_dir.exists() and target_dir.is_dir()
                        except Exception:
                            prefix_already_exists = False

                    if not prefix_already_exists and inv_dirs:
                        if current_prefix_path in inv_dirs or any(
                            d == current_prefix_path or d.startswith(current_prefix_path + "/")
                            for d in inv_dirs
                            if d not in ("", ".")
                        ):
                            prefix_already_exists = True

                    if not prefix_already_exists:
                        seg_in_ticket = bool(
                            re.search(r"\b" + re.escape(current_segment) + r"\b", ticket, re.IGNORECASE)
                        )
                        seg_in_notes_path = _segment_in_notes_path(current_segment, investigation_notes)

                        if not (seg_in_ticket or seg_in_notes_path):
                            return {
                                "status": "rejected",
                                "reason": (
                                    f"Scope proposes creating file under new directory segment '{current_segment}' "
                                    f"('{current_prefix_path}'), which does not exist on disk, does not appear in the ticket "
                                    f"text as a word, and does not appear in investigation notes as a path string."
                                ),
                            }

        # 2. Check if file path, filename, or meaningful stem is explicitly present in notes or ticket
        if in_notes or in_ticket:
            continue

        # 3. If file does not exist on disk, and is not in notes, and not in ticket:
        # Case A: Ticket or subgoal explicitly targets existing code/functions
        if subgoal_targets_existing:
            return {
                "status": "rejected",
                "reason": f"Scope file '{scope_item}' does not exist in repository and was not found during investigation for ticket modifying existing code.",
            }

        # Case B: Investigation notes found specific files, but the proposed file has an alien extension
        # that was never found during investigation (e.g. .py proposed for a .ts repo)
        if notes_extensions and ext and (ext not in notes_extensions):
            ext_exists_in_workspace = False
            if base_dir is not None:
                try:
                    ext_exists_in_workspace = any(base_dir.glob(f"*.{ext}")) or any(base_dir.glob(f"*/*.{ext}"))
                except Exception:
                    pass
            if not ext_exists_in_workspace:
                return {
                    "status": "rejected",
                    "reason": f"Scope file '{scope_item}' has file extension '.{ext}' which does not exist in the repository and was not found during investigation.",
                }

        # Case C: If investigation notes exist and ticket does NOT request creating new files,
        # touching uninvestigated non-existent files is ungrounded
        if notes_lower.strip() and not creates_new:
            return {
                "status": "rejected",
                "reason": f"Scope file '{scope_item}' does not exist in repository and was not identified in investigation notes.",
            }

    return {
        "status": "accepted",
        "reason": f"Scope paths ({', '.join(subgoal.scope)}) grounded in investigated files/directories.",
    }


def _validate_plan_grounding(
    subgoals: List[Subgoal],
    workspace: Optional[Any],
    ticket: str,
    investigation_notes: str,
    investigated_directories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Validate that subgoal scopes are grounded in the repository or investigation notes."""
    results = []
    for subgoal in subgoals:
        res = _check_single_subgoal_grounding(
            subgoal, workspace, ticket, investigation_notes, investigated_directories
        )
        if res["status"] == "rejected":
            raise ValueError(res["reason"])
        results.append(res)
    return results



def node_plan(
    state: State,
    workspace: Optional[Any] = None,
    llm: Optional[Any] = None,
    gatekeeper: Optional[Any] = None,
    recorder: Optional[Any] = None,
) -> State:
    """State 2: PLANNING (Generates structured Subgoals with declared scope).

    We use standard prompt instructions + manual JSON parsing with Pydantic validation
    (PlanSubgoalModel) rather than provider-bound .with_structured_output().
    Rationale:
    1. Complete provider agnosticism: Works identically across any LangChain BaseChatModel
       (Gemini, OpenAI, Claude, local models) and scripted/mock test models without requiring
       vendor-specific tool-calling or JSON-mode schema compilation.
    2. Strict Rule 0 enforcement: Direct visibility into JSONDecodeError and pydantic.ValidationError
       allows exact error messages to be reflected back in the retry prompt context.
    3. Deterministic failure boundaries: Guarantees zero silent coercions, hard stop on retry failure.
    """
    if "trajectory" not in state or state["trajectory"] is None:
        state["trajectory"] = []

    # llm=None backwards compatibility branch (matching 7a/7b pattern)
    if llm is None:
        plan_queue = state.get("plan_queue")
        if not plan_queue:
            ticket = state.get("ticket", "Default task")
            plan_queue = [
                Subgoal(
                    id="sg-1",
                    description=ticket,
                    scope=[],
                    expects_tests=True,
                    status="pending",
                )
            ]
            state["plan_queue"] = plan_queue
        else:
            normalized_queue: List[Subgoal] = []
            for idx, item in enumerate(plan_queue):
                if isinstance(item, dict):
                    sg = Subgoal(**item)
                elif isinstance(item, Subgoal):
                    sg = item
                else:
                    continue
                if not sg.id:
                    sg.id = f"sg-{idx + 1}"
                if not sg.status:
                    sg.status = "pending"
                normalized_queue.append(sg)
            state["plan_queue"] = normalized_queue

        if recorder and state.get("plan_queue"):
            safe_rec = recorder if isinstance(recorder, SafeRecorder) else SafeRecorder(recorder)
            run_id = state.get("run_id") or state.get("thread_id") or "unknown"
            safe_rec.record_subgoals(run_id, state["plan_queue"])

        state["current_subgoal"] = None
        state["last_feedback"] = None
        state["gate_status"] = None
        state["trajectory"].append({
            "node": "plan",
            "subgoals": [
                sg.model_dump() if hasattr(sg, "model_dump") else dict(sg)
                for sg in state["plan_queue"]
            ],
        })
        return state

    ticket = state.get("ticket", "")
    investigation_notes = state.get("investigation_notes", "")

    prompt_lines = [
        "You are the Planning module of the Jev State Engine.",
        "Your role is to create a deterministic, structured execution plan broken down into atomic Subgoals.",
        "",
        f"Ticket Description:\n{ticket}",
    ]
    if investigation_notes:
        prompt_lines.append(f"\nInvestigation Notes:\n{investigation_notes}")
    else:
        prompt_lines.append("\nInvestigation Notes: None")

    if state.get("investigation_incomplete", False):
        prompt_lines.append(
            "\nWARNING: Investigation hit its maximum turn cap before calling finish_investigation. "
            "Investigation notes may be incomplete. Plan cautiously and strictly ground all subgoals "
            "in the files and directories that were successfully discovered."
        )

    if state.get("repo_profile") == "on" and state.get("repo_profile_text"):
        prompt_lines.append(
            f"\nRepository Profile (hard facts detected from manifests, not guesses):\n{state['repo_profile_text']}"
        )

    prompt_lines.extend([
        "",
        "Instructions:",
        "1. Break down the task into an ordered sequence of atomic Subgoals.",
        "2. For each Subgoal, specify:",
        "   - 'description': Clear, atomic, and objective description of the functional change (e.g. 'Create the About page component at src/app/about/page.tsx for Smash Arena').",
        "     CRITICAL: Do NOT include subjective meta-qualifiers, styling instructions, or cross-file comparison clauses in 'description' (e.g. do NOT say 'following dark theme', 'using Tailwind v4 styling', 'consistent with existing pages like contact and home', or 'adhering to glassmorphism conventions'). Architectural and styling conventions belong in Investigation Notes, not in the subgoal description, because the gatekeeper evaluates the diff strictly against every clause in the description.",
        "   - 'scope': Non-empty list of exact file paths to touch or create. No placeholder or empty scopes allowed.",
        "   - 'expects_tests': Boolean (true/false) indicating whether automated tests are expected to pass/run for this step.",
        "3. GROUNDING REQUIREMENTS (CRITICAL):",
        "   - If modifying, extending, or documenting existing code, every file path in 'scope' MUST be grounded strictly in the files discovered in 'Investigation Notes' or explicitly named in the 'Ticket Description'.",
        "   - When Investigation Notes identify candidate files containing relevant or undocumented functions, you MUST select only from those candidate files. Do NOT target documentation (e.g. README.md), configuration files, or non-source files for tickets modifying or documenting existing functions.",
        "   - Do NOT invent, hallucinate, or guess file paths.",
        "   - Do NOT assume any default project language or file extensions (e.g. do NOT assume Python 'src/main.py' if the repository is TypeScript, Go, Rust, or JavaScript). Use the actual language and paths discovered in Investigation Notes.",
        "   - If the ticket explicitly requests creating brand new files not previously existing, those new paths must be consistent with the directory structure established in Investigation Notes.",
        "4. 'expects_tests' CRITERIA (CRITICAL):",
        "   - Set 'expects_tests': false for subgoals that only add or update docstrings, comments, JSDoc/TypeDoc, documentation, type annotations, or code formatting.",
        "   - Set 'expects_tests': false if the ticket does not request writing automated tests and the repository has no existing test suite for the files in scope.",
        "   - Set 'expects_tests': true ONLY when automated tests exist or are being written/modified to verify functional code logic.",
        "5. Output format:",
        "   Return ONLY a valid JSON array of Subgoal objects. Do NOT include markdown commentary or explanations outside the JSON array.",
        "   Schema illustration:",
        '   [{"description": "Atomic change description", "scope": ["relative/path/to/target/file"], "expects_tests": false}]',
    ])
    prompt_text = "\n".join(prompt_lines)
    messages: List[Any] = [HumanMessage(content=prompt_text)]

    subgoals: Optional[List[Subgoal]] = None
    last_error: Optional[str] = None
    retries_used = 0

    for attempt in range(2):
        # API call with retry backoff for rate limiting
        response = None
        for api_attempt in range(5):
            try:
                response = llm.invoke(messages)
                break
            except Exception as e:
                if api_attempt == 4:
                    state["plan_queue"] = []
                    state["current_subgoal"] = None
                    state["gate_status"] = "planning_failed"
                    state["status"] = "escalated"
                    state["last_feedback"] = f"Planning API failure after 5 retries: {e}"
                    state["trajectory"].append({
                        "node": "plan",
                        "error_type": "api_failure",
                        "error": f"API failure after 5 retries: {e}",
                        "subgoals": [],
                    })
                    _safe_escalate_deadlock(
                        gatekeeper,
                        trajectory=state["trajectory"],
                        triggering_tier="planning",
                        thread_id=state.get("thread_id"),
                    )
                    return state
                sleep_time = 25 if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else (5 * (api_attempt + 1))
                time.sleep(sleep_time)

        raw_content = _extract_content_text(getattr(response, "content", response))

        try:
            parsed_subgoals = _parse_and_validate_plan(raw_content)
            _validate_plan_grounding(
                parsed_subgoals,
                workspace,
                ticket,
                investigation_notes,
                state.get("investigated_directories"),
            )
            subgoals = parsed_subgoals
            break

        except Exception as exc:
            subgoals = None
            last_error = str(exc)
            if attempt == 0:
                retries_used = 1
                messages.append(response if isinstance(response, AIMessage) else AIMessage(content=raw_content))
                retry_prompt = (
                    f"your previous output failed validation because: {last_error}, please retry. "
                    "Return ONLY a valid JSON array of Subgoal objects matching the required schema: "
                    "description (non-empty string), scope (non-empty list of file paths), expects_tests (boolean)."
                )
                messages.append(HumanMessage(content=retry_prompt))

    if subgoals is not None:
        for idx, sg in enumerate(subgoals):
            if not sg.id:
                sg.id = f"sg-{idx + 1}"
            if not sg.status:
                sg.status = "pending"
        state["plan_queue"] = subgoals

        if recorder and state.get("plan_queue"):
            safe_rec = recorder if isinstance(recorder, SafeRecorder) else SafeRecorder(recorder)
            run_id = state.get("run_id") or state.get("thread_id") or "unknown"
            safe_rec.record_subgoals(run_id, state["plan_queue"])

        state["current_subgoal"] = None
        state["last_feedback"] = None
        state["gate_status"] = None

        grounding_checks = [
            {
                "subgoal": sg.description,
                "scope": list(sg.scope),
                **_check_single_subgoal_grounding(
                    sg, workspace, ticket, investigation_notes, state.get("investigated_directories")
                ),
            }
            for sg in subgoals
        ]
        subgoals_with_grounding = []
        for sg, gc in zip(subgoals, grounding_checks):
            sg_dict = sg.model_dump() if hasattr(sg, "model_dump") else dict(sg)
            sg_dict["directory_grounding"] = gc
            subgoals_with_grounding.append(sg_dict)

        traj_entry: Dict[str, Any] = {
            "node": "plan",
            "subgoals": subgoals_with_grounding,
            "grounding_checks": grounding_checks,
        }
        if retries_used > 0:
            traj_entry["retries"] = retries_used
        state["trajectory"].append(traj_entry)
        return state

    # Hard failure: planning failed after retry
    state["plan_queue"] = []
    state["current_subgoal"] = None
    state["gate_status"] = "planning_failed"
    state["status"] = "escalated"
    state["last_feedback"] = f"Planning failed after retry: {last_error}"

    fail_grounding_checks: List[Dict[str, Any]] = []
    if 'parsed_subgoals' in locals() and parsed_subgoals:
        for sg in parsed_subgoals:
            res = _check_single_subgoal_grounding(
                sg, workspace, ticket, investigation_notes, state.get("investigated_directories")
            )
            fail_grounding_checks.append({
                "subgoal": getattr(sg, "description", ""),
                "scope": list(getattr(sg, "scope", [])),
                **res,
            })
    if not fail_grounding_checks:
        fail_grounding_checks.append({"status": "rejected", "reason": f"Planning validation failed: {last_error}"})

    state["trajectory"].append({
        "node": "plan",
        "error_type": "validation_failure",
        "error": f"Planning validation failed: {last_error}",
        "subgoals": [],
        "grounding_checks": fail_grounding_checks,
    })
    _safe_escalate_deadlock(
        gatekeeper,
        trajectory=state["trajectory"],
        triggering_tier="planning",
        thread_id=state.get("thread_id"),
    )
    return state



def node_implement(
    state: State,
    workspace: Optional[Any] = None,
    llm: Optional[Any] = None,
    gatekeeper: Optional[Any] = None,
) -> State:
    """State 3: IMPLEMENTATION (Executes subgoal mutations in isolated worktree)."""
    if "trajectory" not in state or state["trajectory"] is None:
        state["trajectory"] = []

    plan_queue = state.get("plan_queue", [])
    current_subgoal = state.get("current_subgoal")

    # If current subgoal is None or previous subgoal passed, pop next from queue
    if current_subgoal is None or state.get("gate_status") == "passed":
        if plan_queue:
            current_subgoal = plan_queue.pop(0)
            state["current_subgoal"] = current_subgoal
            state["gate_status"] = None
            state["failed_attempts"] = []

    subgoal: Optional[Subgoal] = None
    if isinstance(current_subgoal, dict):
        subgoal = Subgoal(**current_subgoal)
    elif isinstance(current_subgoal, Subgoal):
        subgoal = current_subgoal

    if subgoal is not None:
        subgoal.status = "in_progress"
        state["current_subgoal"] = subgoal

    state["worker_tokens_in"] = None
    state["worker_tokens_out"] = None

    if workspace is not None and hasattr(workspace, "create_subgoal_worktree") and subgoal is not None:
        if not state.get("current_worktree_path"):
            subgoal_desc = getattr(subgoal, "description", "") or "subgoal"
            clean_desc = re.sub(r"[^a-zA-Z0-9_-]", "_", subgoal_desc)[:20].strip("_") or "subgoal"
            subgoal_id = f"{clean_desc}_{uuid.uuid4().hex[:6]}"
            base_ref = state.get("integration_branch") or "HEAD"
            if hasattr(workspace, "get_branch_commit") and state.get("integration_branch"):
                try:
                    state["subgoal_base_commit"] = workspace.get_branch_commit(state["integration_branch"])
                except Exception:
                    state["subgoal_base_commit"] = state.get("base_commit")
            elif hasattr(workspace, "base_commit"):
                state["subgoal_base_commit"] = workspace.base_commit
            wt_path = workspace.create_subgoal_worktree(subgoal_id, base_ref=base_ref)
            state["current_worktree_path"] = str(wt_path)
            state["current_worktree_branch"] = f"jev-subgoal-{subgoal_id}"
        elif hasattr(workspace, "worktree_dir") and state.get("current_worktree_path"):
            workspace.worktree_dir = Path(state["current_worktree_path"])

    executed_tools: List[Dict[str, Any]] = []

    if llm is not None and subgoal is not None and workspace is not None:
        submitted = False

        def _execute_stage_file_mutation(path: str, content: str) -> str:
            if not path:
                return "Error: path is required for stage_file_mutation."

            try:
                workspace.resolve_in_worktree(path)
            except (ValueError, PermissionError) as e:
                return f"Error: Access denied: path '{path}' escapes worktree: {e}"

            norm_path = Path(path).as_posix()
            if norm_path.startswith("./"):
                norm_path = norm_path[2:]

            norm_scope = set()
            for s in (subgoal.scope or []):
                s_norm = Path(s).as_posix()
                if s_norm.startswith("./"):
                    s_norm = s_norm[2:]
                norm_scope.add(s_norm)

            if norm_scope and norm_path not in norm_scope:
                return (
                    f"Error: Target path '{path}' is not within declared scope {subgoal.scope}. "
                    "You may only mutate files within the declared scope for this subgoal. "
                    "Subsequent subgoals will address other files."
                )

            if hasattr(workspace, "stage_file_mutation"):
                try:
                    workspace.stage_file_mutation(path, content)
                    return f"Successfully staged mutation for {path}"
                except (ValueError, PermissionError) as e:
                    return f"Error: Access denied staging mutation for '{path}': {e}"
                except Exception as e:
                    return f"Error staging mutation for {path}: {e}"
            return f"Workspace unavailable; could not stage mutation for {path}"

        @tool
        def stage_file_mutation(path: str, content: str) -> str:
            """Stage a file mutation (create or overwrite file content) in the workspace worktree.

            Args:
                path: Path to the file to modify, relative to repository root.
                content: The complete new content of the file.
            """
            return _execute_stage_file_mutation(path, content)

        @tool
        def submit_subgoal(notes: str = "") -> str:
            """Signal that the implementation for the current subgoal is complete and ready for gating.

            Args:
                notes: Optional summary or notes about the changes made.
            """
            nonlocal submitted
            submitted = True
            return "Subgoal submitted."

        tools = [stage_file_mutation, submit_subgoal]

        prompt_lines = []
        if state.get("ticket"):
            prompt_lines.append(f"Ticket: {state['ticket']}")
        prompt_lines.extend([
            f"Current Subgoal: {subgoal.description}",
            f"Declared Scope: {json.dumps(subgoal.scope)}",
        ])
        if subgoal.expects_tests:
            prompt_lines.append("Note: This subgoal expects tests to verify its implementation.")
        else:
            prompt_lines.append("Note: This subgoal does not expect tests.")

        if state.get("investigation_notes"):
            prompt_lines.append(
                f"Investigation Notes (Project Architecture & Conventions):\n{state['investigation_notes']}"
            )

        if state.get("repo_profile") == "on" and state.get("repo_profile_text"):
            prompt_lines.append(
                f"Repository Profile (hard facts detected from manifests, not guesses):\n{state['repo_profile_text']}"
            )

        if state.get("last_feedback"):
            prompt_lines.append(f"Previous attempt failed gate verification. Feedback:\n{state['last_feedback']}")

        # Dead-end block (Phase 3 Item B)
        retry_intel = state.get("retry_intel", "on")
        if retry_intel != "off" and state.get("failed_attempts"):
            block_lines = ["Earlier attempts at this subgoal failed. Do not repeat them."]
            for fa in state["failed_attempts"]:
                att_lines = [
                    f"Attempt {fa.get('attempt_no', '?')} (Outcome: {fa.get('outcome')}, Failed check: {fa.get('failed_check')}):"
                ]
                errs = fa.get("normalized_error_keys") or fa.get("error_keys")
                if errs:
                    att_lines.append(f"Errors: {json.dumps(errs)}")
                if fa.get("diff_excerpt"):
                    att_lines.append(f"Diff excerpt:\n{fa['diff_excerpt']}")
                block_lines.append("\n".join(att_lines))
            prompt_lines.append("\n\n".join(block_lines))


        existing_files_context = []
        worktree = getattr(workspace, "worktree_dir", getattr(workspace, "repo_dir", None))

        # Grounding: Extract Product Identity & Business Domain if root metadata is present
        if worktree:
            wt_path = Path(worktree)
            layout_candidates = [
                wt_path / "src" / "app" / "layout.tsx",
                wt_path / "src" / "app" / "layout.jsx",
                wt_path / "src" / "app" / "layout.js",
                wt_path / "app" / "layout.tsx",
                wt_path / "app" / "layout.jsx",
                wt_path / "app" / "layout.js",
                wt_path / "layout.tsx",
            ]
            extracted_title = None
            extracted_desc = None
            for cand in layout_candidates:
                if cand.exists() and cand.is_file():
                    try:
                        content = cand.read_text(encoding="utf-8")
                        title_match = re.search(r'title:\s*["\']([^"\']+)["\']', content)
                        desc_match = re.search(r'description:\s*["\']([^"\']+)["\']', content)
                        if title_match:
                            extracted_title = title_match.group(1).strip()
                        if desc_match:
                            extracted_desc = desc_match.group(1).strip()
                        if extracted_title or extracted_desc:
                            break
                    except Exception:
                        pass

            pkg_json = wt_path / "package.json"
            if (not extracted_title or not extracted_desc) and pkg_json.exists() and pkg_json.is_file():
                try:
                    pkg_data = json.loads(pkg_json.read_text(encoding="utf-8"))
                    if not extracted_title and "name" in pkg_data:
                        extracted_title = str(pkg_data["name"]).strip()
                    if not extracted_desc and "description" in pkg_data:
                        extracted_desc = str(pkg_data["description"]).strip()
                except Exception:
                    pass

            if extracted_title or extracted_desc:
                identity_lines = ["Product Identity & Business Domain:"]
                if extracted_title:
                    identity_lines.append(f"- Title: {extracted_title}")
                if extracted_desc:
                    identity_lines.append(f"- Description: {extracted_desc}")
                prompt_lines.append("\n".join(identity_lines))

        if worktree and subgoal.scope:
            for s_file in subgoal.scope:
                target_file = Path(worktree) / s_file
                if target_file.exists() and target_file.is_file():
                    try:
                        file_content = target_file.read_text(encoding="utf-8", errors="replace")
                        if len(file_content) > MAX_IMPLEMENT_FILE_CHARS:
                            msg = (
                                f"File '{s_file}' ({len(file_content)} characters) is too large for a safe full-file rewrite "
                                f"(maximum allowed is {MAX_IMPLEMENT_FILE_CHARS} characters)."
                            )
                            state["status"] = "escalated"
                            state["gate_status"] = "escalated"
                            state["last_feedback"] = msg
                            state["trajectory"].append({
                                "node": "implement",
                                "subgoal": getattr(subgoal, "description", ""),
                                "error": msg,
                            })
                            _safe_escalate_deadlock(
                                gatekeeper,
                                trajectory=state.get("trajectory", []),
                                triggering_tier="implementation",
                                integration_branch=state.get("integration_branch"),
                                thread_id=state.get("thread_id"),
                            )
                            return state
                        existing_files_context.append(f"--- Existing file: {s_file} ---\n{file_content}\n--- End of {s_file} ---")
                    except Exception:
                        pass
        if existing_files_context:
            prompt_lines.append("Existing content of files in scope:\n" + "\n\n".join(existing_files_context))

        prompt_lines.append(
            "Instructions:\n"
            "1. Strictly adhere to the project architecture, dependencies, styling conventions, and existing component patterns documented in the Investigation Notes. Do not introduce uninstalled libraries or conflicting layout structures (e.g. do not create redundant headers or navbars if layout.tsx or a global component already provides them).\n"
            "2. Strictly adhere to the application's actual business domain, product identity, and branding documented in the Investigation Notes and repository. Never invent an alternate sport, brand, company name, or unrelated business model (e.g. do not substitute Padel or Tennis for Badminton).\n"
            "3. When creating or modifying UI components, align with the project's export and import conventions (e.g. if root layouts and surrounding components use default exports such as `export default function Component`, provide default exports so default imports resolve cleanly).\n"
            "4. Stage all necessary code changes using the `stage_file_mutation` tool.\n"
            "   Only mutate files within the declared scope.\n"
            "5. When all changes are staged and you are done, call the `submit_subgoal` tool to submit your work for gate verification."
        )
        prompt_text = "\n\n".join(prompt_lines)

        messages: List[Any] = [HumanMessage(content=prompt_text)]
        bound_llm = llm.bind_tools(tools)

        max_turns = 10
        turn = 0
        while turn < max_turns and not submitted:
            turn += 1
            response = None
            for api_attempt in range(5):
                try:
                    response = bound_llm.invoke(messages)
                    break
                except Exception as e:
                    if api_attempt == 4:
                        raise
                    sleep_time = 25 if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else (5 * (api_attempt + 1))
                    time.sleep(sleep_time)

            messages.append(response)

            usage_meta = getattr(response, "usage_metadata", None)
            if isinstance(usage_meta, dict):
                in_t = usage_meta.get("input_tokens")
                out_t = usage_meta.get("output_tokens")
                if in_t is not None:
                    state["worker_tokens_in"] = (state.get("worker_tokens_in") or 0) + in_t
                if out_t is not None:
                    state["worker_tokens_out"] = (state.get("worker_tokens_out") or 0) + out_t

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls and isinstance(response, dict):
                tool_calls = response.get("tool_calls")

            if not tool_calls:
                break

            for tool_call in tool_calls:
                name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", f"call_{turn}")

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                if name == "submit_subgoal":
                    submitted = True
                    result = "Subgoal submitted."
                    stored_args = args
                elif name in ("stage_file_mutation", "write_file"):
                    actual_path = args.get("path") or args.get("file_path") or args.get("filename") or ""
                    actual_content = args.get("content")
                    if actual_content is None:
                        actual_content = args.get("code", args.get("text", ""))
                    result = _execute_stage_file_mutation(actual_path, str(actual_content))
                    c_str = str(actual_content)
                    stored_args = {
                        "path": actual_path,
                        "content_sha": hashlib.sha256(c_str.encode("utf-8")).hexdigest(),
                        "content_chars": len(c_str),
                    }
                else:
                    result = f"Unknown tool: {name}. Available tools: stage_file_mutation, submit_subgoal."
                    stored_args = args

                executed_tools.append({"name": name, "args": stored_args, "result": str(result)[:200]})

                messages.append(
                    ToolMessage(
                        content=str(result),
                        name=name,
                        tool_call_id=str(call_id or f"call_{turn}"),
                    )
                )

            if submitted:
                break

    subgoal_data = (
        subgoal.model_dump()
        if subgoal and hasattr(subgoal, "model_dump")
        else (dict(subgoal) if subgoal else (dict(current_subgoal) if current_subgoal else None))
    )
    traj_entry: Dict[str, Any] = {"node": "implement", "current_subgoal": subgoal_data}
    if executed_tools:
        traj_entry["tool_calls"] = executed_tools
    state["trajectory"].append(traj_entry)
    return state


def _is_active_worktree(state: State, workspace: Optional[Any]) -> bool:
    wt_path = state.get("current_worktree_path")
    if isinstance(wt_path, (str, Path)) and str(wt_path).strip():
        return True
    ws_path = getattr(workspace, "current_worktree_path", None) if workspace else None
    if isinstance(ws_path, (str, Path)) and str(ws_path).strip():
        return True
    if (
        workspace is not None
        and hasattr(workspace, "worktree_dir")
        and hasattr(workspace, "repo_dir")
        and isinstance(workspace.worktree_dir, Path)
        and isinstance(workspace.repo_dir, Path)
        and workspace.worktree_dir != workspace.repo_dir
    ):
        return True
    return False


def _get_active_worktree_info(state: State, workspace: Optional[Any]) -> tuple[Optional[str], Optional[str]]:
    wt_path = state.get("current_worktree_path")
    if not isinstance(wt_path, (str, Path)):
        wt_path = getattr(workspace, "current_worktree_path", None) if workspace else None
    if not isinstance(wt_path, (str, Path)):
        resolved_path = None
    else:
        resolved_path = str(wt_path)

    wt_branch = state.get("current_worktree_branch")
    if not isinstance(wt_branch, str):
        wt_branch = getattr(workspace, "current_worktree_branch", None) if workspace else None
    if not isinstance(wt_branch, str):
        resolved_branch = None
    else:
        resolved_branch = str(wt_branch)

    return resolved_path, resolved_branch


def _format_compile_feedback(
    compile_info: Dict[str, Any], fallback_detail: str = ""
) -> str:
    cmd = compile_info.get("command")
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
    new_errors = compile_info.get("new_errors") or []

    if not new_errors:
        if cmd_str:
            return f"Compile command failed: {cmd_str}\n{fallback_detail or compile_info.get('detail', '')}".strip()
        return fallback_detail or compile_info.get("detail", "Compile check failed.")

    header_lines = []
    if cmd_str:
        header_lines.append(f"Command: {cmd_str}")
    header_lines.append("Errors:")
    header = "\n".join(header_lines)

    max_lines = 20
    max_total_chars = 3000

    selected_errors: List[str] = []

    for err in new_errors:
        err_str = str(err).strip()
        if not err_str:
            continue
        if len(selected_errors) >= max_lines:
            break

        remaining_if_stopped = len(new_errors) - (len(selected_errors) + 1)
        omission_line = f"\n...and {remaining_if_stopped} more errors omitted" if remaining_if_stopped > 0 else ""

        candidate_body = "\n".join(selected_errors + [err_str])
        candidate_total = f"{header}\n{candidate_body}{omission_line}"

        if len(candidate_total) > max_total_chars and selected_errors:
            break

        selected_errors.append(err_str)

    omitted = len(new_errors) - len(selected_errors)
    out_lines = [header] + selected_errors
    if omitted > 0:
        out_lines.append(f"...and {omitted} more errors omitted")

    return "\n".join(out_lines)


def _teardown_worktree(state: State, workspace: Optional[Any]) -> None:
    """Helper to tear down active worktree and restore workspace repo_dir."""
    if workspace is not None:
        is_wt = _is_active_worktree(state, workspace)
        wt_path, wt_branch = _get_active_worktree_info(state, workspace)
        if is_wt and hasattr(workspace, "discard_subgoal_worktree"):
            workspace.discard_subgoal_worktree(wt_path, wt_branch)
        elif hasattr(workspace, "rollback_subgoal"):
            workspace.rollback_subgoal()
        if hasattr(workspace, "repo_dir") and isinstance(workspace.repo_dir, Path):
            workspace.worktree_dir = workspace.repo_dir
    state["current_worktree_path"] = None
    state["current_worktree_branch"] = None


def _snapshot_attempt(
    recorder: Optional[Any],
    state: State,
    subgoal_id: str,
    tier: str,
    outcome: str,
    failed_check: Optional[str] = None,
    diff: str = "",
    compile_info: Optional[Dict[str, Any]] = None,
    tests_ran: bool = False,
    test_outcome: Optional[str] = None,
    verdict: Optional[Any] = None,
    feedback_sent: Optional[str] = None,
    evidence_level: Optional[str] = None,
    dependency_delta: Optional[str] = None,
    jev_state_keys: Optional[str] = None,
    qwen_postmortem: Optional[str] = None,
    qwen_latency_ms: Optional[float] = None,
    qwen_error: Optional[str] = None,
) -> None:
    """Helper to snapshot an attempt row to the ledger."""
    if not recorder:
        return
    if not isinstance(jev_state_keys, str):
        jev_state_keys = None
    safe_rec = recorder if isinstance(recorder, SafeRecorder) else SafeRecorder(recorder)
    run_id = state.get("run_id") or state.get("thread_id") or "unknown"
    if isinstance(diff, str):
        diff_bytes = diff.encode("utf-8")
    elif isinstance(diff, bytes):
        diff_bytes = diff
        diff = diff.decode("utf-8", errors="replace")
    else:
        diff_bytes = b""
        diff = ""
    diff_sha = hashlib.sha256(diff_bytes).hexdigest()
    files: List[str] = []
    if diff:
        try:
            files = sorted(list(Workspace._extract_files_from_diff(diff)))
        except Exception:
            files = []
    norm_errors = _normalize_compile_errors(compile_info)
    jev_valid = getattr(verdict, "valid", None) if verdict else None
    jev_prob = getattr(verdict, "probability", None) if verdict else None
    jev_model = getattr(verdict, "model", None) if verdict else None
    jev_raw = None
    if verdict and getattr(verdict, "raw_response", None):
        try:
            jev_raw = json.dumps(verdict.raw_response)
        except Exception:
            jev_raw = str(verdict.raw_response)
    jev_usage = getattr(verdict, "usage", None) if verdict else None
    jev_tokens_in = None
    jev_tokens_out = None
    if isinstance(jev_usage, dict):
        jev_tokens_in = jev_usage.get("tokens_in") or jev_usage.get("prompt_tokens") or jev_usage.get("input_tokens")
        jev_tokens_out = jev_usage.get("tokens_out") or jev_usage.get("completion_tokens") or jev_usage.get("output_tokens")

    safe_rec.record_attempt(
        run_id=run_id,
        subgoal_id=subgoal_id,
        tier=tier,
        outcome=outcome,
        failed_check=failed_check,
        diff=diff,
        diff_sha=diff_sha,
        files=files,
        compile_errors=norm_errors if norm_errors else None,
        tests_ran=tests_ran,
        test_outcome=test_outcome,
        jev_valid=jev_valid,
        jev_probability=jev_prob,
        jev_model=jev_model,
        jev_raw=jev_raw,
        jev_tokens_in=jev_tokens_in,
        jev_tokens_out=jev_tokens_out,
        feedback_sent=feedback_sent,
        tokens_in=state.get("worker_tokens_in") if subgoal_id != "__verify__" else None,
        tokens_out=state.get("worker_tokens_out") if subgoal_id != "__verify__" else None,
        evidence_level=evidence_level,
        dependency_delta=dependency_delta,
        jev_state_keys=jev_state_keys,
        qwen_postmortem=qwen_postmortem,
        qwen_latency_ms=qwen_latency_ms,
        qwen_error=qwen_error,
    )


def _invoke_qwen_strike_postmortem(
    state: State,
    qwen_client: Optional[Any],
    subgoal: Any,
    rejecting_source: str,
    rejection_reason: str,
    diff: str,
    current_tier_strikes: int,
    jev_verdict: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Invokes Qwen advisory post-mortem for a rejected attempt without altering gate outcomes."""
    effective_client = qwen_client if qwen_client is not None else state.get("qwen_client")
    if not effective_client or not getattr(effective_client, "enabled", False):
        return None, None, None

    try:
        total_strikes = (state.get("total_strikes") or 0) + 1
        state["total_strikes"] = total_strikes
        strike_history = state.get("strike_history")
        if strike_history is None:
            strike_history = []
            state["strike_history"] = strike_history

        strike_number = f"{current_tier_strikes} of 3"
        ticket = state.get("ticket", "")
        thread_id = state.get("thread_id") or state.get("run_id") or "unknown"
        subgoal_id = getattr(subgoal, "id", None) or "sg-1"

        res = effective_client.run_postmortem(
            ticket=ticket,
            rejecting_source=rejecting_source,
            rejection_reason=rejection_reason,
            jev_verdict=jev_verdict,
            diff=diff,
            strike_number=strike_number,
            previous_strikes=list(strike_history),
            thread_id=thread_id,
            subgoal_id=subgoal_id,
            strike_idx=total_strikes,
        )
        if res:
            strike_history.append({
                "strike_no": total_strikes,
                "source": rejecting_source,
                "reason": rejection_reason,
                "root_cause": res.root_cause,
            })
            if getattr(effective_client, "feedback", False):
                advisory = effective_client.format_advisory_feedback(res)
                if advisory:
                    state["last_feedback"] = (state.get("last_feedback") or "") + advisory
            return res.to_json(), res.latency_ms, res.error
    except Exception as exc:
        logger.warning(f"[QWEN] Exception running strike post-mortem: {exc}")
    return None, None, None


def node_gate(
    state: State,
    workspace: Optional[Any] = None,
    gatekeeper: Optional[Any] = None,
    recorder: Optional[Any] = None,
    qwen_client: Optional[Any] = None,
) -> State:
    """The Two-Tier Gate Router.

    Tier 0 (Mechanical):
      - Run workspace.run_mechanical_checks(subgoal).
      - If fail: rollback_subgoal, increment mechanical_strike_count.
        Do NOT call gatekeeper. Do NOT touch semantic_strike_count.
        If mechanical_strike_count reaches 3, escalate_deadlock.
      - If pass: proceed to Tier 1.

    Tier 1 (Semantic):
      - Call gatekeeper.validate_subgoal(subgoal, diff, mechanical_detail=...).
      - If valid: commit_subgoal, reset BOTH strike counters to 0.
      - If invalid: rollback_subgoal, increment semantic_strike_count.
        Do NOT touch mechanical_strike_count.
        If semantic_strike_count reaches 3, escalate_deadlock.
    """
    if state.get("status") == "escalated":
        return state

    if "trajectory" not in state or state["trajectory"] is None:
        state["trajectory"] = []

    if workspace is None:
        raise ValueError("Workspace must be provided to node_gate.")

    subgoal_raw = state.get("current_subgoal")
    if isinstance(subgoal_raw, dict):
        subgoal = Subgoal(**subgoal_raw)
    elif isinstance(subgoal_raw, Subgoal):
        subgoal = subgoal_raw
    else:
        subgoal = Subgoal()

    safe_rec = SafeRecorder(recorder) if recorder and not isinstance(recorder, SafeRecorder) else recorder
    run_id = state.get("run_id") or state.get("thread_id") or "unknown"
    subgoal_id = getattr(subgoal, "id", None) or "sg-1"

    raw_diff = workspace.get_staged_diff() if hasattr(workspace, "get_staged_diff") else ""
    if isinstance(raw_diff, str):
        diff = raw_diff
    elif isinstance(raw_diff, bytes):
        diff = raw_diff.decode("utf-8", errors="replace")
    else:
        diff = ""
    diff_sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()

    # Duplicate-diff guard (Phase 3 Item C)
    retry_intel = state.get("retry_intel")
    if retry_intel is None:
        retry_intel = "on" if state.get("failed_attempts") is not None else "off"
    if retry_intel != "off":
        matching_fa = None
        for fa in state.get("failed_attempts", []):
            if fa.get("diff_sha") == diff_sha:
                matching_fa = fa
                break
        if matching_fa is not None:
            orig_tier = matching_fa.get("tier", "mechanical")
            if orig_tier == "semantic":
                current_strikes = state.get("semantic_strike_count", 0) + 1
                state["semantic_strike_count"] = current_strikes
                state["gate_status"] = "semantic_failure"
            else:
                current_strikes = state.get("mechanical_strike_count", 0) + 1
                state["mechanical_strike_count"] = current_strikes
                state["gate_status"] = "mechanical_failure"

            prev_att_no = matching_fa.get("attempt_no", 1)
            prev_reason = matching_fa.get("failed_check") or matching_fa.get("outcome") or "failure"
            feedback_msg = (
                f"Duplicate diff detected: identical to attempt {prev_att_no} which failed ({prev_reason}). "
                "Do not submit identical diffs."
            )
            state["last_feedback"] = feedback_msg

            qwen_pm, qwen_lat, qwen_err = _invoke_qwen_strike_postmortem(
                state=state,
                qwen_client=qwen_client,
                subgoal=subgoal,
                rejecting_source=f"Duplicate diff ({orig_tier})",
                rejection_reason=feedback_msg,
                diff=diff,
                current_tier_strikes=current_strikes,
                jev_verdict=None,
            )

            _record_failed_attempt_in_state(
                state,
                tier=orig_tier,
                outcome="duplicate_diff",
                failed_check=matching_fa.get("failed_check"),
                diff=diff,
            )
            _snapshot_attempt(
                safe_rec,
                state,
                subgoal_id,
                tier=orig_tier,
                outcome="duplicate_diff",
                failed_check=matching_fa.get("failed_check"),
                diff=diff,
                feedback_sent=state.get("last_feedback"),
                qwen_postmortem=qwen_pm,
                qwen_latency_ms=qwen_lat,
                qwen_error=qwen_err,
            )
            _teardown_worktree(state, workspace)

            state["trajectory"].append({
                "node": "gate",
                "subgoal": subgoal.model_dump() if hasattr(subgoal, "model_dump") else dict(subgoal),
                "gate_status": "duplicate_diff",
                "tier0_result": None,
                "compile": None,
                "test_runner": None,
                "jev_request": None,
                "jev_verdict": None,
                "qwen_postmortem": qwen_pm,
                "qwen_latency_ms": qwen_lat,
                "qwen_error": qwen_err,
            })

            if current_strikes >= 3:
                subgoal.status = "failed"
                state["current_subgoal"] = subgoal
                if safe_rec:
                    safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")
                _safe_escalate_deadlock(
                    gatekeeper,
                    trajectory=state.get("trajectory", []),
                    triggering_tier=orig_tier,
                    integration_branch=state.get("integration_branch"),
                    thread_id=state.get("thread_id"),
                )
            return state

    # Tier 0: Mechanical checks
    subgoal_base_commit = state.get("subgoal_base_commit")
    test_policy = state.get("test_policy", "auto")
    kwargs: Dict[str, Any] = {}
    if _accepts_param(workspace.run_mechanical_checks, "base_commit"):
        kwargs["base_commit"] = subgoal_base_commit
    if _accepts_param(workspace.run_mechanical_checks, "test_policy"):
        kwargs["test_policy"] = test_policy
    mech_result = workspace.run_mechanical_checks(subgoal, **kwargs)

    tier0_telemetry = {
        "passed": mech_result.passed,
        "failed_check": mech_result.failed_check,
        "detail": mech_result.detail,
        "checks_run": getattr(mech_result, "checks_run", []),
        "checks": getattr(mech_result, "checks", {}),
    }
    if not tier0_telemetry["checks"]:
        failed = mech_result.failed_check
        tier0_telemetry["checks"] = {
            "build": {"ran": True, "passed": failed != "build", "detail": mech_result.detail if failed == "build" else ""},
            "compile": {"ran": failed not in ("build",), "passed": failed not in ("build", "compile", "no_compile_command", "env_not_ready") if failed else True, "detail": mech_result.detail if failed in ("compile", "no_compile_command", "env_not_ready") else ""},
            "tests": {"ran": failed not in ("build", "compile", "no_compile_command", "env_not_ready"), "passed": failed not in ("build", "compile", "no_compile_command", "env_not_ready", "tests", "no_tests_collected") if failed else True, "detail": mech_result.detail if failed in ("tests", "no_tests_collected") else ""},
            "scope": {"ran": failed not in ("build", "compile", "no_compile_command", "env_not_ready", "tests", "no_tests_collected"), "passed": failed != "scope", "detail": mech_result.detail if failed == "scope" else ""},
        }
        tier0_telemetry["checks_run"] = [k for k, v in tier0_telemetry["checks"].items() if v.get("ran")]

    tests_ran = False
    if "tests" in tier0_telemetry.get("checks_run", []):
        tests_ran = True
    elif tier0_telemetry.get("checks", {}).get("tests", {}).get("ran"):
        tests_ran = True

    if mech_result.failed_check in ("build", "compile", "no_compile_command", "env_not_ready") or (not tests_ran and not mech_result.passed):
        test_runner_telemetry = {"ran": False}
    else:
        test_runner_telemetry = getattr(mech_result, "test_runner_outcome", None)
        if test_runner_telemetry is None and hasattr(workspace, "last_test_run"):
            test_runner_telemetry = workspace.last_test_run
        if test_runner_telemetry is None:
            test_runner_telemetry = {
                "ran": tests_ran,
                "ecosystem": None,
                "command": None,
                "exit_code": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "output_tail": "",
                "outcome": "PASSED" if mech_result.passed or mech_result.failed_check != "tests" else ("NO_TESTS_COLLECTED" if not tests_ran else "FAILED"),
            }
        elif isinstance(test_runner_telemetry, dict):
            test_runner_telemetry = dict(test_runner_telemetry)
            if "ran" not in test_runner_telemetry:
                test_runner_telemetry["ran"] = tests_ran

    compile_telemetry = getattr(mech_result, "compile_outcome", None)
    if compile_telemetry is None and hasattr(workspace, "last_compile_run"):
        compile_telemetry = workspace.last_compile_run
    if compile_telemetry is None and tier0_telemetry["checks"].get("compile", {}).get("ran"):
        compile_telemetry = {
            "ecosystem": None,
            "command": None,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_tail": "",
            "new_errors": [],
            "base_errors": [],
            "outcome": "PASSED" if mech_result.passed or mech_result.failed_check not in ("compile", "no_compile_command", "env_not_ready") else "FAILED",
        }

    if not mech_result.passed:
        if mech_result.failed_check in ("no_compile_command", "env_not_ready"):
            state["status"] = "escalated"
            state["gate_status"] = mech_result.failed_check
            state["last_feedback"] = mech_result.detail

            qwen_pm, qwen_lat, qwen_err = _invoke_qwen_strike_postmortem(
                state=state,
                qwen_client=qwen_client,
                subgoal=subgoal,
                rejecting_source=f"Tier 0 mechanical check: {mech_result.failed_check}",
                rejection_reason=mech_result.detail,
                diff=diff,
                current_tier_strikes=1,
                jev_verdict=None,
            )

            if retry_intel != "off":
                _record_failed_attempt_in_state(
                    state,
                    tier="mechanical",
                    outcome=mech_result.failed_check,
                    failed_check=mech_result.failed_check,
                    diff=diff,
                    compile_info=compile_telemetry,
                )
            _snapshot_attempt(
                safe_rec,
                state,
                subgoal_id,
                tier="mechanical",
                outcome=mech_result.failed_check,
                failed_check=mech_result.failed_check,
                diff=diff,
                compile_info=compile_telemetry,
                tests_ran=tests_ran,
                test_outcome=test_runner_telemetry.get("outcome") if isinstance(test_runner_telemetry, dict) else None,
                feedback_sent=state.get("last_feedback"),
                qwen_postmortem=qwen_pm,
                qwen_latency_ms=qwen_lat,
                qwen_error=qwen_err,
            )
            _teardown_worktree(state, workspace)
            subgoal.status = "failed"
            state["current_subgoal"] = subgoal
            if safe_rec:
                safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")

            state["trajectory"].append({
                "node": "gate",
                "subgoal": subgoal.model_dump() if hasattr(subgoal, "model_dump") else dict(subgoal),
                "gate_status": mech_result.failed_check,
                "tier0_result": tier0_telemetry,
                "compile": compile_telemetry,
                "test_runner": test_runner_telemetry,
                "jev_request": None,
                "jev_verdict": None,
                "qwen_postmortem": qwen_pm,
                "qwen_latency_ms": qwen_lat,
                "qwen_error": qwen_err,
            })

            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="mechanical",
                integration_branch=state.get("integration_branch"),
                thread_id=state.get("thread_id"),
            )
            return state

        current_mech_strikes = state.get("mechanical_strike_count", 0) + 1
        state["mechanical_strike_count"] = current_mech_strikes
        state["gate_status"] = "mechanical_failure"
        if mech_result.failed_check == "compile":
            state["last_feedback"] = _format_compile_feedback(
                compile_telemetry or getattr(mech_result, "compile_outcome", None) or {},
                fallback_detail=mech_result.detail,
            )
        elif mech_result.failed_check == "tests":
            base_fb = mech_result.detail or "Unit tests failed."
            output_tail = None
            if isinstance(test_runner_telemetry, dict):
                output_tail = test_runner_telemetry.get("output_tail")
            elif hasattr(mech_result, "test_runner_outcome") and isinstance(mech_result.test_runner_outcome, dict):
                output_tail = mech_result.test_runner_outcome.get("output_tail")
            if output_tail:
                tail_capped = output_tail[-4000:]
                if tail_capped not in base_fb:
                    state["last_feedback"] = f"{base_fb}\n\n{tail_capped}"
                else:
                    state["last_feedback"] = base_fb
            else:
                state["last_feedback"] = base_fb
        else:
            state["last_feedback"] = mech_result.detail

        mech_reason = state.get("last_feedback") or mech_result.detail or "Mechanical check failed."
        qwen_pm, qwen_lat, qwen_err = _invoke_qwen_strike_postmortem(
            state=state,
            qwen_client=qwen_client,
            subgoal=subgoal,
            rejecting_source=f"Tier 0 mechanical check: {mech_result.failed_check}",
            rejection_reason=mech_reason,
            diff=diff,
            current_tier_strikes=current_mech_strikes,
            jev_verdict=None,
        )

        if retry_intel != "off":
            _record_failed_attempt_in_state(
                state,
                tier="mechanical",
                outcome="mechanical_failure",
                failed_check=mech_result.failed_check,
                diff=diff,
                compile_info=compile_telemetry,
            )
        _snapshot_attempt(
            safe_rec,
            state,
            subgoal_id,
            tier="mechanical",
            outcome="mechanical_failure",
            failed_check=mech_result.failed_check,
            diff=diff,
            compile_info=compile_telemetry,
            tests_ran=tests_ran,
            test_outcome=test_runner_telemetry.get("outcome") if isinstance(test_runner_telemetry, dict) else None,
            feedback_sent=state.get("last_feedback"),
            qwen_postmortem=qwen_pm,
            qwen_latency_ms=qwen_lat,
            qwen_error=qwen_err,
        )
        _teardown_worktree(state, workspace)

        state["trajectory"].append({
            "node": "gate",
            "subgoal": subgoal.model_dump() if hasattr(subgoal, "model_dump") else dict(subgoal),
            "gate_status": "mechanical_failure",
            "tier0_result": tier0_telemetry,
            "compile": compile_telemetry,
            "test_runner": test_runner_telemetry,
            "jev_request": None,
            "jev_verdict": None,
            "qwen_postmortem": qwen_pm,
            "qwen_latency_ms": qwen_lat,
            "qwen_error": qwen_err,
        })

        if current_mech_strikes >= 3:
            subgoal.status = "failed"
            state["current_subgoal"] = subgoal
            if safe_rec:
                safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")
            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="mechanical",
                integration_branch=state.get("integration_branch"),
                thread_id=state.get("thread_id"),
            )
        return state

    # Tier 0 Passed -> Proceed to Tier 1 (Semantic check via Gatekeeper)
    if gatekeeper is None:
        raise ValueError("Gatekeeper must be provided for Tier 1 validation.")

    ev_level = evidence_level(
        compile_outcome=compile_telemetry,
        test_outcome=test_runner_telemetry,
        test_policy=state.get("test_policy"),
    )

    attempt_dep_delta_obj = None
    attempt_dep_delta = None
    if workspace is not None and hasattr(workspace, "get_manifest_changes"):
        try:
            changes = workspace.get_manifest_changes("HEAD")
            delta = compute_dependency_delta(changes)
            attempt_dep_delta_obj = delta
            attempt_dep_delta = delta.to_json()
        except Exception as e:
            logger.warning(f"{type(e).__name__}: {e}")
            attempt_dep_delta_obj = None
            attempt_dep_delta = None

    inv_notes = state.get("investigation_notes")

    dep_delta_flag = state.get("dep_delta_jev", "off")
    pass_delta = False
    if dep_delta_flag == "on" and attempt_dep_delta_obj is not None:
        if (not attempt_dep_delta_obj.is_empty) or bool(attempt_dep_delta_obj.unparsed):
            pass_delta = True

    gate_kwargs: Dict[str, Any] = {}
    if _accepts_param(gatekeeper.validate_subgoal, "investigation_notes"):
        gate_kwargs["investigation_notes"] = inv_notes
    if pass_delta and _accepts_param(gatekeeper.validate_subgoal, "dependency_delta"):
        gate_kwargs["dependency_delta"] = attempt_dep_delta_obj.model_dump()

    verdict = gatekeeper.validate_subgoal(
        subgoal, diff, mechanical_detail=mech_result.detail, **gate_kwargs
    )
    jev_keys = getattr(gatekeeper, "last_state_keys", None)
    if not isinstance(jev_keys, str):
        jev_keys = None

    jev_request_telemetry = {
        "subgoal": subgoal.model_dump() if hasattr(subgoal, "model_dump") else dict(subgoal),
        "diff_size": len(diff),
        "investigation_notes_included": bool(inv_notes),
    }
    jev_verdict_telemetry = {
        "valid": verdict.valid,
        "probability": getattr(verdict, "probability", 1.0),
        "reason": getattr(verdict, "reason", None),
    }

    if verdict.valid:
        wt_path = state.get("current_worktree_path")
        wt_branch = state.get("current_worktree_branch")
        is_worktree_active = _is_active_worktree(state, workspace)

        parent_sha: Optional[str] = None
        commit_sha: Optional[str] = None

        if wt_path and hasattr(workspace, "merge_subgoal_worktree"):
            target_branch = state.get("integration_branch")
            if not target_branch:
                state["gate_status"] = "merge_failed"
                state["status"] = "escalated"
                err_msg = "Worktree merge failed: integration_branch is missing from state"
                state["last_feedback"] = err_msg
                _snapshot_attempt(
                    safe_rec, state, subgoal_id, tier="merge", outcome="merge_failed",
                    diff=diff, feedback_sent=err_msg, evidence_level=ev_level,
                    dependency_delta=attempt_dep_delta, jev_state_keys=jev_keys,
                )
                _teardown_worktree(state, workspace)
                subgoal.status = "failed"
                state["current_subgoal"] = subgoal
                if safe_rec:
                    safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")
                state["trajectory"].append({
                    "node": "gate",
                    "error_type": "missing_integration_branch",
                    "error": err_msg,
                    "gate_status": "merge_failed",
                })
                _safe_escalate_deadlock(
                    gatekeeper,
                    trajectory=state.get("trajectory", []),
                    triggering_tier="merge",
                    thread_id=state.get("thread_id"),
                )
                return state
            if hasattr(workspace, "get_branch_commit"):
                try:
                    parent_sha = workspace.get_branch_commit(target_branch)
                except Exception:
                    parent_sha = None
            try:
                workspace.merge_subgoal_worktree(
                    wt_path,
                    wt_branch,
                    target_branch=target_branch,
                )
                if hasattr(workspace, "get_branch_commit"):
                    try:
                        commit_sha = workspace.get_branch_commit(target_branch)
                    except Exception:
                        commit_sha = None
            except Exception as e:
                state["gate_status"] = "merge_failed"
                state["status"] = "escalated"
                state["last_feedback"] = f"Worktree merge failed: {e}"
                _snapshot_attempt(
                    safe_rec, state, subgoal_id, tier="merge", outcome="merge_failed",
                    diff=diff, feedback_sent=str(e), evidence_level=ev_level,
                    dependency_delta=attempt_dep_delta, jev_state_keys=jev_keys,
                )
                _teardown_worktree(state, workspace)
                subgoal.status = "failed"
                state["current_subgoal"] = subgoal
                if safe_rec:
                    safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")
                state["trajectory"].append({
                    "node": "gate",
                    "error_type": "merge_failure",
                    "error": str(e),
                    "gate_status": "merge_failed",
                })
                _safe_escalate_deadlock(
                    gatekeeper,
                    trajectory=state.get("trajectory", []),
                    triggering_tier="merge",
                    integration_branch=target_branch,
                    thread_id=state.get("thread_id"),
                )
                return state
        elif is_worktree_active:
            err_msg = "Worktree was created but worktree state was missing or invalid at gate time."
            _snapshot_attempt(
                safe_rec, state, subgoal_id, tier="worktree", outcome="worktree_state_missing",
                diff=diff, feedback_sent=err_msg, evidence_level=ev_level,
                dependency_delta=attempt_dep_delta, jev_state_keys=jev_keys,
            )
            _teardown_worktree(state, workspace)
            subgoal.status = "failed"
            state["current_subgoal"] = subgoal
            if safe_rec:
                safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")
            state["gate_status"] = "merge_failed"
            state["status"] = "escalated"
            state["last_feedback"] = err_msg
            state["trajectory"].append({
                "node": "gate",
                "error_type": "worktree_state_missing",
                "error": err_msg,
                "gate_status": "merge_failed",
            })
            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="worktree",
                integration_branch=state.get("integration_branch"),
                thread_id=state.get("thread_id"),
            )
            return state
        elif hasattr(workspace, "commit_subgoal"):
            if hasattr(workspace, "base_commit"):
                parent_sha = workspace.base_commit
            elif hasattr(workspace, "_get_head_commit"):
                parent_sha = workspace._get_head_commit()
            workspace.commit_subgoal()
            if hasattr(workspace, "_get_head_commit"):
                commit_sha = workspace._get_head_commit()

        state["current_worktree_path"] = None
        state["current_worktree_branch"] = None
        state["mechanical_strike_count"] = 0
        state["semantic_strike_count"] = 0
        if "failed_attempts" in state or retry_intel != "off":
            state["failed_attempts"] = []
        state["gate_status"] = "passed"
        state["last_feedback"] = ""
        subgoal.status = "committed"
        state["current_subgoal"] = subgoal

        _snapshot_attempt(
            safe_rec,
            state,
            subgoal_id,
            tier="semantic",
            outcome="passed",
            diff=diff,
            verdict=verdict,
            tests_ran=tests_ran,
            test_outcome=test_runner_telemetry.get("outcome") if isinstance(test_runner_telemetry, dict) else None,
            feedback_sent="",
            evidence_level=ev_level,
            dependency_delta=attempt_dep_delta,
            jev_state_keys=jev_keys,
        )
        if safe_rec:
            safe_rec.record_subgoal_committed(run_id, subgoal_id, parent_sha=parent_sha, commit_sha=commit_sha)

        state["trajectory"].append({
            "node": "gate",
            "subgoal": subgoal.model_dump() if hasattr(subgoal, "model_dump") else dict(subgoal),
            "gate_status": "passed",
            "tier0_result": tier0_telemetry,
            "compile": compile_telemetry,
            "test_runner": test_runner_telemetry,
            "jev_request": jev_request_telemetry,
            "jev_verdict": jev_verdict_telemetry,
        })
        return state
    else:
        current_sem_strikes = state.get("semantic_strike_count", 0) + 1
        state["semantic_strike_count"] = current_sem_strikes
        state["gate_status"] = "semantic_failure"

        has_extra_answers = (
            getattr(verdict, "kind", None) is not None
            or any(
                getattr(verdict, attr, None) is not None
                for attr in ("incomplete", "wrong_behavior", "conflicts_with_notes")
            )
        )
        if retry_intel != "off" and has_extra_answers:
            state["last_feedback"] = _format_typed_semantic_feedback(
                verdict, subgoal, diff, tests_ran
            )
        else:
            prob_str = f" (confidence: {verdict.probability:.2f})" if hasattr(verdict, "probability") and isinstance(verdict.probability, (int, float)) else ""
            state["last_feedback"] = (
                verdict.reason
                or f"Semantic validation rejected by Gatekeeper{prob_str}. "
                   "Diff contradicted project conventions, exceeded declared scope, or failed to implement the required subgoal. "
                   "Ensure changes match the project architecture, dependencies, and styling in Investigation Notes. "
                   "Ensure changes strictly adhere to the application's actual domain, product identity, and branding documented in Investigation Notes, without introducing conflicting domain concepts, alternate sports/business models, or fabricated external entities."
            )

        sem_reason = verdict.reason or state.get("last_feedback") or "Semantic validation rejected by Gatekeeper."
        qwen_pm, qwen_lat, qwen_err = _invoke_qwen_strike_postmortem(
            state=state,
            qwen_client=qwen_client,
            subgoal=subgoal,
            rejecting_source="Tier 1 Jev semantic validation",
            rejection_reason=sem_reason,
            diff=diff,
            current_tier_strikes=current_sem_strikes,
            jev_verdict=verdict,
        )

        if retry_intel != "off":
            _record_failed_attempt_in_state(
                state,
                tier="semantic",
                outcome="semantic_failure",
                failed_check="jev_rejection",
                diff=diff,
            )
        _snapshot_attempt(
            safe_rec,
            state,
            subgoal_id,
            tier="semantic",
            outcome="semantic_failure",
            diff=diff,
            verdict=verdict,
            tests_ran=tests_ran,
            test_outcome=test_runner_telemetry.get("outcome") if isinstance(test_runner_telemetry, dict) else None,
            feedback_sent=state.get("last_feedback"),
            evidence_level=ev_level,
            dependency_delta=attempt_dep_delta,
            jev_state_keys=jev_keys,
            qwen_postmortem=qwen_pm,
            qwen_latency_ms=qwen_lat,
            qwen_error=qwen_err,
        )
        _teardown_worktree(state, workspace)

        if current_sem_strikes >= 3:
            subgoal.status = "failed"
            state["current_subgoal"] = subgoal
            if safe_rec:
                safe_rec.record_subgoal_status(run_id, subgoal_id, "failed")

        state["trajectory"].append({
            "node": "gate",
            "subgoal": subgoal.model_dump() if hasattr(subgoal, "model_dump") else dict(subgoal),
            "gate_status": "semantic_failure",
            "tier0_result": tier0_telemetry,
            "compile": compile_telemetry,
            "test_runner": test_runner_telemetry,
            "jev_request": jev_request_telemetry,
            "jev_verdict": jev_verdict_telemetry,
            "qwen_postmortem": qwen_pm,
            "qwen_latency_ms": qwen_lat,
            "qwen_error": qwen_err,
        })

        if current_sem_strikes >= 3:
            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="semantic",
                integration_branch=state.get("integration_branch"),
                thread_id=state.get("thread_id"),
            )

        return state


def node_verify(
    state: State,
    workspace: Optional[Any] = None,
    gatekeeper: Optional[Any] = None,
    recorder: Optional[Any] = None,
) -> State:
    """State 4: VERIFICATION (Executes full test suite and final ticket verification on a dedicated verify worktree)."""
    safe_rec = SafeRecorder(recorder) if recorder and not isinstance(recorder, SafeRecorder) else recorder
    if "trajectory" not in state or state["trajectory"] is None:
        state["trajectory"] = []

    integration_branch = state.get("integration_branch")
    base_commit = state.get("base_commit")
    ticket = state.get("ticket", "")
    inv_notes = state.get("investigation_notes")

    verify_wt_path = None
    compile_info = None
    test_runner_info = {"ran": False}
    final_diff = ""

    try:
        # 0. Create dedicated detached worktree off the integration branch
        if not integration_branch:
            raise ValueError("integration_branch is required for verification; direct verification on main is disallowed")
        if workspace is not None and hasattr(workspace, "create_verify_worktree"):
            verify_wt_path = workspace.create_verify_worktree(integration_branch)

        if workspace is not None:
            if hasattr(workspace, "get_cumulative_diff"):
                final_diff = workspace.get_cumulative_diff(base_ref=base_commit)
            elif hasattr(workspace, "get_staged_diff"):
                final_diff = workspace.get_staged_diff()

        if not final_diff and workspace is not None and hasattr(workspace, "get_staged_diff"):
            final_diff = workspace.get_staged_diff()

        # 1. Compile check against base_commit
        if workspace is not None and hasattr(workspace, "check_compile"):
            compile_info = workspace.check_compile(diff=final_diff, base_commit=base_commit)

        if compile_info is not None and compile_info.get("outcome") in ("FAILED", "NO_COMPILE_COMMAND", "ENV_NOT_READY"):
            state["gate_status"] = "verification_failed"
            state["status"] = "escalated" if compile_info.get("outcome") in ("NO_COMPILE_COMMAND", "ENV_NOT_READY") else "verification_failed"
            state["last_feedback"] = f"Final compile check failed: {compile_info.get('detail')}"
            _snapshot_attempt(
                safe_rec,
                state,
                subgoal_id="__verify__",
                tier="mechanical",
                outcome="mechanical_failure" if compile_info.get("outcome") == "FAILED" else compile_info.get("outcome", "").lower(),
                failed_check=compile_info.get("outcome"),
                diff=final_diff,
                compile_info=compile_info,
                tests_ran=False,
                feedback_sent=state.get("last_feedback"),
            )
            verify_entry = {
                "node": "verify",
                "gate_status": "verification_failed",
                "status": state["status"],
                "compile": compile_info,
                "test_runner": {"ran": False},
                "jev_request": {
                    "ticket": ticket,
                    "diff_size": len(final_diff),
                    "investigation_notes_included": bool(inv_notes),
                },
                "jev_verdict": None,
            }
            state["trajectory"].append(verify_entry)
            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="mechanical",
                integration_branch=integration_branch,
                thread_id=state.get("thread_id"),
            )
            return state

        # 2. Run unit tests
        test_output = ""
        tests_ran = False
        outcome = None
        test_policy = state.get("test_policy", "auto")
        if test_policy == "never":
            outcome = TestOutcome.NO_TESTS_COLLECTED
            test_output = "NO_TESTS_COLLECTED (Untested pass: tests disabled via test_policy=never)"
            tests_ran = False
        elif workspace is not None and hasattr(workspace, "run_tests"):
            outcome = workspace.run_tests()
            tests_ran = True
            test_output = getattr(outcome, "value", str(outcome))

        if outcome == TestOutcome.ENV_NOT_READY or test_output == "ENV_NOT_READY":
            test_runner_info = getattr(workspace, "last_test_run", None)
            err_detail = (test_runner_info or {}).get("stderr_tail") or "Test runner executable not found."
            state["gate_status"] = "verification_failed"
            state["status"] = "escalated"
            state["last_feedback"] = f"Final test check failed: Environment not ready: {err_detail}"
            _snapshot_attempt(
                safe_rec,
                state,
                subgoal_id="__verify__",
                tier="mechanical",
                outcome="env_not_ready",
                failed_check="env_not_ready",
                diff=final_diff,
                compile_info=compile_info,
                tests_ran=tests_ran,
                test_outcome="ENV_NOT_READY",
                feedback_sent=state.get("last_feedback"),
            )
            verify_entry = {
                "node": "verify",
                "gate_status": "verification_failed",
                "status": "escalated",
                "compile": compile_info,
                "test_runner": test_runner_info,
                "jev_request": {
                    "ticket": ticket,
                    "diff_size": len(final_diff),
                    "investigation_notes_included": bool(inv_notes),
                },
                "jev_verdict": None,
            }
            state["trajectory"].append(verify_entry)
            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="mechanical",
                integration_branch=integration_branch,
                thread_id=state.get("thread_id"),
            )
            return state

        # Determine if any subgoals expected automated tests
        any_expects_tests = False
        for step in state.get("trajectory", []):
            subgoals = step.get("subgoals", [])
            if isinstance(subgoals, list):
                for sg in subgoals:
                    if isinstance(sg, dict) and sg.get("expects_tests", True):
                        any_expects_tests = True
                    elif hasattr(sg, "expects_tests") and getattr(sg, "expects_tests", True):
                        any_expects_tests = True

        if not any_expects_tests and test_output == "NO_TESTS_COLLECTED":
            test_output = "NO_TESTS_COLLECTED (Untested pass: ticket does not require automated tests)"
        elif test_output == "NO_TEST_FRAMEWORK":
            test_output = "NO_TEST_FRAMEWORK (Untested pass: no test framework present in workspace)"

        if not tests_ran:
            test_runner_info = {
                "ran": False,
                "ecosystem": None,
                "command": None,
                "exit_code": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "output_tail": "",
                "outcome": getattr(outcome, "value", str(outcome)) if 'outcome' in locals() and outcome is not None else "NO_TESTS_COLLECTED",
            }
        else:
            test_runner_info = getattr(workspace, "last_test_run", None)
            if test_runner_info is None:
                test_runner_info = {
                    "ran": True,
                    "ecosystem": None,
                    "command": None,
                    "exit_code": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "output_tail": "",
                    "outcome": getattr(outcome, "value", str(outcome)) if 'outcome' in locals() else "UNKNOWN",
                }
            elif isinstance(test_runner_info, dict):
                test_runner_info = dict(test_runner_info)
                test_runner_info["ran"] = True

        if outcome == TestOutcome.FAILED or test_output == "FAILED":
            output_tail = None
            if isinstance(test_runner_info, dict):
                output_tail = test_runner_info.get("output_tail")
            elif hasattr(workspace, "last_test_run") and isinstance(workspace.last_test_run, dict):
                output_tail = workspace.last_test_run.get("output_tail")
            if output_tail:
                tail_capped = output_tail[-4000:]
                test_output = f"FAILED\n{tail_capped}"

        verdict = None
        if gatekeeper is not None and hasattr(gatekeeper, "verify_ticket"):
            if _accepts_param(gatekeeper.verify_ticket, "investigation_notes"):
                verdict = gatekeeper.verify_ticket(ticket, final_diff, test_output, investigation_notes=inv_notes)
            else:
                verdict = gatekeeper.verify_ticket(ticket, final_diff, test_output)

            verify_keys = getattr(gatekeeper, "last_state_keys", None)
            if not isinstance(verify_keys, str):
                verify_keys = None

            if verdict.valid:
                # Discard verify worktree safely first
                if verify_wt_path and hasattr(workspace, "discard_verify_worktree"):
                    workspace.discard_verify_worktree(verify_wt_path)
                    verify_wt_path = None

                # Fast-forward main to integration branch
                if workspace is not None and hasattr(workspace, "fast_forward_main"):
                    workspace.fast_forward_main(integration_branch, base_commit=base_commit)

                # Delete integration branch on successful landing
                if workspace is not None and hasattr(workspace, "delete_integration_branch"):
                    workspace.delete_integration_branch(integration_branch)

                state["gate_status"] = "verified"
                state["status"] = "completed"
                state["last_feedback"] = ""
                verify_ev_level = evidence_level(
                    compile_outcome=compile_info,
                    test_outcome=test_runner_info,
                    test_policy=state.get("test_policy"),
                )
                _snapshot_attempt(
                    safe_rec,
                    state,
                    subgoal_id="__verify__",
                    tier="semantic",
                    outcome="passed",
                    diff=final_diff,
                    compile_info=compile_info,
                    tests_ran=tests_ran,
                    test_outcome=test_runner_info.get("outcome") if isinstance(test_runner_info, dict) else test_output,
                    verdict=verdict,
                    feedback_sent="",
                    evidence_level=verify_ev_level,
                    jev_state_keys=verify_keys,
                )
            else:
                state["gate_status"] = "verification_failed"
                state["status"] = "verification_failed"
                prob_str = f" (confidence: {verdict.probability:.2f})" if hasattr(verdict, "probability") and isinstance(verdict.probability, (int, float)) else ""
                base_fb = verdict.reason or f"Verification rejected by Gatekeeper{prob_str}."
                if outcome == TestOutcome.FAILED or "FAILED" in str(test_output):
                    output_tail = None
                    if isinstance(test_runner_info, dict):
                        output_tail = test_runner_info.get("output_tail")
                    elif hasattr(workspace, "last_test_run") and isinstance(workspace.last_test_run, dict):
                        output_tail = workspace.last_test_run.get("output_tail")
                    if output_tail:
                        tail_capped = output_tail[-4000:]
                        if tail_capped not in base_fb:
                            base_fb = f"{base_fb}\n\nUnit tests failed:\n{tail_capped}"
                state["last_feedback"] = base_fb
                verify_ev_level = evidence_level(
                    compile_outcome=compile_info,
                    test_outcome=test_runner_info,
                    test_policy=state.get("test_policy"),
                )
                _snapshot_attempt(
                    safe_rec,
                    state,
                    subgoal_id="__verify__",
                    tier="semantic",
                    outcome="semantic_failure",
                    diff=final_diff,
                    compile_info=compile_info,
                    tests_ran=tests_ran,
                    test_outcome=test_runner_info.get("outcome") if isinstance(test_runner_info, dict) else test_output,
                    verdict=verdict,
                    feedback_sent=state.get("last_feedback"),
                    evidence_level=verify_ev_level,
                    jev_state_keys=verify_keys,
                )
        else:
            if verify_wt_path and hasattr(workspace, "discard_verify_worktree"):
                workspace.discard_verify_worktree(verify_wt_path)
                verify_wt_path = None
            if workspace is not None and hasattr(workspace, "fast_forward_main"):
                workspace.fast_forward_main(integration_branch, base_commit=base_commit)
            if workspace is not None and hasattr(workspace, "delete_integration_branch"):
                workspace.delete_integration_branch(integration_branch)
            state["gate_status"] = "verified"
            state["status"] = "completed"
            verify_ev_level = evidence_level(
                compile_outcome=compile_info,
                test_outcome=test_runner_info,
                test_policy=state.get("test_policy"),
            )
            _snapshot_attempt(
                safe_rec,
                state,
                subgoal_id="__verify__",
                tier="semantic",
                outcome="passed",
                diff=final_diff,
                compile_info=compile_info,
                tests_ran=tests_ran,
                test_outcome=test_runner_info.get("outcome") if isinstance(test_runner_info, dict) else test_output,
                verdict=None,
                feedback_sent="",
                evidence_level=verify_ev_level,
            )

        verify_entry: Dict[str, Any] = {
            "node": "verify",
            "gate_status": state.get("gate_status"),
            "status": state.get("status"),
            "compile": compile_info,
            "test_runner": test_runner_info,
            "jev_request": {
                "ticket": ticket,
                "diff_size": len(final_diff),
                "investigation_notes_included": bool(inv_notes),
            },
            "jev_verdict": {
                "valid": verdict.valid,
                "probability": getattr(verdict, "probability", 1.0),
                "reason": getattr(verdict, "reason", None),
            } if verdict is not None else None,
        }
        if verdict is not None:
            if getattr(verdict, "probability", None) is not None:
                verify_entry["probability"] = verdict.probability
            if getattr(verdict, "reason", None):
                verify_entry["reason"] = verdict.reason

        state["trajectory"].append(verify_entry)

        if state.get("gate_status") == "verification_failed":
            _safe_escalate_deadlock(
                gatekeeper,
                trajectory=state.get("trajectory", []),
                triggering_tier="verification",
                integration_branch=integration_branch,
                thread_id=state.get("thread_id"),
            )
        return state

    except (MainDivergedError, TrackedModificationsError) as e:
        status_key = "main_diverged" if isinstance(e, MainDivergedError) else "tracked_modifications"
        state["gate_status"] = status_key
        state["status"] = "escalated"
        state["last_feedback"] = str(e)
        _snapshot_attempt(
            safe_rec,
            state,
            subgoal_id="__verify__",
            tier="git",
            outcome=status_key,
            diff=final_diff,
            compile_info=compile_info if 'compile_info' in locals() else None,
            feedback_sent=str(e),
        )
        state["trajectory"].append({
            "node": "verify",
            "error_type": status_key,
            "error": str(e),
            "gate_status": status_key,
            "status": "escalated",
            "compile": compile_info,
            "test_runner": test_runner_info,
            "jev_request": {
                "ticket": ticket,
                "diff_size": len(final_diff),
                "investigation_notes_included": bool(inv_notes),
            },
            "jev_verdict": None,
        })
        _safe_escalate_deadlock(
            gatekeeper,
            trajectory=state.get("trajectory", []),
            triggering_tier="verification",
            integration_branch=integration_branch,
            thread_id=state.get("thread_id"),
        )
        return state

    except Exception as e:
        state["gate_status"] = "verification_failed"
        state["status"] = "escalated"
        state["last_feedback"] = f"Verification error: {str(e)}"
        _snapshot_attempt(
            safe_rec,
            state,
            subgoal_id="__verify__",
            tier="verify",
            outcome="verification_exception",
            diff=final_diff,
            compile_info=compile_info if 'compile_info' in locals() else getattr(workspace, "last_compile_run", None),
            feedback_sent=f"Verification error: {str(e)}",
        )
        state["trajectory"].append({
            "node": "verify",
            "error_type": "verification_exception",
            "error": f"Verification error: {str(e)}",
            "gate_status": "verification_failed",
            "status": "escalated",
            "compile": compile_info if 'compile_info' in locals() else getattr(workspace, "last_compile_run", None),
            "test_runner": test_runner_info,
            "jev_request": {
                "ticket": ticket,
                "diff_size": len(final_diff),
                "investigation_notes_included": bool(inv_notes),
            },
            "jev_verdict": None,
        })
        _safe_escalate_deadlock(
            gatekeeper,
            trajectory=state.get("trajectory", []),
            triggering_tier="verification",
            integration_branch=integration_branch,
            thread_id=state.get("thread_id"),
        )
        return state

    finally:
        if verify_wt_path and workspace is not None and hasattr(workspace, "discard_verify_worktree"):
            workspace.discard_verify_worktree(verify_wt_path)



def node_escalate(
    state: State,
    workspace: Optional[Any] = None,
    gatekeeper: Optional[Any] = None,
) -> State:
    """Escalation termination node for HITL review."""
    state["status"] = "escalated"
    if workspace is not None:
        is_wt = _is_active_worktree(state, workspace)
        wt_path, wt_branch = _get_active_worktree_info(state, workspace)
        if is_wt and hasattr(workspace, "discard_subgoal_worktree"):
            workspace.discard_subgoal_worktree(wt_path, wt_branch)
        if hasattr(workspace, "repo_dir") and isinstance(workspace.repo_dir, Path):
            workspace.worktree_dir = workspace.repo_dir
    state["current_worktree_path"] = None
    state["current_worktree_branch"] = None
    return state


def route_plan(state: State) -> str:
    """Evaluates whether planning succeeded or failed to determine next node."""
    if state.get("gate_status") == "planning_failed":
        return "escalate"
    return "implement"


def route_gate(state: State) -> str:
    """Evaluates strike counters and plan queue to determine the next graph node."""
    if (
        state.get("mechanical_strike_count", 0) >= 3
        or state.get("semantic_strike_count", 0) >= 3
        or state.get("gate_status") == "merge_failed"
        or state.get("status") == "escalated"
    ):
        return "escalate"
    if state.get("gate_status") == "passed":
        if state.get("plan_queue"):
            return "implement"
        return "verify"
    return "implement"


class JevEngine:
    """The FSM Engine orchestrating the LangGraph DAG with Two-Tier Gating."""

    def __init__(
        self,
        workspace: Optional[Any] = None,
        gatekeeper: Optional[Any] = None,
        llm: Optional[Any] = None,
        db_path: Optional[Union[str, Path]] = "checkpoints.db",
        checkpointer: Optional[BaseCheckpointSaver] = None,
        test_policy: str = "auto",
        recorder: Optional[Any] = None,
        retry_intel: str = "on",
        repo_profile: str = "off",
        dep_delta_jev: str = "off",
        max_investigation_turns: int = 20,
        qwen_client: Optional[Any] = None,
    ):
        self.workspace = workspace
        self.gatekeeper = gatekeeper
        self.llm = llm
        self.test_policy = test_policy
        self.retry_intel = retry_intel
        self.repo_profile = repo_profile
        self.dep_delta_jev = dep_delta_jev
        self.max_investigation_turns = max_investigation_turns
        self.qwen_client = qwen_client
        if recorder is None:
            self.recorder = SafeRecorder(NullRecorder())
        elif isinstance(recorder, SafeRecorder):
            self.recorder = recorder
        else:
            self.recorder = SafeRecorder(recorder)

        if checkpointer is not None:
            self.checkpointer = checkpointer
        elif db_path is not None:
            self.checkpointer = SqliteSaver.from_conn_string(str(db_path))
        else:
            self.checkpointer = None

        self.app = self.build_graph()

    def _node_investigate(self, state: State) -> State:
        return node_investigate(
            state,
            workspace=self.workspace,
            llm=self.llm,
            max_investigation_turns=self.max_investigation_turns,
        )

    def _node_plan(self, state: State) -> State:
        return node_plan(state, workspace=self.workspace, llm=self.llm, gatekeeper=self.gatekeeper, recorder=self.recorder)

    def _node_implement(self, state: State) -> State:
        return node_implement(state, workspace=self.workspace, llm=self.llm, gatekeeper=self.gatekeeper)

    def node_gate(self, state: State) -> State:
        return node_gate(
            state,
            workspace=self.workspace,
            gatekeeper=self.gatekeeper,
            recorder=self.recorder,
            qwen_client=self.qwen_client,
        )

    def _node_verify(self, state: State) -> State:
        return node_verify(state, workspace=self.workspace, gatekeeper=self.gatekeeper, recorder=self.recorder)

    def _node_escalate(self, state: State) -> State:
        return node_escalate(state, workspace=self.workspace, gatekeeper=self.gatekeeper)

    def _route_after_plan(self, state: State) -> str:
        return route_plan(state)

    def _route_after_gate(self, state: State) -> str:
        return route_gate(state)

    def _route_after_verify(self, state: State) -> str:
        if state.get("gate_status") == "verified":
            return END
        return "escalate"

    def build_graph(self) -> Any:
        builder = StateGraph(State)

        builder.add_node("investigate", self._node_investigate)
        builder.add_node("plan", self._node_plan)
        builder.add_node("implement", self._node_implement)
        builder.add_node("gate", self.node_gate)
        builder.add_node("verify", self._node_verify)
        builder.add_node("escalate", self._node_escalate)

        builder.add_edge(START, "investigate")
        builder.add_edge("investigate", "plan")

        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "implement": "implement",
                "escalate": "escalate",
            },
        )
        builder.add_edge("implement", "gate")


        builder.add_conditional_edges(
            "gate",
            self._route_after_gate,
            {
                "implement": "implement",
                "verify": "verify",
                "escalate": "escalate",
            },
        )

        builder.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {
                END: END,
                "escalate": "escalate",
            },
        )

        builder.add_edge("escalate", END)

        return builder.compile(checkpointer=self.checkpointer)

    def execute(self, ticket: str, thread_id: Optional[str] = None) -> State:
        if thread_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            rnd = uuid.uuid4().hex[:4]
            thread_id = f"run-{ts}-{rnd}"

        run_id = thread_id
        if self.recorder.has_run(run_id):
            run_id = f"{thread_id}_{uuid.uuid4().hex[:6]}"

        base_commit = None
        if self.workspace is not None:
            if hasattr(self.workspace, "base_commit") and self.workspace.base_commit:
                base_commit = self.workspace.base_commit
            elif hasattr(self.workspace, "_get_head_commit"):
                base_commit = self.workspace._get_head_commit()

        integration_branch = f"jev-ticket-{thread_id}"

        has_checkpoint = False
        if self.checkpointer is not None:
            try:
                cp = self.checkpointer.get({"configurable": {"thread_id": thread_id}})
                if cp is not None:
                    has_checkpoint = True
            except Exception:
                pass

        branch_exists = False
        if self.workspace is not None and hasattr(self.workspace, "_run_repo_git"):
            try:
                res = self.workspace._run_repo_git(["rev-parse", "--verify", integration_branch])
                if res.returncode == 0:
                    branch_exists = True
            except Exception:
                pass

        if has_checkpoint or branch_exists:
            warning_msg = (
                f"WARNING: Thread '{thread_id}' already has existing checkpoints or integration branch '{integration_branch}'. "
                f"Restarting thread from scratch and deleting prior integration branch (no resume)."
            )
            print(warning_msg, file=sys.stderr, flush=True)

        if self.workspace is not None and hasattr(self.workspace, "create_integration_branch"):
            self.workspace.create_integration_branch(integration_branch, base_commit=base_commit)

        worker_model = None
        if self.llm is not None:
            worker_model = getattr(self.llm, "model", None) or getattr(self.llm, "model_name", None)
            if not worker_model and hasattr(self.llm, "__class__"):
                worker_model = self.llm.__class__.__name__

        profile_text = None
        profile_chars = None
        if self.repo_profile == "on":
            try:
                from jev.repo_profile import get_or_build_repo_profile

                main_repo_dir = self.workspace.repo_dir if self.workspace is not None else Path.cwd()
                prof = get_or_build_repo_profile(main_repo_dir)
                rendered = prof.to_prompt_text()
                profile_text = rendered if rendered else None
                profile_chars = len(rendered) if rendered else None
            except Exception as e:
                logger.warning(f"{type(e).__name__}: {e}")
                profile_text = None
                profile_chars = None

        engine_commit = _get_engine_commit()
        self.recorder.record_run_start(
            run_id=run_id,
            thread_id=thread_id,
            ticket=ticket,
            base_commit=base_commit,
            engine_commit=engine_commit,
            worker_model=worker_model,
            test_policy=self.test_policy,
            retry_intel=self.retry_intel,
            repo_profile=self.repo_profile,
            profile_chars=profile_chars,
            dep_delta_jev=self.dep_delta_jev,
        )

        initial_state: State = {
            "ticket": ticket,
            "thread_id": thread_id,
            "run_id": run_id,
            "base_commit": base_commit,
            "integration_branch": integration_branch,
            "subgoal_base_commit": base_commit,
            "test_policy": self.test_policy,
            "retry_intel": self.retry_intel,
            "repo_profile": self.repo_profile,
            "repo_profile_text": profile_text,
            "dep_delta_jev": self.dep_delta_jev,
            "failed_attempts": [],
            "plan_queue": [],
            "current_subgoal": None,
            "mechanical_strike_count": 0,
            "semantic_strike_count": 0,
            "trajectory": [],
            "gate_status": None,
            "last_feedback": None,
            "status": None,
            "strike_history": [],
            "total_strikes": 0,
        }

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        final_state = self.app.invoke(initial_state, config=config)

        final_status = final_state.get("status") or "unknown"
        final_commit = None
        if self.workspace is not None:
            if hasattr(self.workspace, "_get_head_commit"):
                try:
                    final_commit = self.workspace._get_head_commit()
                except Exception:
                    pass
            if not final_commit and hasattr(self.workspace, "base_commit"):
                final_commit = self.workspace.base_commit

        inv_notes = final_state.get("investigation_notes") or ""
        notes_chars = len(inv_notes)

        run_dep_delta = None
        if final_status == "completed" and final_commit and base_commit:
            if final_commit != base_commit:
                if self.workspace is not None and hasattr(self.workspace, "get_manifest_changes"):
                    try:
                        changes = self.workspace.get_manifest_changes(base_commit, final_commit)
                        delta = compute_dependency_delta(changes)
                        run_dep_delta = delta.to_json()
                    except Exception as e:
                        logger.warning(f"{type(e).__name__}: {e}")
                        run_dep_delta = None
            else:
                run_dep_delta = compute_dependency_delta({}).to_json()

        self.recorder.record_run_end(
            run_id=run_id,
            final_status=final_status,
            final_commit=final_commit,
            notes_chars=notes_chars,
            dependency_delta=run_dep_delta,
        )

        return final_state
