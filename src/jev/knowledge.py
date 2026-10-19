"""Phase 4b-2: KnowledgeStore, evidence capture, staleness tracking, and NullMemory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from jev.dependency_delta import LOCKFILES, MANIFEST_BASENAMES
from jev.ledger import compute_repo_key, get_jev_home
from jev.memory_extract import CompileTrap

logger = logging.getLogger(__name__)


def get_knowledge_db_path(repo_key: str) -> Path:
    """Returns the default knowledge.db path for a repository key: <JEV_HOME>/repos/<repo_key>/knowledge.db."""
    return get_jev_home() / "repos" / repo_key / "knowledge.db"


def is_env_file(basename: str) -> bool:
    """Check if a file basename is an environment or manifest file."""
    lower = basename.lower()
    if basename in MANIFEST_BASENAMES or lower in {b.lower() for b in MANIFEST_BASENAMES}:
        return True
    if basename in LOCKFILES or lower in {b.lower() for b in LOCKFILES}:
        return True
    if basename in ("setup.py", "setup.cfg", "settings.gradle", "settings.gradle.kts"):
        return True
    if lower.startswith("requirements") and lower.endswith(".txt"):
        return True
    if lower.startswith("tsconfig") and lower.endswith(".json"):
        return True
    return False


def compute_fact_id(kind: str, code: str, file: str, message: str) -> str:
    """id = first 16 hex of sha256 over kind, code, file, normalized message joined by a NUL character."""
    payload = f"{kind}\0{code}\0{file}\0{message}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def capture_evidence(
    repo_dir: Union[str, Path],
    base_commit: Optional[str],
    concerned_paths: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, str], str]:
    """Capture environment and concerned file hashes at base_commit.

    Returns (evidence_dict, status) where status is 'active' or 'unverifiable'.
    """
    if not base_commit:
        return ({}, "unverifiable")

    try:
        res = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", str(base_commit)],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return ({}, "unverifiable")
    except Exception:
        return ({}, "unverifiable")

    all_files = set(line.strip() for line in res.stdout.splitlines() if line.strip())

    if concerned_paths:
        ancestor_dirs = set()
        for cp in concerned_paths:
            p = Path(cp.replace("\\", "/"))
            cur = p.parent
            while True:
                as_str = cur.as_posix()
                if as_str == ".":
                    as_str = ""
                ancestor_dirs.add(as_str)
                if as_str == "":
                    break
                cur = cur.parent
    else:
        ancestor_dirs = {""}

    matched_files = set()
    if concerned_paths:
        for cp in concerned_paths:
            cp_norm = Path(cp.replace("\\", "/")).as_posix()
            if cp_norm in all_files:
                matched_files.add(cp_norm)

    for f in all_files:
        f_norm = Path(f.replace("\\", "/")).as_posix()
        parent = Path(f_norm).parent.as_posix()
        if parent == ".":
            parent = ""
        if parent in ancestor_dirs:
            if is_env_file(Path(f_norm).name):
                matched_files.add(f_norm)

    selected_files = sorted(matched_files)[:20]

    evidence_dict: Dict[str, str] = {}
    for p in selected_files:
        try:
            show_res = subprocess.run(
                ["git", "show", f"{base_commit}:{p}"],
                cwd=str(repo_dir),
                capture_output=True,
            )
            if show_res.returncode != 0:
                return ({}, "unverifiable")
            evidence_dict[p] = hashlib.sha256(show_res.stdout).hexdigest()
        except Exception:
            return ({}, "unverifiable")

    return (evidence_dict, "active")


class UpsertResult(dict):
    """Result returned by upsert_traps."""

    def __init__(self, inserted: int = 0, updated: int = 0, skipped: int = 0):
        super().__init__(inserted=inserted, updated=updated, skipped=skipped)
        self.inserted = inserted
        self.updated = updated
        self.skipped = skipped


class StatusCounts(dict):
    """Result returned by refresh_status."""

    def __init__(self, active: int = 0, stale: int = 0, unverifiable: int = 0):
        super().__init__(active=active, stale=stale, unverifiable=unverifiable)
        self.active = active
        self.stale = stale
        self.unverifiable = unverifiable


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol satisfied by KnowledgeStore and NullMemory."""

    def upsert_traps(
        self, traps: Sequence[CompileTrap], repo_dir: Union[str, Path]
    ) -> UpsertResult: ...

    def is_stale(
        self, fact_row: Mapping[str, Any], repo_dir: Union[str, Path]
    ) -> Tuple[bool, str]: ...

    def refresh_status(self, repo_dir: Union[str, Path]) -> StatusCounts: ...

    def list_facts(self, status: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def forget(self, fact_id: Optional[str] = None) -> None: ...

    def count(self) -> int: ...


class KnowledgeStore:
    """Persistent SQLite-backed knowledge store for CompileTrap facts."""

    def __init__(
        self,
        db_path: Union[str, Path],
        repo_key: str,
        ttl_days: int = 30,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.db_path = Path(db_path)
        self.repo_key = repo_key
        self.ttl_days = ttl_days
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        self._init_db()

    def _now(self) -> datetime:
        dt = self._clock()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _init_db(self) -> None:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if self.db_path.exists():
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    with conn:
                        cur = conn.cursor()
                        cur.execute("PRAGMA quick_check")
                        row = cur.fetchone()
                        if not row or row[0] != "ok":
                            raise sqlite3.DatabaseError(f"Corrupt database quick_check: {row}")
                        cur.execute("PRAGMA user_version")
                    conn.close()
                except Exception as e:
                    logger.warning("Corrupt knowledge.db detected at %s, recreating empty: %s", self.db_path, e)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if self.db_path.exists():
                        try:
                            self.db_path.unlink()
                        except OSError:
                            pass

        try:
            with self._connect(readonly=False) as conn:
                conn.execute("PRAGMA user_version = 1")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS facts (
                        id TEXT PRIMARY KEY,
                        repo_key TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        code TEXT,
                        file TEXT,
                        message TEXT,
                        statement TEXT,
                        fix_hint TEXT,
                        count INTEGER NOT NULL,
                        provenance_json TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        injected_count INTEGER DEFAULT 0,
                        helped_count INTEGER DEFAULT 0
                    )
                    """
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            if "readonly" in str(e).lower():
                pass
            else:
                raise

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if str(self.db_path) == ":memory:":
            conn = sqlite3.connect(":memory:")
        elif readonly:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_traps(
        self, traps: Sequence[CompileTrap], repo_dir: Union[str, Path]
    ) -> UpsertResult:
        inserted = 0
        updated = 0
        skipped = 0
        now_iso = self._now().isoformat()

        with self._connect(readonly=False) as conn:
            cur = conn.cursor()
            for trap in traps:
                if trap.repo_key is not None and trap.repo_key != self.repo_key:
                    skipped += 1
                    continue

                fact_id = compute_fact_id(
                    trap.kind, trap.signature[0], trap.signature[1], trap.signature[2]
                )
                concerned_paths = (
                    trap.evidence_paths
                    if trap.evidence_paths
                    else ([trap.signature[1]] if trap.signature[1] else [])
                )
                evidence_dict, ev_status = capture_evidence(
                    repo_dir, trap.provenance.base_commit, concerned_paths
                )
                status = ev_status

                cur.execute(
                    "SELECT first_seen, last_seen, count, provenance_json, injected_count, helped_count FROM facts WHERE id = ?",
                    (fact_id,),
                )
                existing = cur.fetchone()

                prov_json = json.dumps(trap.provenance.model_dump(), sort_keys=True)
                ev_json = json.dumps(evidence_dict, sort_keys=True)

                if existing:
                    updated += 1
                    first_seen = existing["first_seen"]
                    injected_count = existing["injected_count"]
                    helped_count = existing["helped_count"]
                    # If same count and same provenance, last_seen is unchanged (leaving table byte-identical apart from updated_at)
                    if existing["count"] == trap.count and existing["provenance_json"] == prov_json:
                        last_seen = existing["last_seen"]
                    else:
                        last_seen = now_iso
                else:
                    inserted += 1
                    first_seen = now_iso
                    last_seen = now_iso
                    injected_count = 0
                    helped_count = 0

                cur.execute(
                    """
                    INSERT INTO facts (
                        id, repo_key, kind, code, file, message, statement, fix_hint,
                        count, provenance_json, evidence_json, status,
                        first_seen, last_seen, updated_at, injected_count, helped_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        repo_key=excluded.repo_key,
                        kind=excluded.kind,
                        code=excluded.code,
                        file=excluded.file,
                        message=excluded.message,
                        statement=excluded.statement,
                        fix_hint=excluded.fix_hint,
                        count=excluded.count,
                        provenance_json=excluded.provenance_json,
                        evidence_json=excluded.evidence_json,
                        status=excluded.status,
                        last_seen=excluded.last_seen,
                        updated_at=excluded.updated_at
                    """,
                    (
                        fact_id,
                        self.repo_key,
                        trap.kind,
                        trap.signature[0],
                        trap.signature[1],
                        trap.signature[2],
                        trap.statement,
                        trap.fix_hint,
                        trap.count,
                        prov_json,
                        ev_json,
                        status,
                        first_seen,
                        last_seen,
                        now_iso,
                        injected_count,
                        helped_count,
                    ),
                )
            conn.commit()

        return UpsertResult(inserted=inserted, updated=updated, skipped=skipped)

    def is_stale(
        self, fact_row: Mapping[str, Any], repo_dir: Union[str, Path]
    ) -> Tuple[bool, str]:
        if fact_row.get("status") == "unverifiable":
            return (True, "unverifiable")

        last_seen_str = fact_row.get("last_seen")
        if last_seen_str:
            last_seen_dt = datetime.fromisoformat(last_seen_str)
            if last_seen_dt.tzinfo is None:
                last_seen_dt = last_seen_dt.replace(tzinfo=timezone.utc)
            now = self._now()
            if (now - last_seen_dt) > timedelta(days=self.ttl_days):
                return (True, "ttl")

        try:
            res = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                return (True, "unverifiable")
            head_files = set(line.strip() for line in res.stdout.splitlines() if line.strip())
        except Exception:
            return (True, "unverifiable")

        ev_raw = fact_row.get("evidence_json", "{}")
        try:
            evidence_dict = json.loads(ev_raw) if isinstance(ev_raw, str) else dict(ev_raw)
        except Exception:
            evidence_dict = {}

        # 1. Environment files: check for diff or missing
        env_paths = sorted([p for p in evidence_dict if is_env_file(Path(p).name)])
        for p in env_paths:
            if p not in head_files:
                return (True, f"env_changed:{p}")
            try:
                show_res = subprocess.run(
                    ["git", "show", f"HEAD:{p}"],
                    cwd=str(repo_dir),
                    capture_output=True,
                )
                if show_res.returncode != 0:
                    return (True, f"env_changed:{p}")
                cur_hash = hashlib.sha256(show_res.stdout).hexdigest()
                if cur_hash != evidence_dict[p]:
                    return (True, f"env_changed:{p}")
            except Exception:
                return (True, f"env_changed:{p}")

        # 2. Concerned paths: check if missing (content changes do not make it stale)
        concerned = set()
        if fact_row.get("file"):
            concerned.add(fact_row["file"])
        for p in evidence_dict:
            if not is_env_file(Path(p).name):
                concerned.add(p)

        for p in sorted(concerned):
            if p not in head_files:
                return (True, f"file_missing:{p}")

        return (False, "")

    def refresh_status(self, repo_dir: Union[str, Path]) -> StatusCounts:
        active_count = 0
        stale_count = 0
        unverifiable_count = 0
        now_iso = self._now().isoformat()

        with self._connect(readonly=False) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM facts")
            rows = [dict(r) for r in cur.fetchall()]

            for row in rows:
                stale, reason = self.is_stale(row, repo_dir)
                if reason == "unverifiable":
                    new_status = "unverifiable"
                elif stale:
                    new_status = "stale"
                else:
                    new_status = "active"

                if new_status == "active":
                    active_count += 1
                elif new_status == "stale":
                    stale_count += 1
                else:
                    unverifiable_count += 1

                if new_status != row["status"]:
                    cur.execute(
                        "UPDATE facts SET status = ?, updated_at = ? WHERE id = ?",
                        (new_status, now_iso, row["id"]),
                    )
            conn.commit()

        return StatusCounts(
            active=active_count, stale=stale_count, unverifiable=unverifiable_count
        )

    def list_facts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connect(readonly=True) as conn:
            cur = conn.cursor()
            if status is not None:
                cur.execute(
                    "SELECT * FROM facts WHERE status = ? ORDER BY last_seen DESC, id ASC",
                    (status,),
                )
            else:
                cur.execute("SELECT * FROM facts ORDER BY last_seen DESC, id ASC")
            return [dict(r) for r in cur.fetchall()]

    def forget(self, fact_id: Optional[str] = None) -> None:
        with self._connect(readonly=False) as conn:
            cur = conn.cursor()
            if fact_id is None:
                cur.execute("DELETE FROM facts")
            else:
                cur.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            conn.commit()

    def count(self) -> int:
        with self._connect(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM facts")
            row = cur.fetchone()
            return row[0] if row else 0


class NullMemory:
    """No-op implementation of MemoryStore that returns empty results."""

    def upsert_traps(
        self, traps: Sequence[CompileTrap], repo_dir: Union[str, Path]
    ) -> UpsertResult:
        return UpsertResult(inserted=0, updated=0, skipped=len(traps))

    def is_stale(
        self, fact_row: Mapping[str, Any], repo_dir: Union[str, Path]
    ) -> Tuple[bool, str]:
        return (False, "")

    def refresh_status(self, repo_dir: Union[str, Path]) -> StatusCounts:
        return StatusCounts(active=0, stale=0, unverifiable=0)

    def list_facts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def forget(self, fact_id: Optional[str] = None) -> None:
        pass

    def count(self) -> int:
        return 0
