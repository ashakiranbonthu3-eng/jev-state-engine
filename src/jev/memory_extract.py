"""Phase 4a: Pure, read-only memory extractor for CompileTrap facts.

Extracts deterministic, deduplicated compile error facts from ledger tables.
Contains no LLM imports, no network access, and makes no writes to any database.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class CompileTrapProvenance(BaseModel):
    """Provenance metadata tracking where the compile trap was observed and resolved."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    subgoal_id: str
    attempt_no: int
    fix_attempt_no: Optional[int] = None
    base_commit: Optional[str] = None
    fix_commit: Optional[str] = None


class CompileTrap(BaseModel):
    """A frozen factual record of a compile error encountered during Tier 0."""

    model_config = ConfigDict(frozen=True)

    kind: str = "compile_trap"
    repo_key: Optional[str] = None
    signature: Tuple[str, str, str]  # (code, file, normalized_message)
    statement: str
    fix_hint: Optional[str] = None
    count: int = 1
    provenance: CompileTrapProvenance
    evidence_hashes: Dict[str, str] = Field(default_factory=dict)


def _cut_diff(diff: Optional[str], max_lines: int = 40, max_chars: int = 3000) -> Optional[str]:
    """Excerpt diff up to max_lines and max_chars."""
    if not diff or not isinstance(diff, str):
        return None
    lines = diff.splitlines(keepends=True)
    excerpt = "".join(lines[:max_lines])
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    return excerpt or None


def _quote_temp_paths(text: str) -> str:
    """Quotes unquoted path-like tokens that appear to be temporary directories."""
    temp_pattern = re.compile(
        r'(?<!["\'])(?:(?:[A-Za-z]:)?[\\/][^\s\'"]*?(?:tmp|temp|pytest-[a-zA-Z0-9_.-]+)[\\/][^\s\'"]+)(?!["\'])'
    )
    return temp_pattern.sub(lambda m: f'"{m.group(0)}"', text)


def normalize_error(
    code: Optional[str],
    file: Optional[str],
    message: Optional[str],
    known_files: Optional[Sequence[str]] = None,
) -> Tuple[str, str, str]:
    """Normalize code, file, and error message into a canonical signature."""
    norm_code = (code or "").strip() or "tier0_failure"

    raw_file = (file or "").strip().replace("\\", "/")
    # Strip line and column numbers from the file if present (e.g. foo.py:12:34)
    raw_file = re.sub(r':\d+(?::\d+)?$', '', raw_file)

    norm_file = raw_file
    if known_files and norm_file:
        for kf in sorted(known_files, key=len, reverse=True):
            kf_posix = kf.replace("\\", "/")
            if norm_file.endswith(kf_posix):
                norm_file = kf_posix
                break
    elif norm_file:
        # Strip absolute prefix if present
        m = re.search(r'/(?:repo|workspace|project|src|lib|tests|packages|apps)/(.+)$', norm_file)
        if m:
            matched_segment = norm_file[m.start() + 1 :]
            norm_file = matched_segment
        else:
            # Strip drive letter or leading root
            norm_file = re.sub(r'^(?:[A-Za-z]:)?[\\/]+', '', norm_file)
            # If path still contains temp directory prefixes, strip up to temp path
            norm_file = re.sub(r'^.*?/(?:tmp|temp|pytest-[^/]+)/', '', norm_file)

    norm_file = norm_file.strip()

    raw_msg = (message or "").strip()

    # 1. Strip absolute path prefixes in message to match repo-relative form
    if known_files:
        for kf in sorted(known_files, key=len, reverse=True):
            kf_posix = kf.replace("\\", "/")
            pattern = re.compile(r'(?:[A-Za-z]:)?[^\s\'"]*?/' + re.escape(kf_posix))
            raw_msg = pattern.sub(kf_posix, raw_msg)
    if norm_file:
        pattern = re.compile(r'(?:[A-Za-z]:)?[^\s\'"]*?/' + re.escape(norm_file))
        raw_msg = pattern.sub(norm_file, raw_msg)

    # 2. Strip line and column numbers
    # Remove :line:col or :line after words/filenames
    raw_msg = re.sub(r'(?<=[A-Za-z0-9_./-])(?::\d+){1,2}\b:?', '', raw_msg)
    # Remove "line 12, col 34" or "line 12"
    raw_msg = re.sub(r'\bline\s+\d+(?:,\s*(?:col|column)\s*\d+)?\b', '', raw_msg, flags=re.IGNORECASE)
    raw_msg = re.sub(r'\b(?:col|column)\s+\d+\b', '', raw_msg, flags=re.IGNORECASE)
    raw_msg = re.sub(r'\(\s*\d+\s*,\s*\d+\s*\)', '', raw_msg)

    # 3. Quotes content that looks like a temp path
    raw_msg = _quote_temp_paths(raw_msg)

    # 4. Collapse whitespace
    norm_msg = re.sub(r'\s+', ' ', raw_msg).strip()

    return norm_code, norm_file, norm_msg


def _build_statement(code: str, file: str, normalized_message: str) -> str:
    """Construct statement from standard template without LLM text."""
    effective_code = code or "tier0_failure"
    if file:
        return f"Attempt failed Tier 0 with {effective_code} in {file}: {normalized_message}"
    return f"Attempt failed Tier 0 with {effective_code}: {normalized_message}"


def _extract_file_content_from_attempt(
    file_path: str, files_json_raw: Any, diff_raw: Any
) -> Optional[str]:
    """Extract content of the specified file from attempt metadata if present."""
    if not file_path:
        return None

    norm_target = file_path.replace("\\", "/")

    # Check files_json if it is a dictionary of content
    if files_json_raw:
        try:
            parsed = json.loads(files_json_raw) if isinstance(files_json_raw, str) else files_json_raw
            if isinstance(parsed, dict) and norm_target in parsed:
                val = parsed[norm_target]
                if isinstance(val, str):
                    return val
        except Exception:
            pass

    # Check diff for new file additions
    if diff_raw and isinstance(diff_raw, str):
        target_name = Path(norm_target).name
        if target_name in diff_raw or norm_target in diff_raw:
            sections = re.split(r'\ndiff --git ', "\n" + diff_raw)
            for sec in sections:
                if not sec.strip():
                    continue
                header_line = sec.splitlines()[0] if sec.splitlines() else ""
                if norm_target in header_line or target_name in header_line:
                    # Collect additions
                    additions = []
                    for line in sec.splitlines():
                        if line.startswith("+") and not line.startswith("+++"):
                            additions.append(line[1:])
                    if additions:
                        return "\n".join(additions)

    return None


def extract_compile_traps(
    attempts: Optional[Sequence[Mapping[str, Any]]] = None,
    subgoals: Optional[Sequence[Mapping[str, Any]]] = None,
    runs: Optional[Sequence[Mapping[str, Any]]] = None,
    repo_key: Optional[str] = None,
    *args: Any,
    **kwargs: Any,
) -> List[CompileTrap]:
    """Pure, read-only extraction of CompileTrap facts from ledger mappings."""
    actual_attempts: List[Mapping[str, Any]] = list(attempts) if attempts is not None else []
    actual_subgoals: List[Mapping[str, Any]] = list(subgoals) if subgoals is not None else []
    actual_runs: List[Mapping[str, Any]] = list(runs) if runs is not None else []

    # If positional args were provided, distribute any that were not set
    for item in args:
        if not isinstance(item, (list, tuple)):
            continue
        if not item:
            continue
        sample = item[0]
        if isinstance(sample, (dict, Mapping, sqlite3.Row)):
            if "attempt_no" in sample and not actual_attempts:
                actual_attempts = list(item)
            elif ("scope" in sample or "parent_sha" in sample) and not actual_subgoals:
                actual_subgoals = list(item)
            elif ("ticket" in sample or "engine_commit" in sample or "repo_key" in sample or "started_at" in sample) and not actual_runs:
                actual_runs = list(item)

    # Build lookup maps for runs and subgoals
    run_map: Dict[str, Mapping[str, Any]] = {}
    for r in actual_runs:
        rid = r.get("run_id")
        if rid:
            run_map[str(rid)] = r

    subgoal_commit_map: Dict[Tuple[str, str], str] = {}
    for sg in actual_subgoals:
        rid = sg.get("run_id")
        sid = sg.get("subgoal_id")
        c_sha = sg.get("commit_sha")
        if rid and sid and c_sha:
            subgoal_commit_map[(str(rid), str(sid))] = str(c_sha)

    # Map each run to its repo_key
    run_repo_map: Dict[str, Optional[str]] = {}
    for rid, r in run_map.items():
        rk = r.get("repo_key") or repo_key
        run_repo_map[rid] = rk

    # Group attempts by (run_id, subgoal_id) to find subsequent fixes
    attempts_by_sg: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for att in actual_attempts:
        rid = str(att.get("run_id") or "")
        sid = str(att.get("subgoal_id") or "")
        attempts_by_sg.setdefault((rid, sid), []).append(att)

    # Sort attempts within each subgoal by attempt_no ascending
    for sg_key in attempts_by_sg:
        attempts_by_sg[sg_key].sort(key=lambda a: int(a.get("attempt_no") or 0))

    # Raw extracted occurrences: map (repo_key, signature) -> list of occurrence dicts
    grouped_occurrences: Dict[Tuple[Optional[str], Tuple[str, str, str]], List[Dict[str, Any]]] = {}

    for att in actual_attempts:
        # Check if attempt failed Tier 0 with compile errors
        tier = att.get("tier")
        outcome = att.get("outcome")
        failed_check = att.get("failed_check")

        # Exclude attempts that passed Tier 0
        if tier == "semantic" or outcome in ("passed", "semantic_failure"):
            continue

        raw_errors_json = att.get("compile_errors_json")
        if not raw_errors_json:
            continue

        try:
            compile_errors = json.loads(raw_errors_json) if isinstance(raw_errors_json, str) else raw_errors_json
        except Exception:
            compile_errors = []

        if not isinstance(compile_errors, list) or not compile_errors:
            continue

        rid = str(att.get("run_id") or "")
        sid = str(att.get("subgoal_id") or "")
        att_no = int(att.get("attempt_no") or 1)
        r_key = att.get("repo_key") or run_repo_map.get(rid) or repo_key

        # Extract known files from attempt
        known_files: List[str] = []
        raw_files = att.get("files_json")
        if raw_files:
            try:
                parsed_files = json.loads(raw_files) if isinstance(raw_files, str) else raw_files
                if isinstance(parsed_files, list):
                    known_files.extend(str(f) for f in parsed_files)
                elif isinstance(parsed_files, dict):
                    known_files.extend(str(f) for f in parsed_files.keys())
            except Exception:
                pass

        # Find first later attempt of the same subgoal that passed Tier 0
        sg_attempts = attempts_by_sg.get((rid, sid), [])
        fix_attempt = None
        for cand in sg_attempts:
            cand_no = int(cand.get("attempt_no") or 0)
            if cand_no > att_no:
                cand_tier = cand.get("tier")
                cand_outcome = cand.get("outcome")
                # Tier 0 passed if semantic tier or passed
                if cand_tier == "semantic" or cand_outcome in ("passed", "semantic_failure"):
                    fix_attempt = cand
                    break

        fix_hint = _cut_diff(fix_attempt.get("diff")) if fix_attempt else None
        fix_attempt_no = int(fix_attempt.get("attempt_no")) if fix_attempt and fix_attempt.get("attempt_no") is not None else None
        fix_commit = subgoal_commit_map.get((rid, sid)) if fix_attempt else None
        base_commit = run_map.get(rid, {}).get("base_commit")

        created_at = att.get("created_at") or ""

        for err in compile_errors:
            if not isinstance(err, dict):
                continue
            e_code = str(err.get("code") or "")
            e_file = str(err.get("file") or "")
            e_msg = str(err.get("message") or "")

            norm_code, norm_file, norm_msg = normalize_error(
                e_code, e_file, e_msg, known_files=known_files
            )

            # If the message is completely empty, drop the entry (don't record an empty-message trap)
            if not norm_msg:
                continue

            sig = (norm_code, norm_file, norm_msg)

            # Calculate evidence_hashes
            ev_hashes: Dict[str, str] = {}
            if norm_file:
                content = _extract_file_content_from_attempt(norm_file, att.get("files_json"), att.get("diff"))
                if content is not None:
                    ev_hashes[norm_file] = hashlib.sha256(content.encode("utf-8")).hexdigest()

            occ = {
                "repo_key": r_key,
                "signature": sig,
                "created_at": created_at,
                "run_id": rid,
                "subgoal_id": sid,
                "attempt_no": att_no,
                "fix_hint": fix_hint,
                "fix_attempt_no": fix_attempt_no,
                "base_commit": base_commit,
                "fix_commit": fix_commit,
                "evidence_hashes": ev_hashes,
            }
            grouped_occurrences.setdefault((r_key, sig), []).append(occ)

    facts: List[CompileTrap] = []

    for (r_key, sig), occurrences in grouped_occurrences.items():
        # Sort occurrences by timestamp / attempt ordering to identify most recent
        occurrences.sort(
            key=lambda o: (
                o.get("created_at") or "",
                o.get("run_id") or "",
                o.get("subgoal_id") or "",
                int(o.get("attempt_no") or 0),
            )
        )

        most_recent = occurrences[-1]
        count = len(occurrences)

        # Keep fix_hint from the most recent occurrence that has one
        most_recent_fix = next(
            (o for o in reversed(occurrences) if o.get("fix_hint") is not None),
            None,
        )

        final_fix_hint = most_recent_fix.get("fix_hint") if most_recent_fix else None
        final_fix_att_no = most_recent_fix.get("fix_attempt_no") if most_recent_fix else None
        final_fix_commit = most_recent_fix.get("fix_commit") if most_recent_fix else None

        # Build evidence hashes from most recent with hashes or latest
        merged_hashes: Dict[str, str] = {}
        for o in occurrences:
            merged_hashes.update(o.get("evidence_hashes") or {})

        provenance = CompileTrapProvenance(
            run_id=most_recent["run_id"],
            subgoal_id=most_recent["subgoal_id"],
            attempt_no=most_recent["attempt_no"],
            fix_attempt_no=final_fix_att_no,
            base_commit=most_recent.get("base_commit"),
            fix_commit=final_fix_commit,
        )

        statement = _build_statement(sig[0], sig[1], sig[2])

        trap = CompileTrap(
            kind="compile_trap",
            repo_key=r_key,
            signature=sig,
            statement=statement,
            fix_hint=final_fix_hint,
            count=count,
            provenance=provenance,
            evidence_hashes=merged_hashes,
        )
        facts.append(trap)

    # Deterministic canonical ordering
    facts.sort(
        key=lambda t: (
            t.repo_key or "",
            t.signature[0],
            t.signature[1],
            t.signature[2],
            t.provenance.run_id,
            t.provenance.subgoal_id,
            t.provenance.attempt_no,
        )
    )

    return facts


def extract_from_ledger(
    db_path: Union[str, Path],
    repo_key: Optional[str] = None,
) -> List[CompileTrap]:
    """Reads ledger database in read-only mode and extracts CompileTrap facts."""
    db_file = Path(db_path)
    if not db_file.exists():
        return []

    # Open SQLite in read-only URI mode
    uri_path = db_file.resolve().as_posix()
    ro_uri = f"file:{uri_path}?mode=ro"

    conn = sqlite3.connect(ro_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # Check existing tables
        tables = {
            r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        runs = [dict(r) for r in cur.execute("SELECT * FROM runs").fetchall()] if "runs" in tables else []
        subgoals = [dict(r) for r in cur.execute("SELECT * FROM subgoals").fetchall()] if "subgoals" in tables else []
        attempts = [dict(r) for r in cur.execute("SELECT * FROM attempts").fetchall()] if "attempts" in tables else []

        # If repo_key not explicitly given, try parent dir name if structured as repos/<repo_key>/ledger.db
        resolved_repo_key = repo_key
        if not resolved_repo_key and db_file.parent.name and db_file.parent.parent.name == "repos":
            resolved_repo_key = db_file.parent.name

        return extract_compile_traps(
            attempts=attempts,
            subgoals=subgoals,
            runs=runs,
            repo_key=resolved_repo_key,
        )
    finally:
        conn.close()
