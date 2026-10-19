"""Phase 4a & 4b-1: Pure, read-only memory extractor for CompileTrap facts.

Extracts deterministic, deduplicated compile error facts from ledger tables.
Contains no LLM imports, no network access, no hashing, and makes no writes to any database.
"""

from collections import Counter
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
    evidence_paths: List[str] = Field(default_factory=list)


def _parse_diff_files(diff_text: Optional[str], default_file: Optional[str] = None) -> Dict[str, List[str]]:
    """Parse body lines (+/- excluding headers) grouped by file using diff --git headers."""
    if not diff_text or not isinstance(diff_text, str):
        return {}

    lines = diff_text.splitlines()
    files: Dict[str, List[str]] = {}
    current_file = None

    has_git_header = any(line.startswith("diff --git ") for line in lines)

    if not has_git_header:
        file_key = default_file or ""
        body_lines = []
        for line in lines:
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+") or line.startswith("-"):
                body_lines.append(line)
        if body_lines:
            files[file_key] = body_lines
        return files

    for line in lines:
        if line.startswith("diff --git "):
            m = re.search(r'diff --git (?:a/)?(\S+) (?:b/)?(\S+)', line)
            if m:
                current_file = m.group(2)
                if current_file.startswith("b/"):
                    current_file = current_file[2:]
            else:
                current_file = line[len("diff --git "):].strip()
            if current_file not in files:
                files[current_file] = []
        elif current_file is not None:
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+") or line.startswith("-"):
                files[current_file].append(line)

    return files


def compute_diff_delta(
    failing_diff: Optional[str],
    passing_diff: Optional[str],
    default_file: Optional[str] = None,
    max_lines: int = 40,
    max_chars: int = 3000,
) -> Optional[str]:
    """Compute the delta between failing attempt diff and passing attempt diff."""
    if not passing_diff:
        return None

    fail_files = _parse_diff_files(failing_diff, default_file=default_file)
    pass_files = _parse_diff_files(passing_diff, default_file=default_file)

    all_files = sorted(set(fail_files.keys()) | set(pass_files.keys()))
    delta_blocks = []

    for f in all_files:
        f_fail_lines = fail_files.get(f, [])
        f_pass_lines = pass_files.get(f, [])

        pass_counts = Counter(f_pass_lines)
        removed = []
        for line in f_fail_lines:
            if pass_counts[line] > 0:
                pass_counts[line] -= 1
            else:
                removed.append(line)

        fail_counts = Counter(f_fail_lines)
        added = []
        for line in f_pass_lines:
            if fail_counts[line] > 0:
                fail_counts[line] -= 1
            else:
                added.append(line)

        if not removed and not added:
            continue

        file_header = f"file: {f}" if f else "file: unknown"
        file_lines = [file_header]
        if removed:
            file_lines.append("failed attempt only:")
            for r in removed:
                file_lines.append(r)
        if added:
            file_lines.append("passing attempt only:")
            for a in added:
                file_lines.append(a)

        delta_blocks.append("\n".join(file_lines))

    if not delta_blocks:
        return None

    full_render = "\n".join(delta_blocks)
    lines = full_render.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        full_render = "\n".join(lines)

    if len(full_render) > max_chars:
        full_render = full_render[:max_chars]

    return full_render or None


def _quote_temp_paths(text: str) -> str:
    """Replaces temp-path-looking substrings with <TMP> idempotently without inserting quotes."""
    temp_pattern = re.compile(
        r'(?:[A-Za-z]:)?[/\\](?:[^/\\\s\'",]+[/\\])*(?:tmp|temp|AppData[/\\]Local[/\\]Temp|pytest-[^/\\\s\'",]+)(?:[/\\]pytest-[^/\\\s\'",]+)*(?:[/\\](?:[a-f0-9]{8,}|[0-9]+|[a-zA-Z0-9]{1,2}))?'
    )
    return temp_pattern.sub("<TMP>", text)


def _convert_windows_paths(text: str) -> str:
    """Convert backslashes to / inside path-looking tokens."""
    pattern = re.compile(r'(?:[A-Za-z]:[\\/][^\s\'",()\[\]]*|[^\s\'",()\[\]]*\\[^\s\'",()\[\]]*)')
    return pattern.sub(lambda m: m.group(0).replace("\\", "/"), text)


def normalize_error(
    code: Optional[str],
    file: Optional[str],
    message: Optional[str],
    known_files: Optional[Sequence[str]] = None,
) -> Tuple[str, str, str]:
    """Normalize code, file, and error message into a canonical signature."""
    norm_code = (code or "").strip()

    raw_file = (file or "").strip().replace("\\", "/")
    # Strip line and column numbers from the file if present (e.g. foo.py:12:34 or Foo.java:[12,8])
    raw_file = re.sub(r':?\[\s*\d+\s*(?:,\s*\d+\s*)?\]$', '', raw_file)
    raw_file = re.sub(r':?\(\s*\d+\s*(?:,\s*\d+\s*)?\)$', '', raw_file)
    raw_file = re.sub(r':\d+(?::\d+)?$', '', raw_file)

    norm_file = raw_file
    matched_known = False
    if known_files and norm_file:
        for kf in sorted(known_files, key=len, reverse=True):
            kf_posix = kf.replace("\\", "/")
            if norm_file == kf_posix or norm_file.endswith("/" + kf_posix) or kf_posix.endswith("/" + norm_file):
                norm_file = kf_posix
                matched_known = True
                break

    if not matched_known and norm_file:
        if norm_file.startswith("/") or re.match(r'^[A-Za-z]:/', norm_file):
            norm_file = norm_file.rsplit("/", 1)[-1]

    norm_file = norm_file.strip()

    raw_msg = (message or "").strip()

    raw_msg = _convert_windows_paths(raw_msg)

    # Strip bracketed line/column forms: [12,8], :[12,8], (12,8), :12:8, :12, (line 1)
    raw_msg = re.sub(r':?\[\s*\d+\s*(?:,\s*\d+\s*)?\]', '', raw_msg)
    raw_msg = re.sub(r':?\(\s*\d+\s*(?:,\s*\d+\s*)?\)', '', raw_msg)
    raw_msg = re.sub(r'\(\s*line\s+\d+(?:,\s*(?:col|column)\s*\d+)?\s*\)', '', raw_msg, flags=re.IGNORECASE)
    raw_msg = re.sub(r'(?<=[A-Za-z0-9_./-])(?::\d+){1,2}\b:?', '', raw_msg)
    raw_msg = re.sub(r'\bline\s+\d+(?:,\s*(?:col|column)\s*\d+)?\b', '', raw_msg, flags=re.IGNORECASE)
    raw_msg = re.sub(r'\b(?:col|column)\s+\d+\b', '', raw_msg, flags=re.IGNORECASE)

    # Strip absolute prefixes of known files
    all_known = list(known_files or [])
    if norm_file and norm_file not in all_known:
        all_known.append(norm_file)

    for kf in sorted(all_known, key=len, reverse=True):
        kf_posix = kf.replace("\\", "/")
        pattern = re.compile(r'(?:[A-Za-z]:/|/)[^\s\'",()\[\]]*?/' + re.escape(kf_posix) + r'\b')
        raw_msg = pattern.sub(kf_posix, raw_msg)

    def _reduce_unmatched(m: re.Match) -> str:
        token = m.group(0)
        base = token.rsplit("/", 1)[-1]
        return base if base else token

    raw_msg = re.sub(r'(?:[A-Za-z]:/|/)[^\s\'",()\[\]]+', _reduce_unmatched, raw_msg)

    raw_msg = _quote_temp_paths(raw_msg)

    norm_msg = re.sub(r'\s+', ' ', raw_msg).strip()

    return norm_code, norm_file, norm_msg


def _build_statement(code: str, file: str, normalized_message: str) -> str:
    """Construct statement from standard template without LLM text."""
    if code and file:
        return f"Attempt failed Tier 0 with {code} in {file}: {normalized_message}"
    if code and not file:
        return f"Attempt failed Tier 0 with {code}: {normalized_message}"
    if file and not code:
        return f"Attempt failed Tier 0 in {file}: {normalized_message}"
    return f"Attempt failed Tier 0: {normalized_message}"


def _derive_error_from_feedback(
    feedback: str, known_files: Optional[Sequence[str]] = None
) -> Dict[str, str]:
    """Derive ONE error from mechanical engine feedback_sent."""
    fb = feedback.strip()
    if not fb:
        return {}

    pattern = re.compile(
        r'^(?:(?P<code>[A-Za-z0-9_]*(?:Error|Exception))\b\s*)?'
        r'(?:in\s+(?P<file>[^\s:]+):\s*|:\s*)?'
        r'(?P<rest>.*)$',
        re.DOTALL,
    )
    m = pattern.match(fb)
    if m:
        code = m.group("code") or ""
        raw_file = m.group("file") or ""
        rest = m.group("rest") or ""

        resolved_file = raw_file
        if raw_file and known_files:
            rf_posix = raw_file.replace("\\", "/")
            for kf in known_files:
                kf_posix = kf.replace("\\", "/")
                if kf_posix == rf_posix or kf_posix.endswith("/" + rf_posix):
                    resolved_file = kf_posix
                    break

        return {"code": code, "file": resolved_file, "message": rest}

    return {"code": "", "file": "", "message": fb}


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
        tier = att.get("tier")
        outcome = att.get("outcome")
        failed_check = att.get("failed_check")

        # Exclude attempts that passed Tier 0
        if tier == "semantic" or outcome in ("passed", "semantic_failure"):
            continue

        if outcome != "mechanical_failure":
            continue

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

        raw_errors_json = att.get("compile_errors_json")
        compile_errors = []
        if raw_errors_json:
            try:
                parsed_errors = json.loads(raw_errors_json) if isinstance(raw_errors_json, str) else raw_errors_json
                if isinstance(parsed_errors, list):
                    compile_errors = parsed_errors
            except Exception:
                compile_errors = []

        # Point 1: When no compile_errors_json and failed_check in {"build", "compile"} and feedback_sent
        if not compile_errors:
            if failed_check in {"build", "compile"}:
                feedback = att.get("feedback_sent")
                if feedback and isinstance(feedback, str) and feedback.strip():
                    derived = _derive_error_from_feedback(feedback.strip(), known_files=known_files)
                    if derived:
                        compile_errors = [derived]
            if not compile_errors:
                continue

        rid = str(att.get("run_id") or "")
        sid = str(att.get("subgoal_id") or "")
        att_no = int(att.get("attempt_no") or 1)
        r_key = att.get("repo_key") or run_repo_map.get(rid) or repo_key

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

            # Point 1: fix_hint is the delta between failing attempt's diff and passing attempt's diff
            fix_hint = compute_diff_delta(
                att.get("diff"),
                fix_attempt.get("diff") if fix_attempt else None,
                default_file=norm_file,
            ) if fix_attempt else None

            # Point 2: evidence_paths is a sorted list of repo-relative file paths concerned by error
            ev_paths = [norm_file] if norm_file else []

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
                "evidence_paths": ev_paths,
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

        # Point 2: merged evidence_paths is the sorted union of paths across occurrences
        merged_paths = sorted({p for o in occurrences for p in o.get("evidence_paths", []) if p})

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
            evidence_paths=merged_paths,
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

    # Point 3: extract_from_ledger uses repo_key if given, otherwise the name of the ledger's parent folder
    # when the path looks like .../repos/<name>/ledger.db, otherwise None.
    resolved_repo_key = repo_key
    if not resolved_repo_key:
        posix_path = db_file.as_posix()
        m = re.search(r'(?:^|/)repos/([^/]+)/ledger\.db$', posix_path, re.IGNORECASE)
        if m:
            resolved_repo_key = m.group(1)

    # Open SQLite in read-only URI mode
    uri_path = db_file.resolve().as_posix()
    ro_uri = f"file:{uri_path}?mode=ro"

    conn = sqlite3.connect(ro_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        tables = {
            r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        runs = [dict(r) for r in cur.execute("SELECT * FROM runs").fetchall()] if "runs" in tables else []
        subgoals = [dict(r) for r in cur.execute("SELECT * FROM subgoals").fetchall()] if "subgoals" in tables else []
        attempts = [dict(r) for r in cur.execute("SELECT * FROM attempts").fetchall()] if "attempts" in tables else []

        return extract_compile_traps(
            attempts=attempts,
            subgoals=subgoals,
            runs=runs,
            repo_key=resolved_repo_key,
        )
    finally:
        conn.close()
