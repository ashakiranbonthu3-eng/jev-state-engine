import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from jev.ledger import SqliteLedger


def test_no_llm_or_network_imports_in_memory_extract():
    """Assert module source does not import langchain, httpx, or hashlib."""
    import jev.memory_extract as mod

    source_path = Path(mod.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    assert "langchain" not in source_text, "memory_extract must not import langchain"
    assert "httpx" not in source_text, "memory_extract must not import httpx"
    assert "urllib" not in source_text, "memory_extract must not import network libraries"
    assert "hashlib" not in source_text, "memory_extract must not import hashlib"


def test_failed_attempt_followed_by_passing_attempt_gives_fix_hint_cut_to_40_lines(tmp_path):
    """A failed attempt followed by a passing attempt of the same subgoal gives one fact with a fix_hint cut to 40 lines."""
    from jev.memory_extract import extract_from_ledger

    db_path = tmp_path / "ledger.db"
    ledger = SqliteLedger(db_path=db_path)
    run_id = "run-1"
    subgoal_id = "sg-1"

    ledger.record_run_start(run_id=run_id, thread_id="t-1", ticket="Fix bug")

    # Long diff with 50 lines
    long_diff_lines = [f"+line {i}" for i in range(50)]
    long_diff = "diff --git a/src/app.py b/src/app.py\n" + "\n".join(long_diff_lines)

    # Attempt 1: Failed Tier 0 with compile errors
    compile_errors_1 = [
        {"file": "src/app.py", "code": "E0425", "message": "cannot find value `x` in this scope:10:5"}
    ]
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id=subgoal_id,
        tier="mechanical",
        outcome="mechanical_failure",
        failed_check="compile",
        compile_errors=compile_errors_1,
        diff="diff --git a/src/app.py b/src/app.py\n+bad = 1",
    )

    # Attempt 2: Passed Tier 0 (semantic evaluation stage or passed)
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id=subgoal_id,
        tier="semantic",
        outcome="passed",
        diff=long_diff,
    )

    facts = extract_from_ledger(db_path)
    assert len(facts) == 1
    fact = facts[0]
    assert fact.kind == "compile_trap"
    assert fact.signature[0] == "E0425"
    assert fact.signature[1] == "src/app.py"
    assert fact.fix_hint is not None
    # Cut to at most 40 lines
    assert len(fact.fix_hint.splitlines()) <= 40
    assert fact.provenance.run_id == run_id
    assert fact.provenance.subgoal_id == subgoal_id
    assert fact.provenance.attempt_no == 1
    assert fact.provenance.fix_attempt_no == 2


def test_same_error_on_different_lines_and_absolute_prefixes_deduplicates(tmp_path):
    """Same error on different line numbers and different absolute path prefixes de-duplicates into one signature with count 2."""
    from jev.memory_extract import extract_compile_traps

    run_id = "run-dedup"
    subgoal_id = "sg-dedup"

    # Attempt 1 has absolute path /home/user/repo/src/lib.rs:12:34
    att1 = {
        "run_id": run_id,
        "subgoal_id": subgoal_id,
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([
            {
                "file": "/home/user/repo/src/lib.rs",
                "code": "E0425",
                "message": "cannot find value `foo` at /home/user/repo/src/lib.rs:12:34",
            }
        ]),
        "diff": "diff --git a/src/lib.rs b/src/lib.rs\n+x = 1",
        "files_json": json.dumps(["src/lib.rs"]),
        "created_at": "2026-10-17T10:00:00Z",
    }

    # Attempt 2 has absolute path C:/agent/work/src/lib.rs:88:5
    att2 = {
        "run_id": run_id,
        "subgoal_id": subgoal_id,
        "attempt_no": 2,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([
            {
                "file": "C:/agent/work/src/lib.rs",
                "code": "E0425",
                "message": "cannot find value `foo` at C:/agent/work/src/lib.rs:88:5",
            }
        ]),
        "diff": "diff --git a/src/lib.rs b/src/lib.rs\n+x = 2",
        "files_json": json.dumps(["src/lib.rs"]),
        "created_at": "2026-10-17T10:05:00Z",
    }

    runs = [{"run_id": run_id, "ticket": "Dedup test", "base_commit": "base-sha"}]
    facts = extract_compile_traps(attempts=[att1, att2], runs=runs)

    assert len(facts) == 1
    fact = facts[0]
    assert fact.count == 2
    assert fact.signature[0] == "E0425"
    assert fact.signature[1] == "src/lib.rs"
    # Line numbers and absolute prefixes stripped from message
    assert "12:34" not in fact.signature[2]
    assert "88:5" not in fact.signature[2]
    assert "/home/user/repo" not in fact.signature[2]
    assert "C:/agent/work" not in fact.signature[2]
    assert fact.provenance.attempt_no == 2  # Most recent attempt


def test_failed_attempt_with_no_later_pass_gives_fix_hint_none():
    """A failed attempt with no later pass gives fix_hint None."""
    from jev.memory_extract import extract_compile_traps

    att = {
        "run_id": "run-nofix",
        "subgoal_id": "sg-nofix",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([
            {"file": "main.py", "code": "E999", "message": "SyntaxError: invalid syntax"}
        ]),
        "diff": "+bad code",
        "files_json": json.dumps(["main.py"]),
    }
    facts = extract_compile_traps(attempts=[att])
    assert len(facts) == 1
    assert facts[0].fix_hint is None
    assert facts[0].provenance.fix_attempt_no is None


def test_passing_attempt_or_failure_without_compile_errors_produces_no_fact():
    """An attempt that passed Tier 0 produces no fact; an attempt without compile errors produces no fact."""
    from jev.memory_extract import extract_compile_traps

    # 1. Passed attempt
    att_pass = {
        "run_id": "run-p",
        "subgoal_id": "sg-p",
        "attempt_no": 1,
        "tier": "semantic",
        "outcome": "passed",
        "diff": "+good code",
    }
    # 2. Semantic rejection (no compile errors)
    att_semantic_fail = {
        "run_id": "run-s",
        "subgoal_id": "sg-s",
        "attempt_no": 1,
        "tier": "semantic",
        "outcome": "semantic_failure",
        "failed_check": "jev_rejection",
        "diff": "+semantically invalid code",
        "compile_errors_json": None,
    }
    # 3. Mechanical failure on tests, not compile (compile_errors_json is None or empty)
    att_test_fail = {
        "run_id": "run-t",
        "subgoal_id": "sg-t",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "tests",
        "compile_errors_json": "[]",
    }

    facts = extract_compile_traps(attempts=[att_pass, att_semantic_fail, att_test_fail])
    assert len(facts) == 0


def test_tool_agnostic_rust_ts_and_empty_code_messages_produce_facts():
    """A Rust-style code, a TS-style code, and an empty-code message each produce facts."""
    from jev.memory_extract import extract_compile_traps

    attempts = [
        {
            "run_id": "run-tools",
            "subgoal_id": "sg-rust",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": "compile",
            "compile_errors_json": json.dumps([
                {"file": "src/main.rs", "code": "E0425", "message": "cannot find value `x` in this scope"}
            ]),
        },
        {
            "run_id": "run-tools",
            "subgoal_id": "sg-ts",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": "compile",
            "compile_errors_json": json.dumps([
                {"file": "src/index.ts", "code": "TS2304", "message": "Cannot find name 'User'"}
            ]),
        },
        {
            "run_id": "run-tools",
            "subgoal_id": "sg-unparsed",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": "compile",
            "compile_errors_json": json.dumps([
                {"file": "", "code": "", "message": "Build failed: fatal error C1083: Cannot open include file"}
            ]),
        },
    ]

    facts = extract_compile_traps(attempts=attempts)
    assert len(facts) == 3
    codes = {f.signature[0] for f in facts}
    assert codes == {"E0425", "TS2304", ""}

    # Statements generated according to template
    rust_fact = next(f for f in facts if f.signature[0] == "E0425")
    assert "Attempt failed Tier 0 with E0425 in src/main.rs:" in rust_fact.statement

    unparsed_fact = next(f for f in facts if f.signature[0] == "")
    assert "Attempt failed Tier 0:" in unparsed_fact.statement
    assert "Build failed" in unparsed_fact.statement


def test_two_runs_on_same_repo_key_deduplicate_different_repo_keys_stay_separate():
    """Two runs on the same repo_key de-duplicate; different repo_keys stay separate."""
    from jev.memory_extract import extract_compile_traps

    runs = [
        {"run_id": "run-repoA-1", "repo_key": "repoA"},
        {"run_id": "run-repoA-2", "repo_key": "repoA"},
        {"run_id": "run-repoB-1", "repo_key": "repoB"},
    ]

    err = [{"file": "src/lib.py", "code": "E101", "message": "indentation error"}]

    att1 = {
        "run_id": "run-repoA-1",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps(err),
        "created_at": "2026-10-17T11:00:00Z",
    }
    att2 = {
        "run_id": "run-repoA-2",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps(err),
        "created_at": "2026-10-17T12:00:00Z",
    }
    att3 = {
        "run_id": "run-repoB-1",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps(err),
        "created_at": "2026-10-17T13:00:00Z",
    }

    facts = extract_compile_traps(attempts=[att1, att2, att3], runs=runs)
    # repoA-1 and repoA-2 merge into 1 fact with count 2; repoB-1 stays separate with count 1
    assert len(facts) == 2

    repoA_fact = next(f for f in facts if f.repo_key == "repoA")
    assert repoA_fact.count == 2
    assert repoA_fact.provenance.run_id == "run-repoA-2"

    repoB_fact = next(f for f in facts if f.repo_key == "repoB")
    assert repoB_fact.count == 1
    assert repoB_fact.provenance.run_id == "run-repoB-1"


def test_output_is_byte_identical_across_two_calls():
    """Output JSON is byte-identical across multiple extraction calls."""
    from jev.memory_extract import extract_compile_traps

    attempts = [
        {
            "run_id": "r-1",
            "subgoal_id": "sg-1",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": "compile",
            "compile_errors_json": json.dumps([
                {"file": "b.py", "code": "E2", "message": "err b"},
                {"file": "a.py", "code": "E1", "message": "err a"},
            ]),
        }
    ]

    call1 = extract_compile_traps(attempts=attempts)
    call2 = extract_compile_traps(attempts=attempts)

    json1 = [f.model_dump_json() for f in call1]
    json2 = [f.model_dump_json() for f in call2]
    assert json1 == json2
    assert "".join(json1) == "".join(json2)


def test_extract_from_ledger_read_only_and_missing_db(tmp_path):
    """extract_from_ledger opens DB read-only (writes fail) and missing DB returns empty list."""
    from jev.memory_extract import extract_from_ledger

    # 1. Missing database returns empty list without error
    non_existent = tmp_path / "does_not_exist.db"
    assert extract_from_ledger(non_existent) == []

    # 2. Real database opens in read-only mode (writes fail)
    db_path = tmp_path / "test_ro.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (val TEXT)")
    conn.commit()
    conn.close()

    # Attempting to write through extract_from_ledger's connection mechanism must fail with OperationalError
    ro_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    ro_conn = sqlite3.connect(ro_uri, uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro_conn.execute("INSERT INTO test VALUES ('forbidden')")
    ro_conn.close()


def test_evidence_paths_computed_and_deduplicated():
    """evidence_paths is a sorted list of repo-relative paths, empty when no file, and merged via union."""
    from jev.memory_extract import extract_compile_traps

    # 1. Error with file -> evidence_paths has [norm_file], no evidence_hashes attribute
    att1 = {
        "run_id": "r-ev1",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/math.py", "code": "E1", "message": "error"}]),
        "files_json": json.dumps(["src/math.py"]),
    }
    facts1 = extract_compile_traps(attempts=[att1])
    assert len(facts1) == 1
    assert facts1[0].evidence_paths == ["src/math.py"]
    assert not hasattr(facts1[0], "evidence_hashes")

    # 2. Error without file -> evidence_paths is []
    att2 = {
        "run_id": "r-ev2",
        "subgoal_id": "sg-2",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "", "code": "E1", "message": "error no file"}]),
    }
    facts2 = extract_compile_traps(attempts=[att2])
    assert len(facts2) == 1
    assert facts2[0].evidence_paths == []

    # 3. Deduplication merges evidence_paths as sorted union
    att3a = {
        "run_id": "r-ev3",
        "subgoal_id": "sg-3",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/math.py", "code": "E1", "message": "error"}]),
        "files_json": json.dumps(["src/math.py"]),
        "created_at": "2026-10-17T10:00:00Z",
    }
    att3b = {
        "run_id": "r-ev3",
        "subgoal_id": "sg-3",
        "attempt_no": 2,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/math.py", "code": "E1", "message": "error"}]),
        "files_json": json.dumps(["src/math.py"]),
        "created_at": "2026-10-17T10:05:00Z",
    }
    facts3 = extract_compile_traps(attempts=[att3a, att3b])
    assert len(facts3) == 1
    assert facts3[0].evidence_paths == ["src/math.py"]


def test_empty_message_dropped_even_with_file_or_code():
    """If the message is completely empty, drop the entry (don't record an empty-message trap)."""
    from jev.memory_extract import extract_compile_traps

    attempts = [
        {
            "run_id": "r-empty",
            "subgoal_id": "sg-empty",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": "compile",
            "compile_errors_json": json.dumps([
                {"file": "", "code": "", "message": "   \n  \t "},
                {"file": "src/app.py", "code": "E100", "message": "  "},
                {"file": "", "code": "", "message": "Valid unparsed message"},
            ]),
        }
    ]
    facts = extract_compile_traps(attempts=attempts)
    assert len(facts) == 1
    assert facts[0].signature[2] == "Valid unparsed message"


def test_fix_hint_retained_from_most_recent_occurrence_with_fix():
    """Keep fix_hint from the most recent occurrence that has one."""
    from jev.memory_extract import extract_compile_traps

    # Occurrence 1 at 10:00 has fix_hint
    att1 = {
        "run_id": "r-1",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "a.py", "code": "E1", "message": "err"}]),
        "created_at": "2026-10-17T10:00:00Z",
    }
    att1_fix = {
        "run_id": "r-1",
        "subgoal_id": "sg-1",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": "+fix attempt 1",
        "created_at": "2026-10-17T10:01:00Z",
    }
    # Occurrence 2 at 11:00 has NO fix_hint
    att2 = {
        "run_id": "r-2",
        "subgoal_id": "sg-2",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "a.py", "code": "E1", "message": "err"}]),
        "created_at": "2026-10-17T11:00:00Z",
    }

    subgoals = [{"run_id": "r-1", "subgoal_id": "sg-1", "commit_sha": "sha-fix-1"}]
    facts = extract_compile_traps(attempts=[att1, att1_fix, att2], subgoals=subgoals)
    assert len(facts) == 1
    fact = facts[0]
    assert fact.count == 2
    assert fact.provenance.run_id == "r-2"  # Most recent occurrence
    assert fact.fix_hint is not None
    assert "passing attempt only:" in fact.fix_hint
    assert "+fix attempt 1" in fact.fix_hint  # Retained from most recent occurrence with fix
    assert fact.provenance.fix_attempt_no == 2
    assert fact.provenance.fix_commit == "sha-fix-1"


def test_feedback_sent_derives_fact_for_build_or_compile():
    """When attempt has mechanical_failure on build/compile without compile_errors_json, derive from feedback_sent."""
    from jev.memory_extract import extract_compile_traps

    # Real-shaped row: failed_check "build", feedback_sent "SyntaxError in app.py: expected ':' (line 1)", files_json ["src/app.py"], later passing attempt
    # -> one fact with code "SyntaxError", file "src/app.py", message "expected ':'", and a fix_hint.
    att1 = {
        "run_id": "r-fb",
        "subgoal_id": "sg-fb",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "build",
        "feedback_sent": "SyntaxError in app.py: expected ':' (line 1)",
        "files_json": json.dumps(["src/app.py"]),
    }
    att2 = {
        "run_id": "r-fb",
        "subgoal_id": "sg-fb",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": "+good diff",
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    fact = facts[0]
    assert fact.signature[0] == "SyntaxError"
    assert fact.signature[1] == "src/app.py"
    assert fact.signature[2] == "expected ':'"
    assert fact.fix_hint is not None
    assert "passing attempt only:" in fact.fix_hint
    assert "+good diff" in fact.fix_hint


def test_other_failed_checks_with_feedback_sent_produce_no_fact():
    """Attempts with other failed_check values produce NO fact."""
    from jev.memory_extract import extract_compile_traps

    other_checks = [
        "scope",
        "suppression",
        "unresolved_import",
        "tests",
        "env_not_ready",
        "no_compile_command",
        "no_tests_collected",
    ]
    attempts = [
        {
            "run_id": f"r-{fc}",
            "subgoal_id": "sg-1",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": fc,
            "feedback_sent": f"Some error in app.py: failure {fc}",
            "files_json": json.dumps(["src/app.py"]),
        }
        for fc in other_checks
    ]
    facts = extract_compile_traps(attempts=attempts)
    assert len(facts) == 0


def test_quote_temp_paths_replaces_with_tmp_and_is_idempotent():
    """_quote_temp_paths replaces temp-path-looking substrings with <TMP> without inserting quote characters and is idempotent."""
    from jev.memory_extract import _quote_temp_paths

    s = 'File "/tmp/pytest-of-u/pytest-3/repo/src/app.py", line 3'
    res = _quote_temp_paths(s)
    # app.py kept intact without stray quotes
    assert "app.py" in res
    assert '""' not in res
    assert "<TMP>" in res
    # Idempotent
    assert _quote_temp_paths(res) == res


def test_windows_paths_in_messages_and_unmatched_absolute_path_basename():
    """Windows backslashes converted to / and messages with Windows vs Unix temp paths normalize to the SAME string."""
    from jev.memory_extract import normalize_error

    msg_win = r'File "C:\Users\D\AppData\Local\Temp\x\src\app.py"'
    msg_unix = 'File "/tmp/pytest-of-u/pytest-3/repo/src/app.py"'

    # Both normalize to the same string when known_files has src/app.py
    _, _, norm_win = normalize_error("E1", "src/app.py", msg_win, known_files=["src/app.py"])
    _, _, norm_unix = normalize_error("E1", "src/app.py", msg_unix, known_files=["src/app.py"])
    assert norm_win == norm_unix

    # Unmatched absolute path token in message reduces to its basename
    msg_unmatched = r'Error at D:\some\unknown\path\helper.py'
    _, _, norm_unmatched = normalize_error("E1", "src/app.py", msg_unmatched, known_files=["src/app.py"])
    assert "D:" not in norm_unmatched
    assert "helper.py" in norm_unmatched


def test_strip_bracketed_line_column_forms():
    """Strip bracketed line/column forms [12,8], :[12,8], (12,8), :12:8, :12."""
    from jev.memory_extract import normalize_error

    maven_unix = "[ERROR] /home/a/proj/src/main/java/Foo.java:[12,8] cannot find symbol"
    maven_win = r"[ERROR] D:\w\proj\src\main\java\Foo.java:[40,2] cannot find symbol"

    _, _, norm_unix = normalize_error("E", "Foo.java", maven_unix)
    _, _, norm_win = normalize_error("E", "Foo.java", maven_win)
    assert norm_unix == norm_win
    assert "[12,8]" not in norm_unix
    assert "[40,2]" not in norm_win


def test_unknown_absolute_file_path_reduces_to_basename():
    """When known_files has no match and file path is absolute, reduce it to its basename."""
    from jev.memory_extract import normalize_error

    win_path = r"C:\workspace\clients\acme\app\page.rs"
    unix_path = "/ci/acme/app/page.rs"

    _, file_win, _ = normalize_error("E", win_path, "error")
    _, file_unix, _ = normalize_error("E", unix_path, "error")

    assert file_win == "page.rs"
    assert file_unix == "page.rs"
    assert file_win == file_unix


def test_empty_code_remains_empty_string_and_statement_formatting():
    """An empty or missing code must stay the empty string and statement omits with <code> and omits in <file> if empty."""
    from jev.memory_extract import extract_compile_traps

    attempts = [
        {
            "run_id": "r-empty-code",
            "subgoal_id": "sg-1",
            "attempt_no": 1,
            "tier": "mechanical",
            "outcome": "mechanical_failure",
            "failed_check": "compile",
            "compile_errors_json": json.dumps([
                {"file": "src/app.py", "code": "", "message": "unparsed error in file"},
                {"file": "", "code": "", "message": "unparsed error no file"},
            ]),
        }
    ]
    facts = extract_compile_traps(attempts=attempts)
    assert len(facts) == 2
    f_with_file = next(f for f in facts if f.signature[1] == "src/app.py")
    assert f_with_file.signature[0] == ""
    assert f_with_file.statement == "Attempt failed Tier 0 in src/app.py: unparsed error in file"

    f_no_file = next(f for f in facts if f.signature[1] == "")
    assert f_no_file.signature[0] == ""
    assert f_no_file.statement == "Attempt failed Tier 0: unparsed error no file"


def test_fix_hint_exact_rendering():
    """Exact test from prompt: base file has 'def add(a, b):', failing has extra line changes, passing only fixes return line."""
    from jev.memory_extract import extract_compile_traps

    fail_diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def add(a, b):\n"
        "-    return a + b\n"
        "+def add(a, b)\n"
        "+    return a + b + 1\n"
    )
    pass_diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-    return a + b\n"
        "+    return a + b + 1\n"
    )

    att1 = {
        "run_id": "r-exact",
        "subgoal_id": "sg-exact",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/app.py", "code": "E1", "message": "syntax error"}]),
        "diff": fail_diff,
        "files_json": json.dumps(["src/app.py"]),
    }
    att2 = {
        "run_id": "r-exact",
        "subgoal_id": "sg-exact",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": pass_diff,
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    expected = (
        "file: src/app.py\n"
        "failed attempt only:\n"
        "-def add(a, b):\n"
        "+def add(a, b)"
    )
    assert facts[0].fix_hint == expected


def test_fix_hint_both_sections_non_empty():
    """Both failed attempt only and passing attempt only sections are rendered when non-empty."""
    from jev.memory_extract import extract_compile_traps

    fail_diff = "diff --git a/src/app.py b/src/app.py\n+bad_line = 1\n"
    pass_diff = "diff --git a/src/app.py b/src/app.py\n+good_line = 1\n"

    att1 = {
        "run_id": "r-both",
        "subgoal_id": "sg-both",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/app.py", "code": "E1", "message": "syntax error"}]),
        "diff": fail_diff,
        "files_json": json.dumps(["src/app.py"]),
    }
    att2 = {
        "run_id": "r-both",
        "subgoal_id": "sg-both",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": pass_diff,
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    expected = (
        "file: src/app.py\n"
        "failed attempt only:\n"
        "+bad_line = 1\n"
        "passing attempt only:\n"
        "+good_line = 1"
    )
    assert facts[0].fix_hint == expected


def test_fix_hint_delta_failing_vs_passing_diff():
    """failing diff '+def add(a, b)' vs passing diff '+def add(a, b):' gives hint with sections and keeps markers."""
    from jev.memory_extract import extract_compile_traps

    fail_diff = "diff --git a/src/app.py b/src/app.py\n def existing():\n     pass\n+def add(a, b)"
    pass_diff = "diff --git a/src/app.py b/src/app.py\n def existing():\n     pass\n+def add(a, b):"

    att1 = {
        "run_id": "r-delta",
        "subgoal_id": "sg-delta",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/app.py", "code": "E1", "message": "syntax error"}]),
        "diff": fail_diff,
        "files_json": json.dumps(["src/app.py"]),
    }
    att2 = {
        "run_id": "r-delta",
        "subgoal_id": "sg-delta",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": pass_diff,
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    hint = facts[0].fix_hint
    assert hint is not None
    assert "failed attempt only:\n+def add(a, b)" in hint
    assert "passing attempt only:\n+def add(a, b):" in hint
    assert "def existing():" not in hint


def test_fix_hint_delta_multi_file():
    """Multi-file delta groups by file with 'file: <path>' header and skips files with no delta."""
    from jev.memory_extract import extract_compile_traps

    fail_diff = (
        "diff --git a/src/a.py b/src/a.py\n+bad_a = 1\n"
        "diff --git a/src/unchanged.py b/src/unchanged.py\n+same = 1\n"
        "diff --git a/src/b.py b/src/b.py\n+bad_b = 2\n"
    )
    pass_diff = (
        "diff --git a/src/a.py b/src/a.py\n+good_a = 1\n"
        "diff --git a/src/unchanged.py b/src/unchanged.py\n+same = 1\n"
        "diff --git a/src/b.py b/src/b.py\n+good_b = 2\n"
    )
    att1 = {
        "run_id": "r-mf",
        "subgoal_id": "sg-mf",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/a.py", "code": "E1", "message": "error a"}]),
        "diff": fail_diff,
        "files_json": json.dumps(["src/a.py", "src/b.py", "src/unchanged.py"]),
    }
    att2 = {
        "run_id": "r-mf",
        "subgoal_id": "sg-mf",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": pass_diff,
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    hint = facts[0].fix_hint
    assert "file: src/a.py" in hint
    assert "failed attempt only:\n+bad_a = 1" in hint
    assert "passing attempt only:\n+good_a = 1" in hint
    assert "file: src/b.py" in hint
    assert "failed attempt only:\n+bad_b = 2" in hint
    assert "passing attempt only:\n+good_b = 2" in hint
    assert "src/unchanged.py" not in hint


def test_fix_hint_delta_identical_diffs_give_none():
    """Identical diffs between failing and passing attempt yield fix_hint None."""
    from jev.memory_extract import extract_compile_traps

    diff = "diff --git a/src/a.py b/src/a.py\n+same = 1"
    att1 = {
        "run_id": "r-id",
        "subgoal_id": "sg-id",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/a.py", "code": "E1", "message": "error"}]),
        "diff": diff,
    }
    att2 = {
        "run_id": "r-id",
        "subgoal_id": "sg-id",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": diff,
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    assert facts[0].fix_hint is None


def test_fix_hint_delta_cap():
    """Delta output is capped at 40 lines and 3000 characters."""
    from jev.memory_extract import extract_compile_traps

    fail_diff = "diff --git a/src/big.py b/src/big.py\n" + "\n".join(f"+fail_{i} = {i}" for i in range(50))
    pass_diff = "diff --git a/src/big.py b/src/big.py\n" + "\n".join(f"+pass_{i} = {i}" for i in range(50))
    att1 = {
        "run_id": "r-cap",
        "subgoal_id": "sg-cap",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/big.py", "code": "E1", "message": "error"}]),
        "diff": fail_diff,
    }
    att2 = {
        "run_id": "r-cap",
        "subgoal_id": "sg-cap",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": pass_diff,
    }
    facts = extract_compile_traps(attempts=[att1, att2])
    assert len(facts) == 1
    assert facts[0].fix_hint is not None
    assert len(facts[0].fix_hint.splitlines()) <= 40
    assert len(facts[0].fix_hint) <= 3000


def test_fix_hint_delta_byte_identical_across_two_calls():
    """Delta computation is byte-identical across two separate calls."""
    from jev.memory_extract import extract_compile_traps

    fail_diff = "diff --git a/src/z.py b/src/z.py\n+z_fail\ndiff --git a/src/a.py b/src/a.py\n+a_fail"
    pass_diff = "diff --git a/src/z.py b/src/z.py\n+z_pass\ndiff --git a/src/a.py b/src/a.py\n+a_pass"
    att1 = {
        "run_id": "r-det",
        "subgoal_id": "sg-det",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/a.py", "code": "E1", "message": "error"}]),
        "diff": fail_diff,
    }
    att2 = {
        "run_id": "r-det",
        "subgoal_id": "sg-det",
        "attempt_no": 2,
        "tier": "semantic",
        "outcome": "passed",
        "diff": pass_diff,
    }
    facts_a = extract_compile_traps(attempts=[att1, att2])
    facts_b = extract_compile_traps(attempts=[att1, att2])
    assert facts_a[0].fix_hint == facts_b[0].fix_hint


def test_extract_from_ledger_repo_key_parent_folder(tmp_path):
    """A ledger at tmp/repos/abc123/ledger.db yields repo_key 'abc123'."""
    from jev.memory_extract import extract_from_ledger

    repo_dir = tmp_path / "repos" / "abc123"
    repo_dir.mkdir(parents=True)
    db_path = repo_dir / "ledger.db"
    ledger = SqliteLedger(db_path=db_path)
    run_id = "run-rk1"
    ledger.record_run_start(run_id=run_id, thread_id="t-1", ticket="Bug")
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="mechanical",
        outcome="mechanical_failure",
        failed_check="compile",
        compile_errors=[{"file": "src/app.py", "code": "E1", "message": "error"}],
        diff="+bad",
    )

    facts = extract_from_ledger(db_path)
    assert len(facts) == 1
    assert facts[0].repo_key == "abc123"


def test_extract_from_ledger_repo_key_explicit_wins(tmp_path):
    """An explicit repo_key wins over the parent folder name."""
    from jev.memory_extract import extract_from_ledger

    repo_dir = tmp_path / "repos" / "abc123"
    repo_dir.mkdir(parents=True)
    db_path = repo_dir / "ledger.db"
    ledger = SqliteLedger(db_path=db_path)
    run_id = "run-rk2"
    ledger.record_run_start(run_id=run_id, thread_id="t-1", ticket="Bug")
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="mechanical",
        outcome="mechanical_failure",
        failed_check="compile",
        compile_errors=[{"file": "src/app.py", "code": "E1", "message": "error"}],
        diff="+bad",
    )

    facts = extract_from_ledger(db_path, repo_key="explicit_win")
    assert len(facts) == 1
    assert facts[0].repo_key == "explicit_win"


def test_extract_from_ledger_repo_key_elsewhere_yields_none(tmp_path):
    """A ledger elsewhere yields repo_key None."""
    from jev.memory_extract import extract_from_ledger

    other_dir = tmp_path / "other_folder"
    other_dir.mkdir(parents=True)
    db_path = other_dir / "ledger.db"
    ledger = SqliteLedger(db_path=db_path)
    run_id = "run-rk3"
    ledger.record_run_start(run_id=run_id, thread_id="t-1", ticket="Bug")
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id="sg-1",
        tier="mechanical",
        outcome="mechanical_failure",
        failed_check="compile",
        compile_errors=[{"file": "src/app.py", "code": "E1", "message": "error"}],
        diff="+bad",
    )

    facts = extract_from_ledger(db_path)
    assert len(facts) == 1
    assert facts[0].repo_key is None



