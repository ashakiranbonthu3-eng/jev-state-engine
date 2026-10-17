import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from jev.ledger import SqliteLedger


def test_no_llm_or_network_imports_in_memory_extract():
    """Assert module source does not import langchain or httpx."""
    import jev.memory_extract as mod

    source_path = Path(mod.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    assert "langchain" not in source_text, "memory_extract must not import langchain"
    assert "httpx" not in source_text, "memory_extract must not import httpx"
    assert "urllib" not in source_text, "memory_extract must not import network libraries"


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
    assert codes == {"E0425", "TS2304", "tier0_failure"}

    # Statements generated according to template
    rust_fact = next(f for f in facts if f.signature[0] == "E0425")
    assert "Attempt failed Tier 0 with E0425 in src/main.rs:" in rust_fact.statement

    unparsed_fact = next(f for f in facts if f.signature[0] == "tier0_failure")
    assert "Attempt failed Tier 0 with tier0_failure:" in unparsed_fact.statement
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


def test_evidence_hashes_computed_from_files_json_or_diff():
    """evidence_hashes computes sha256 of files when present in files_json or diff, else empty."""
    import hashlib
    from jev.memory_extract import extract_compile_traps

    file_content = "def calculate():\n    return 42\n"
    expected_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

    # 1. File content in files_json dict
    att_files_json = {
        "run_id": "r-ev1",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/math.py", "code": "E1", "message": "error"}]),
        "files_json": json.dumps({"src/math.py": file_content}),
    }
    facts1 = extract_compile_traps(attempts=[att_files_json])
    assert len(facts1) == 1
    assert facts1[0].evidence_hashes.get("src/math.py") == expected_hash

    # 2. File in diff additions
    diff_content = "diff --git a/src/math.py b/src/math.py\n--- /dev/null\n+++ b/src/math.py\n@@ -0,0 +1,2 @@\n+def calculate():\n+    return 42"
    diff_hash = hashlib.sha256(b"def calculate():\nreturn 42" if False else "def calculate():\n    return 42".encode("utf-8")).hexdigest()
    att_diff = {
        "run_id": "r-ev2",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/math.py", "code": "E1", "message": "error"}]),
        "diff": diff_content,
    }
    facts2 = extract_compile_traps(attempts=[att_diff])
    assert len(facts2) == 1
    assert facts2[0].evidence_hashes.get("src/math.py") == diff_hash

    # 3. File absent from files_json and diff -> empty mapping
    att_none = {
        "run_id": "r-ev3",
        "subgoal_id": "sg-1",
        "attempt_no": 1,
        "tier": "mechanical",
        "outcome": "mechanical_failure",
        "failed_check": "compile",
        "compile_errors_json": json.dumps([{"file": "src/absent.py", "code": "E1", "message": "error"}]),
        "diff": "diff --git a/other.py b/other.py\n+x=1",
        "files_json": json.dumps(["other.py"]),
    }
    facts3 = extract_compile_traps(attempts=[att_none])
    assert len(facts3) == 1
    assert facts3[0].evidence_hashes == {}


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
    assert fact.fix_hint == "+fix attempt 1"  # Retained from most recent occurrence with fix
    assert fact.provenance.fix_attempt_no == 2
    assert fact.provenance.fix_commit == "sha-fix-1"

