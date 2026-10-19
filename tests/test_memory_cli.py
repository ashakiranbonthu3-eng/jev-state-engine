import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

from jev.knowledge import KnowledgeStore, get_knowledge_db_path
from jev.ledger import SqliteLedger, compute_repo_key, get_jev_home

# Import main and build_parser from scripts.memory
import scripts.memory as memory_cli


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Sets JEV_HOME to a temporary directory."""
    jev_home = tmp_path / "jev_home"
    jev_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JEV_HOME", str(jev_home))
    return jev_home


@pytest.fixture
def repo_and_ledger(tmp_path, isolated_env):
    """Creates a git repo with a commit and a corresponding ledger.db with failing/passing attempts."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

    app_file = repo_dir / "src" / "app.py"
    app_file.parent.mkdir(parents=True)
    app_file.write_text("def run():\n    pass\n", encoding="utf-8")

    pkg_file = repo_dir / "package.json"
    pkg_file.write_text('{"name": "test-pkg", "version": "1.0.0"}', encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)

    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), check=True, capture_output=True, text=True)
    base_commit = rev.stdout.strip()

    repo_key = compute_repo_key(repo_dir)
    ledger_dir = isolated_env / "repos" / repo_key
    ledger_dir.mkdir(parents=True)
    ledger_db = ledger_dir / "ledger.db"

    ledger = SqliteLedger(db_path=ledger_db)
    run_id = "run-1"
    subgoal_id = "sg-1"

    ledger.record_run_start(
        run_id=run_id,
        thread_id="t-1",
        ticket="Fix bug",
        base_commit=base_commit,
    )

    # Attempt 1: failing tier 0 compile
    compile_errors = [
        {"file": "src/app.py", "code": "E0425", "message": "undefined variable `x`"}
    ]
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id=subgoal_id,
        tier="mechanical",
        outcome="mechanical_failure",
        failed_check="compile",
        compile_errors=compile_errors,
        diff="diff --git a/src/app.py b/src/app.py\n+bad_line = x",
    )

    # Attempt 2: passing tier 1 semantic
    ledger.record_attempt(
        run_id=run_id,
        subgoal_id=subgoal_id,
        tier="semantic",
        outcome="passed",
        diff="diff --git a/src/app.py b/src/app.py\n+good_line = 1",
    )

    return repo_dir, repo_key, ledger_db, base_commit


def test_rebuild_creates_fact_and_keeps_injected_count(repo_and_ledger, isolated_env, capsys):
    repo_dir, repo_key, ledger_db, _ = repo_and_ledger
    kdb_path = get_knowledge_db_path(repo_key)
    assert not kdb_path.exists()

    # Run rebuild
    ret = memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    assert ret == 0
    out = capsys.readouterr().out
    assert "inserted: 1" in out
    assert "updated: 0" in out
    assert "skipped: 0" in out
    assert "active: 1" in out

    assert kdb_path.exists()
    store = KnowledgeStore(kdb_path, repo_key=repo_key)
    assert store.count() == 1
    facts = store.list_facts()
    fact_id = facts[0]["id"]
    assert facts[0]["code"] == "E0425"
    assert facts[0]["injected_count"] == 0

    # Simulate injection count update
    with sqlite3.connect(str(kdb_path)) as conn:
        conn.execute("UPDATE facts SET injected_count = 9, helped_count = 3 WHERE id = ?", (fact_id,))
        conn.commit()

    # Run rebuild again
    ret2 = memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    assert ret2 == 0
    out2 = capsys.readouterr().out
    assert "inserted: 0" in out2
    assert "updated: 1" in out2

    facts_after = store.list_facts()
    assert facts_after[0]["injected_count"] == 9
    assert facts_after[0]["helped_count"] == 3


def test_list_prints_fact_and_filters_and_check(repo_and_ledger, isolated_env, capsys):
    repo_dir, repo_key, ledger_db, _ = repo_and_ledger
    memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    capsys.readouterr()

    kdb_path = get_knowledge_db_path(repo_key)
    kdb_bytes_before = kdb_path.read_bytes()

    # List all
    ret = memory_cli.main(["list", "--repo-dir", str(repo_dir)])
    assert ret == 0
    out = capsys.readouterr().out
    assert "status: active" in out
    assert "count: 1" in out
    assert "code: E0425" in out
    assert "file: src/app.py" in out
    assert "undefined variable `x`" in out
    # First line of fix hint
    assert "fix_hint: file: src/app.py" in out

    # Knowledge db file bytes must remain untouched
    assert kdb_path.read_bytes() == kdb_bytes_before

    # Status filter
    ret_stale = memory_cli.main(["list", "--repo-dir", str(repo_dir), "--status", "stale"])
    assert ret_stale == 0
    out_stale = capsys.readouterr().out
    assert "no facts" in out_stale

    # List with --check before editing env file: ok
    ret_chk = memory_cli.main(["list", "--repo-dir", str(repo_dir), "--check"])
    assert ret_chk == 0
    out_chk = capsys.readouterr().out
    assert "check: ok" in out_chk
    assert kdb_path.read_bytes() == kdb_bytes_before

    # Edit package.json and commit
    pkg_file = repo_dir / "package.json"
    pkg_file.write_text('{"name": "test-pkg", "version": "2.0.0"}', encoding="utf-8")
    subprocess.run(["git", "add", "package.json"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "bump"], cwd=str(repo_dir), check=True, capture_output=True)

    # List with --check after editing env file: stale env_changed:package.json
    ret_chk2 = memory_cli.main(["list", "--repo-dir", str(repo_dir), "--check"])
    assert ret_chk2 == 0
    out_chk2 = capsys.readouterr().out
    assert "check: stale env_changed:package.json" in out_chk2

    # The check must NOT update stored status (still active)
    store = KnowledgeStore(kdb_path, repo_key=repo_key)
    assert store.list_facts()[0]["status"] == "active"
    assert kdb_path.read_bytes() == kdb_bytes_before


def test_forget_id_and_forget_all(repo_and_ledger, isolated_env, monkeypatch, capsys):
    repo_dir, repo_key, _, _ = repo_and_ledger
    memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    capsys.readouterr()

    kdb_path = get_knowledge_db_path(repo_key)
    store = KnowledgeStore(kdb_path, repo_key=repo_key)
    facts = store.list_facts()
    fact_id = facts[0]["id"]

    # forget with missing flags -> argparse error
    with pytest.raises(SystemExit):
        memory_cli.main(["forget", "--repo-dir", str(repo_dir)])

    # forget --id
    ret = memory_cli.main(["forget", "--repo-dir", str(repo_dir), "--id", fact_id])
    assert ret == 0
    out = capsys.readouterr().out
    assert "1" in out
    assert store.count() == 0

    # Rebuild again so fact is back
    memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    capsys.readouterr()
    assert store.count() == 1

    # forget --all without yes, user says 'n'
    monkeypatch.setattr("builtins.input", lambda _: "n")
    ret_n = memory_cli.main(["forget", "--repo-dir", str(repo_dir), "--all"])
    assert ret_n == 0
    assert store.count() == 1

    # forget --all without yes, user says 'y'
    monkeypatch.setattr("builtins.input", lambda _: "y")
    ret_y = memory_cli.main(["forget", "--repo-dir", str(repo_dir), "--all"])
    assert ret_y == 0
    assert store.count() == 0

    # Rebuild again
    memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    capsys.readouterr()
    assert store.count() == 1

    # forget --all --yes skips prompt
    ret_yes = memory_cli.main(["forget", "--repo-dir", str(repo_dir), "--all", "--yes"])
    assert ret_yes == 0
    assert store.count() == 0


def test_rebuild_with_no_ledger_prints_no_ledger_and_creates_no_kdb(tmp_path, isolated_env, capsys):
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(empty_repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(empty_repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(empty_repo), check=True, capture_output=True)

    ret = memory_cli.main(["rebuild", "--repo-dir", str(empty_repo)])
    assert ret == 0
    out = capsys.readouterr().out
    assert "no ledger found" in out

    repo_key = compute_repo_key(empty_repo)
    kdb_path = get_knowledge_db_path(repo_key)
    assert not kdb_path.exists()


def test_repo_key_and_db_path_resolution(repo_and_ledger, isolated_env, capsys):
    repo_dir, repo_key, _, _ = repo_and_ledger
    memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    capsys.readouterr()

    kdb_path = get_knowledge_db_path(repo_key)

    # list via --repo-key
    ret1 = memory_cli.main(["list", "--repo-key", repo_key])
    assert ret1 == 0
    out1 = capsys.readouterr().out
    assert "status: active" in out1

    # list via --db-path
    ret2 = memory_cli.main(["list", "--db-path", str(kdb_path)])
    assert ret2 == 0
    out2 = capsys.readouterr().out
    assert "status: active" in out2


def test_ledger_db_bytes_identical_before_and_after_subcommands(repo_and_ledger, isolated_env):
    repo_dir, repo_key, ledger_db, _ = repo_and_ledger
    ledger_bytes_initial = ledger_db.read_bytes()

    # Rebuild
    memory_cli.main(["rebuild", "--repo-dir", str(repo_dir)])
    assert ledger_db.read_bytes() == ledger_bytes_initial

    # List
    memory_cli.main(["list", "--repo-dir", str(repo_dir), "--check"])
    assert ledger_db.read_bytes() == ledger_bytes_initial

    # Forget
    memory_cli.main(["forget", "--repo-dir", str(repo_dir), "--all", "--yes"])
    assert ledger_db.read_bytes() == ledger_bytes_initial


def test_real_jev_home_isolation(repo_and_ledger, isolated_env):
    real_jev = Path.home() / ".jev"
    # Ensure our operations never touched ~/.jev
    assert not (real_jev / "repos" / repo_and_ledger[1] / "knowledge.db").exists()
