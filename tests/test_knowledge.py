import json
import logging
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from jev.knowledge import (
    KnowledgeStore,
    MemoryStore,
    NullMemory,
    StatusCounts,
    UpsertResult,
    capture_evidence,
    compute_fact_id,
    get_knowledge_db_path,
)
from jev.ledger import get_jev_home
from jev.memory_extract import CompileTrap, CompileTrapProvenance


@pytest.fixture
def fake_clock():
    current_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def clock():
        return current_time

    def advance(delta: timedelta):
        nonlocal current_time
        current_time += delta

    clock.advance = advance
    return clock


@pytest.fixture
def git_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

    # Initial file
    app_file = repo_dir / "src" / "app.py"
    app_file.parent.mkdir(parents=True)
    app_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    pkg_file = repo_dir / "package.json"
    pkg_file.write_text('{"name": "test-pkg", "version": "1.0.0"}', encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo_dir), check=True, capture_output=True)

    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), check=True, capture_output=True, text=True)
    initial_commit = rev.stdout.strip()

    return repo_dir, initial_commit


def make_trap(
    code="E001",
    file="src/app.py",
    message="syntax error",
    base_commit="HEAD",
    repo_key="repo_1",
    count=1,
    evidence_paths=None,
):
    return CompileTrap(
        kind="compile_trap",
        repo_key=repo_key,
        signature=(code, file, message),
        statement=f"Attempt failed Tier 0 with {code} in {file}: {message}",
        fix_hint="failed attempt only:\n-bad\npassing attempt only:\n+good",
        count=count,
        provenance=CompileTrapProvenance(
            run_id="run-1",
            subgoal_id="sg-1",
            attempt_no=1,
            base_commit=base_commit,
        ),
        evidence_paths=evidence_paths or ([file] if file else []),
    )


def test_knowledge_store_init_and_pragma(tmp_path, fake_clock):
    db_path = tmp_path / "k.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)
    assert store.count() == 0

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA user_version")
        ver = cur.fetchone()[0]
        assert ver == 1


def test_upsert_twice_idempotent_and_keeps_counters_and_first_seen(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(base_commit=base_commit, repo_key="repo_1", count=1)

    # First upsert
    res1 = store.upsert_traps([trap], repo_dir)
    assert res1.inserted == 1
    assert res1.updated == 0
    assert res1.skipped == 0
    assert store.count() == 1

    facts1 = store.list_facts()
    fact1 = facts1[0]
    first_seen_initial = fact1["first_seen"]
    assert fact1["injected_count"] == 0
    assert fact1["helped_count"] == 0

    # Simulate injection / helper update
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE facts SET injected_count = 5, helped_count = 2 WHERE id = ?", (fact1["id"],))
        conn.commit()

    # Advance clock
    fake_clock.advance(timedelta(hours=2))

    # Second upsert with identical trap
    res2 = store.upsert_traps([trap], repo_dir)
    assert res2.inserted == 0
    assert res2.updated == 1
    assert res2.skipped == 0

    facts2 = store.list_facts()
    fact2 = facts2[0]
    assert fact2["first_seen"] == first_seen_initial
    assert fact2["injected_count"] == 5
    assert fact2["helped_count"] == 2
    assert fact2["count"] == 1
    assert fact2["last_seen"] == fact1["last_seen"]
    assert fact2["updated_at"] > fact1["updated_at"]

    # Byte-identical apart from updated_at
    dict1 = {k: v for k, v in fact1.items() if k not in ("updated_at", "injected_count", "helped_count")}
    dict2 = {k: v for k, v in fact2.items() if k not in ("updated_at", "injected_count", "helped_count")}
    assert dict1 == dict2


def test_count_is_replaced_not_summed(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap1 = make_trap(base_commit=base_commit, repo_key="repo_1", count=3)
    store.upsert_traps([trap1], repo_dir)
    assert store.list_facts()[0]["count"] == 3

    trap2 = make_trap(base_commit=base_commit, repo_key="repo_1", count=5)
    store.upsert_traps([trap2], repo_dir)
    assert store.list_facts()[0]["count"] == 5  # Replaced, not 3+5=8


def test_repo_key_mismatch_skipped_and_counted(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap_mismatch = make_trap(base_commit=base_commit, repo_key="different_key")
    res = store.upsert_traps([trap_mismatch], repo_dir)
    assert res.inserted == 0
    assert res.updated == 0
    assert res.skipped == 1
    assert store.count() == 0

    # None repo_key is accepted and takes store's key
    trap_none = make_trap(base_commit=base_commit, repo_key=None)
    res2 = store.upsert_traps([trap_none], repo_dir)
    assert res2.inserted == 1
    assert res2.skipped == 0
    facts = store.list_facts()
    assert facts[0]["repo_key"] == "repo_1"


def test_evidence_directory_chain_and_root_not_siblings(tmp_path):
    repo_dir = tmp_path / "multi_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

    # Root env files
    (repo_dir / "package.json").write_text("{}", encoding="utf-8")
    (repo_dir / "tsconfig.json").write_text("{}", encoding="utf-8")

    # pkg-a chain
    pkg_a_dir = repo_dir / "packages" / "pkg-a"
    (pkg_a_dir / "src").mkdir(parents=True)
    (pkg_a_dir / "package.json").write_text("{}", encoding="utf-8")
    (pkg_a_dir / "tsconfig.json").write_text("{}", encoding="utf-8")
    (pkg_a_dir / "src" / "index.ts").write_text("export const a = 1;\n", encoding="utf-8")

    # pkg-b sibling chain
    pkg_b_dir = repo_dir / "packages" / "pkg-b"
    pkg_b_dir.mkdir(parents=True)
    (pkg_b_dir / "package.json").write_text("{}", encoding="utf-8")
    (pkg_b_dir / "tsconfig.json").write_text("{}", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "multi-package setup"], cwd=str(repo_dir), check=True, capture_output=True)

    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), check=True, capture_output=True, text=True)
    commit = rev.stdout.strip()

    evidence, status = capture_evidence(
        repo_dir=repo_dir,
        base_commit=commit,
        concerned_paths=["packages/pkg-a/src/index.ts"],
    )

    assert status == "active"
    keys = set(evidence.keys())
    assert "package.json" in keys
    assert "tsconfig.json" in keys
    assert "packages/pkg-a/package.json" in keys
    assert "packages/pkg-a/tsconfig.json" in keys
    assert "packages/pkg-a/src/index.ts" in keys

    # Unrelated sibling directories must not be included
    assert "packages/pkg-b/package.json" not in keys
    assert "packages/pkg-b/tsconfig.json" not in keys


def test_unknown_base_commit_gives_status_unverifiable(git_repo, tmp_path, fake_clock):
    repo_dir, _ = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(base_commit="0000000000000000000000000000000000000000")
    res = store.upsert_traps([trap], repo_dir)
    assert res.inserted == 1

    facts = store.list_facts()
    assert facts[0]["status"] == "unverifiable"
    assert facts[0]["evidence_json"] == "{}"


def test_is_stale_package_json_edited_env_changed(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(base_commit=base_commit)
    store.upsert_traps([trap], repo_dir)
    fact = store.list_facts()[0]

    # Fresh fact is not stale
    stale, reason = store.is_stale(fact, repo_dir)
    assert not stale
    assert reason == ""

    # Edit package.json at HEAD
    (repo_dir / "package.json").write_text('{"name": "test-pkg", "version": "2.0.0"}', encoding="utf-8")
    subprocess.run(["git", "add", "package.json"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "bump package.json"], cwd=str(repo_dir), check=True, capture_output=True)

    stale, reason = store.is_stale(fact, repo_dir)
    assert stale
    assert reason == "env_changed:package.json"


def test_is_stale_concerned_file_deleted_gives_file_missing(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(file="src/app.py", base_commit=base_commit)
    store.upsert_traps([trap], repo_dir)
    fact = store.list_facts()[0]

    # Remove src/app.py at HEAD
    subprocess.run(["git", "rm", "src/app.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "delete app.py"], cwd=str(repo_dir), check=True, capture_output=True)

    stale, reason = store.is_stale(fact, repo_dir)
    assert stale
    assert reason == "file_missing:src/app.py"


def test_is_stale_editing_concerned_file_content_is_not_stale(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(file="src/app.py", base_commit=base_commit)
    store.upsert_traps([trap], repo_dir)
    fact = store.list_facts()[0]

    # Edit src/app.py content only
    (repo_dir / "src" / "app.py").write_text("def hello():\n    return 'changed'\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/app.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "edit app.py content"], cwd=str(repo_dir), check=True, capture_output=True)

    stale, reason = store.is_stale(fact, repo_dir)
    assert not stale
    assert reason == ""


def test_is_stale_ttl_boundary(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", ttl_days=30, clock=fake_clock)

    trap = make_trap(base_commit=base_commit)
    store.upsert_traps([trap], repo_dir)
    fact = store.list_facts()[0]

    # Just inside TTL boundary (30 days exactly)
    fake_clock.advance(timedelta(days=30))
    stale, reason = store.is_stale(fact, repo_dir)
    assert not stale
    assert reason == ""

    # Just outside TTL boundary (30 days + 1 second)
    fake_clock.advance(timedelta(seconds=1))
    stale, reason = store.is_stale(fact, repo_dir)
    assert stale
    assert reason == "ttl"


def test_refresh_status_updates_stored_status(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(base_commit=base_commit)
    store.upsert_traps([trap], repo_dir)
    assert store.list_facts()[0]["status"] == "active"

    # Edit package.json
    (repo_dir / "package.json").write_text('{"name": "test-pkg", "version": "3.0.0"}', encoding="utf-8")
    subprocess.run(["git", "add", "package.json"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "edit package.json"], cwd=str(repo_dir), check=True, capture_output=True)

    counts = store.refresh_status(repo_dir)
    assert counts.active == 0
    assert counts.stale == 1
    assert counts.unverifiable == 0
    assert counts["stale"] == 1

    facts = store.list_facts()
    assert facts[0]["status"] == "stale"


def test_forget_one_and_forget_all_and_list_facts_order(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap1 = make_trap(code="E1", message="first error", base_commit=base_commit)
    store.upsert_traps([trap1], repo_dir)

    fake_clock.advance(timedelta(minutes=10))
    trap2 = make_trap(code="E2", message="second error", base_commit=base_commit)
    store.upsert_traps([trap2], repo_dir)

    fake_clock.advance(timedelta(minutes=10))
    trap3 = make_trap(code="E3", message="third error", base_commit=base_commit)
    store.upsert_traps([trap3], repo_dir)

    facts = store.list_facts()
    assert len(facts) == 3
    # Ordered by last_seen desc
    assert facts[0]["code"] == "E3"
    assert facts[1]["code"] == "E2"
    assert facts[2]["code"] == "E1"

    # Filter by status
    assert len(store.list_facts(status="active")) == 3
    assert len(store.list_facts(status="stale")) == 0

    # Forget one
    id_to_forget = facts[1]["id"]
    store.forget(id_to_forget)
    assert store.count() == 2
    remaining_ids = [f["id"] for f in store.list_facts()]
    assert id_to_forget not in remaining_ids

    # Forget all
    store.forget(None)
    assert store.count() == 0
    assert store.list_facts() == []


def test_corrupt_knowledge_db_recreated_with_warning(tmp_path, fake_clock, caplog):
    db_path = tmp_path / "knowledge.db"
    db_path.write_text("THIS IS NOT A VALID SQLITE DB", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    assert store.count() == 0
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "knowledge.db" in r.message]
    assert len(warning_records) == 1

    # Works normally after recreation
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA user_version")
        assert cur.fetchone()[0] == 1


def test_reads_work_on_readonly_connection(git_repo, tmp_path, fake_clock):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1", clock=fake_clock)

    trap = make_trap(base_commit=base_commit)
    store.upsert_traps([trap], repo_dir)

    # Open read-only sqlite connection directly and verify
    ro_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    ro_conn = sqlite3.connect(ro_uri, uri=True)
    try:
        cur = ro_conn.cursor()
        cur.execute("SELECT count(*) FROM facts")
        assert cur.fetchone()[0] == 1
    finally:
        ro_conn.close()

    # KnowledgeStore read methods
    assert store.count() == 1
    facts = store.list_facts()
    assert len(facts) == 1


def test_null_memory_noop_and_satisfies_protocol(git_repo, tmp_path):
    repo_dir, base_commit = git_repo
    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore(db_path, repo_key="repo_1")
    null_mem = NullMemory()

    # MemoryStore protocol checks
    assert isinstance(store, MemoryStore)
    assert isinstance(null_mem, MemoryStore)

    trap = make_trap(base_commit=base_commit)
    res = null_mem.upsert_traps([trap], repo_dir)
    assert res.inserted == 0
    assert res.updated == 0
    assert res.skipped == 1

    stale, reason = null_mem.is_stale({}, repo_dir)
    assert not stale
    assert reason == ""

    counts = null_mem.refresh_status(repo_dir)
    assert counts.active == 0
    assert counts.stale == 0
    assert counts.unverifiable == 0

    assert null_mem.list_facts() == []
    null_mem.forget("any_id")
    null_mem.forget()
    assert null_mem.count() == 0


def test_real_jev_home_isolation(tmp_path, monkeypatch):
    custom_home = tmp_path / "custom_jev"
    monkeypatch.setenv("JEV_HOME", str(custom_home))

    path = get_knowledge_db_path("my_repo_key")
    assert str(path).startswith(str(custom_home))
    assert path.name == "knowledge.db"
    assert path.parent.name == "my_repo_key"

    real_jev = Path.home() / ".jev"
    # Ensure real ~/.jev does not have this test repo created
    assert not (real_jev / "repos" / "my_repo_key" / "knowledge.db").exists()
