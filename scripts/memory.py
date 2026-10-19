#!/usr/bin/env python3
"""CLI tool for managing and inspecting the Jev KnowledgeStore (memory)."""

import argparse
from pathlib import Path
import re
import sys
from typing import List, Optional, Tuple

# Ensure repo's src directory is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from jev.knowledge import KnowledgeStore, get_knowledge_db_path
from jev.ledger import compute_repo_key, get_jev_home
from jev.memory_extract import extract_from_ledger


def build_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Path to repository to inspect",
    )
    common_parser.add_argument(
        "--repo-key",
        type=str,
        default=None,
        help="16-hex repo key to inspect",
    )
    common_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Direct path to knowledge.db database",
    )

    parser = argparse.ArgumentParser(
        description="Inspect and manage Jev State Engine compile error memories",
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: list
    parser_list = subparsers.add_parser("list", parents=[common_parser], help="List facts in knowledge store")
    parser_list.add_argument(
        "--status",
        choices=["active", "stale", "unverifiable"],
        default=None,
        help="Filter facts by status",
    )
    parser_list.add_argument(
        "--check",
        action="store_true",
        help="Evaluate staleness against current HEAD (requires --repo-dir)",
    )

    # Subcommand: forget
    parser_forget = subparsers.add_parser("forget", parents=[common_parser], help="Delete facts from knowledge store")
    forget_group = parser_forget.add_mutually_exclusive_group(required=True)
    forget_group.add_argument("--id", type=str, help="Specific fact ID to forget")
    forget_group.add_argument("--all", action="store_true", help="Forget all facts in store")
    parser_forget.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt for --all")

    # Subcommand: rebuild
    parser_rebuild = subparsers.add_parser("rebuild", help="Rebuild knowledge store from repo's ledger")
    parser_rebuild.add_argument(
        "--repo-dir",
        type=str,
        required=True,
        help="Path to repository to rebuild from",
    )
    parser_rebuild.add_argument(
        "--repo-key",
        type=str,
        default=None,
        help="16-hex repo key to inspect",
    )
    parser_rebuild.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Direct path to knowledge.db database",
    )

    return parser


def resolve_repo_and_db(args: argparse.Namespace) -> Tuple[str, Path, Optional[Path]]:
    """Resolves (repo_key, knowledge_db_path, repo_dir_path)."""
    repo_dir = Path(args.repo_dir).resolve() if getattr(args, "repo_dir", None) else None

    if getattr(args, "db_path", None):
        knowledge_db = Path(args.db_path).resolve()
        repo_key = getattr(args, "repo_key", None)
        if not repo_key and repo_dir:
            repo_key = compute_repo_key(repo_dir)
        if not repo_key:
            posix_path = knowledge_db.as_posix()
            m = re.search(r'(?:^|/)repos/([^/]+)/knowledge\.db$', posix_path, re.IGNORECASE)
            repo_key = m.group(1) if m else "unknown"
        return repo_key, knowledge_db, repo_dir

    repo_key = getattr(args, "repo_key", None)
    if not repo_key and repo_dir:
        repo_key = compute_repo_key(repo_dir)
    if not repo_key:
        repo_key = compute_repo_key(Path.cwd())

    knowledge_db = get_jev_home() / "repos" / repo_key / "knowledge.db"
    return repo_key, knowledge_db, repo_dir


def cmd_list(args: argparse.Namespace) -> int:
    if args.check and not getattr(args, "repo_dir", None):
        print("Error: --check requires --repo-dir", file=sys.stderr)
        return 1

    repo_key, db_path, repo_dir = resolve_repo_and_db(args)

    if not db_path.exists():
        print("no facts")
        return 0

    store = KnowledgeStore(db_path, repo_key=repo_key, readonly=True)
    facts = store.list_facts(status=args.status)
    if not facts:
        print("no facts")
        return 0

    for i, fact in enumerate(facts):
        fix_hint = fact.get("fix_hint")
        if fix_hint and fix_hint.strip():
            hint_first_line = fix_hint.strip().splitlines()[0]
        else:
            hint_first_line = "none"

        print(f"id: {fact['id']}")
        print(f"status: {fact['status']}")
        print(f"count: {fact['count']}")
        print(f"last_seen: {fact['last_seen']}")
        print(f"code: {fact.get('code') or ''}")
        print(f"file: {fact.get('file') or ''}")
        print(f"message: {fact.get('message') or ''}")
        print(f"fix_hint: {hint_first_line}")

        if args.check and repo_dir:
            stale, reason = store.is_stale(fact, repo_dir)
            if stale:
                print(f"check: stale {reason}")
            else:
                print("check: ok")

        if i < len(facts) - 1:
            print()
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    repo_key, db_path, _ = resolve_repo_and_db(args)
    if not db_path.exists():
        print("Removed 0 facts.")
        return 0

    if args.all and not args.yes:
        ans = input("Delete all facts? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    store = KnowledgeStore(db_path, repo_key=repo_key)
    if args.all:
        deleted = store.forget(None)
    else:
        deleted = store.forget(args.id)

    print(f"Removed {deleted} fact(s).")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    repo_dir = Path(args.repo_dir).resolve()
    repo_key = args.repo_key or compute_repo_key(repo_dir)

    ledger_path = get_jev_home() / "repos" / repo_key / "ledger.db"
    if not ledger_path.exists():
        print("no ledger found")
        return 0

    traps = extract_from_ledger(ledger_path, repo_key=repo_key)

    kdb_path = get_knowledge_db_path(repo_key)
    store = KnowledgeStore(kdb_path, repo_key=repo_key)
    res = store.upsert_traps(traps, repo_dir)
    counts = store.refresh_status(repo_dir)

    print(f"inserted: {res.inserted}, updated: {res.updated}, skipped: {res.skipped}")
    print(f"active: {counts.active}, stale: {counts.stale}, unverifiable: {counts.unverifiable}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "list":
        return cmd_list(args)
    elif args.subcommand == "forget":
        return cmd_forget(args)
    elif args.subcommand == "rebuild":
        return cmd_rebuild(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
