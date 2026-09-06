"""Check code-only publication content with the Python standard library.

The default reads staged Git blobs, not possibly different working-tree files.
Findings include paths/categories only; matching secret text is never printed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from collections.abc import Iterator


ALLOWED_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".html",
    ".cff", ".lock",
}
ALLOWED_NAMES = {".gitignore", ".gitattributes"}
RESTRICTED_DIRECTORIES = {
    "data", "raw", "results", "runs", "experiments", "analysis_output",
    "publication_figures", "checkpoints", "uploads", "site-packages",
}
WORKTREE_PRUNED = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", "site-packages", "node_modules",
}
SECRET_PATTERNS = {
    "private-key material": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "GitHub credential": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "cloud access-key identifier": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "service API credential": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "literal credential assignment": re.compile(
        r"(?im)\b(?:api_key|access_token|auth_token|password|client_secret)"
        r"\s*[:=]\s*[\"'][A-Za-z0-9_+/=-]{20,}[\"']"
    ),
    "credential-bearing URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
}


def check_content(relative_path: str, content: bytes, mode: str = "100644") -> list[str]:
    """Return publication-boundary violations without including sensitive values."""
    path = PurePosixPath(relative_path)
    findings: list[str] = []
    if path.is_absolute() or ".." in path.parts:
        findings.append("non-relative path")
    if mode not in {"100644", "100755"}:
        findings.append("symlink, submodule or unsupported Git mode")
    if any(part.lower() in RESTRICTED_DIRECTORIES for part in path.parts):
        findings.append("controlled data/output/dependency directory")
    if path.name.startswith(".env"):
        findings.append("environment/credential file")
    if path.name not in ALLOWED_NAMES and path.suffix.lower() not in ALLOWED_SUFFIXES:
        findings.append("non-source artifact type")
    if len(content) > 2_000_000:
        findings.append("oversized source file; review required")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return findings + ["binary or non-UTF-8 content"]
    if "\x00" in text:
        findings.append("binary content")
    findings.extend(name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text))
    return findings


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError("Git read failed; initialize/stage source or use --worktree.")
    return result.stdout


def index_entries(root: Path) -> Iterator[tuple[str, str, bytes]]:
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise RuntimeError("Unresolved index entries must be resolved before publication.")
        yield path_bytes.decode("utf-8"), mode, git(root, "cat-file", "-p", oid)


def history_entries(root: Path) -> Iterator[tuple[str, str, bytes]]:
    seen: set[tuple[str, str]] = set()
    commits = git(root, "rev-list", "--all").decode("ascii").splitlines()
    if not commits:
        raise RuntimeError("No reachable commit exists for history verification.")
    for commit in commits:
        for record in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not record:
                continue
            metadata, path_bytes = record.split(b"\t", 1)
            mode, object_type, oid = metadata.decode("ascii").split()
            path = path_bytes.decode("utf-8")
            if (path, oid) in seen:
                continue
            seen.add((path, oid))
            content = git(root, "cat-file", "-p", oid) if object_type == "blob" else b""
            yield path, mode, content


def worktree_entries(root: Path) -> Iterator[tuple[str, str, bytes]]:
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(subdirectories):
            candidate = current / name
            if candidate.is_symlink():
                yield candidate.relative_to(root).as_posix(), "120000", b""
                subdirectories.remove(name)
            elif name in WORKTREE_PRUNED or name.endswith(".egg-info"):
                subdirectories.remove(name)
        for name in filenames:
            candidate = current / name
            mode = "120000" if candidate.is_symlink() else "100644"
            yield candidate.relative_to(root).as_posix(), mode, (
                b"" if candidate.is_symlink() else candidate.read_bytes()
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-history", action="store_true", help="Scan all locally reachable commits")
    group.add_argument("--worktree", action="store_true", help="Scan source before Git initialization")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    entries = history_entries(root) if args.all_history else (
        worktree_entries(root) if args.worktree else index_entries(root)
    )
    checked = 0
    failures = 0
    try:
        for path, mode, content in entries:
            checked += 1
            findings = check_content(path, content, mode)
            if findings:
                failures += 1
                print(f"FAIL {path}: {'; '.join(findings)}")
    except (RuntimeError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not checked:
        print("ERROR: No source files were checked.", file=sys.stderr)
        return 2
    print(f"{'PASS' if not failures else 'FAIL'}: {checked} source versions checked; {failures} offending files.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
