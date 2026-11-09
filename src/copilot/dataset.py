"""Cleaning the fetched projects and splitting them into train/test.

The idea is to keep code that looks like what someone would actually be
writing, so tests, docs, examples etc. get dropped.
"""

import ast
import hashlib
import io
import random
import re
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    "test", "tests", "testing", "doc", "docs", "example", "examples",
    "sample", "samples", "benchmark", "benchmarks", "vendor", "vendored",
    "_vendor", "build", "dist",
}
SKIP_FILES = re.compile(r"^(test_.*|.*_test|conftest|setup)\.py$")

# these don't count as real tokens for sizing a project
NOT_COUNTED = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}


@dataclass(frozen=True)
class SourceFile:
    project: str
    path: str  # relative to the repo root, e.g. data/raw/flask/src/flask/app.py
    tokens: int


def count_tokens(src: str) -> int:
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    return sum(1 for t in toks if t.type not in NOT_COUNTED)


def skip_reason(rel_path: str, src: str, seen: set[str]) -> str | None:
    """Why a file gets dropped, or None if I keep it. Adds kept files to `seen`."""
    parts = rel_path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]) or SKIP_FILES.match(parts[-1]):
        return "test/docs/example"
    if not src.strip():
        return "empty"
    try:
        ast.parse(src)
    except (SyntaxError, ValueError):
        return "doesn't parse"  # mostly newer syntax than 3.10
    digest = hashlib.sha1(src.encode()).hexdigest()
    if digest in seen:
        return "duplicate"
    seen.add(digest)
    return None


def collect_project(project_dir: Path, seen: set[str]) -> tuple[list[SourceFile], Counter]:
    """Every .py file in a project that passes the filters, plus counts of what got dropped."""
    kept, dropped = [], Counter()
    for f in sorted(project_dir.rglob("*.py")):
        rel = f.relative_to(project_dir).as_posix()
        if rel.startswith(".git/"):
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        reason = skip_reason(rel, src, seen)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(SourceFile(project_dir.name, f.as_posix(), count_tokens(src)))
    return kept, dropped


def cap_project(files: list[SourceFile], max_tokens: int, seed: int) -> list[SourceFile]:
    """Randomly keep files until the project hits max_tokens, so no project takes over."""
    if sum(f.tokens for f in files) <= max_tokens:
        return list(files)
    order = sorted(files, key=lambda f: f.path)
    random.Random(f"{seed}:{order[0].project}").shuffle(order)
    picked, total = [], 0
    for f in order:
        if total + f.tokens <= max_tokens:
            picked.append(f)
            total += f.tokens
    return sorted(picked, key=lambda f: f.path)


def split_by_project(files: list[SourceFile], test_frac: float, seed: int) -> tuple[list, list]:
    """Whole projects go to test until test is as close to test_frac of the tokens as I can get.

    A project is never on both sides, so the test projects are code the model has never seen.
    """
    sizes = Counter()
    for f in files:
        sizes[f.project] += f.tokens
    total = sum(sizes.values())

    projects = sorted(sizes)
    random.Random(seed).shuffle(projects)
    test, test_tokens = set(), 0
    for p in projects:
        if test_tokens >= test_frac * total:
            break
        # only add it if it gets me closer to the target
        if abs(test_tokens + sizes[p] - test_frac * total) < abs(test_tokens - test_frac * total):
            test.add(p)
            test_tokens += sizes[p]

    train = [f for f in files if f.project not in test]
    return train, [f for f in files if f.project in test]


def split_by_file(files: list[SourceFile], test_frac: float, seed: int) -> tuple[list, list]:
    order = sorted(files, key=lambda f: f.path)
    random.Random(seed).shuffle(order)
    n_test = round(len(order) * test_frac)
    test = sorted(order[:n_test], key=lambda f: f.path)
    train = sorted(order[n_test:], key=lambda f: f.path)
    return train, test


def split(files: list[SourceFile], unit: str, test_frac: float, seed: int) -> tuple[list, list]:
    if unit == "project":
        return split_by_project(files, test_frac, seed)
    if unit == "file":
        return split_by_file(files, test_frac, seed)
    raise ValueError(f"unit has to be 'project' or 'file', got {unit!r}")
