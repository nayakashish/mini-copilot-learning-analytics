"""Download every project in data/sources.csv into data/raw/<name>/ at the commit I pinned.

Only the .py files get checked out (and only the folders in `paths` if it's
set), so none of the other stuff in these repos ends up on disk.

Fine to run again: projects already at the right commit get skipped. It still
redoes the file filter though, so changing `paths` works without re-downloading.

Run from the repo root:
    python3 scripts/fetch_data.py                 # every project
    python3 scripts/fetch_data.py flask sherlock  # only these
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

SOURCES = Path("data/sources.csv")
RAW = Path("data/raw")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def sparse_patterns(paths: str) -> list[str]:
    """Turn the paths column into sparse-checkout patterns.

    Empty means every .py file. Otherwise each entry (split on ;) is either a
    .py file or a folder, and for folders I take all the .py files inside.
    """
    entries = [p.strip().strip("/") for p in paths.split(";") if p.strip()]
    if not entries:
        return ["*.py"]
    return [f"/{p}" if p.endswith(".py") else f"/{p}/**/*.py" for p in entries]


def current_commit(repo: Path) -> str | None:
    try:
        return git(repo, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return None  # nothing checked out yet


def fetch(row: dict[str, str]) -> None:
    name, url, sha = row["name"], row["repo_url"], row["commit"]
    dest = RAW / name

    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        git(dest, "init", "-q")
        git(dest, "remote", "add", "origin", url)
        git(dest, "config", "core.sparseCheckout", "true")

    (dest / ".git" / "info").mkdir(exist_ok=True)
    (dest / ".git" / "info" / "sparse-checkout").write_text(
        "\n".join(sparse_patterns(row["paths"])) + "\n"
    )

    if current_commit(dest) == sha:
        git(dest, "sparse-checkout", "reapply")
        print(f"  {name}: already at {sha[:10]}, patterns re-applied")
        return

    print(f"  {name}: fetching {sha[:10]} from {url}")
    # just this one commit, no history, and only download the files I actually check out
    git(dest, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha)
    git(dest, "checkout", "-q", "--detach", sha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="only fetch these projects")
    args = parser.parse_args()

    with SOURCES.open(newline="") as f:
        rows = [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]

    unknown = set(args.names) - {r["name"] for r in rows}
    if unknown:
        print(f"Unknown project(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if args.names:
        rows = [r for r in rows if r["name"] in args.names]

    failed = []
    for row in rows:
        try:
            fetch(row)
        except subprocess.CalledProcessError as e:
            print(f"  {row['name']}: FAILED\n{e.stderr}", file=sys.stderr)
            failed.append(row["name"])

    print(f"Done: {len(rows) - len(failed)}/{len(rows)} projects in {RAW}/")
    if failed:
        print(f"Failed: {', '.join(failed)}  (rerun to retry just these)", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
