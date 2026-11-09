"""Clean the projects in data/raw/, cap their size, and write the 70/30 split.

Writes data/splits/train.txt and data/splits/test.txt (one file path per line)
and data/stats.csv with the numbers per project.

Run from the repo root:
    python3 scripts/make_splits.py
    python3 scripts/make_splits.py --unit file   # split by file instead
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from copilot.dataset import cap_project, collect_project, split  # noqa: E402

RAW = Path("data/raw")
SPLITS = Path("data/splits")
STATS = Path("data/stats.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", choices=["project", "file"], default="project")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=80_000, help="cap per project")
    parser.add_argument("--seed", type=int, default=329)
    args = parser.parse_args()

    if not RAW.exists():
        print("No data/raw/ yet, run scripts/fetch_data.py first", file=sys.stderr)
        return 1

    seen: set[str] = set()
    files, rows = [], {}
    for project_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        kept, dropped = collect_project(project_dir, seen)
        used = cap_project(kept, args.max_tokens, args.seed)
        files += used
        rows[project_dir.name] = {
            "project": project_dir.name,
            "files_kept": len(kept),
            "tokens_kept": sum(f.tokens for f in kept),
            "files_used": len(used),
            "tokens_used": sum(f.tokens for f in used),
            "dropped": "; ".join(f"{k}: {v}" for k, v in sorted(dropped.items())),
        }

    train, test = split(files, args.unit, args.test_frac, args.seed)

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("test", test)):
        (SPLITS / f"{name}.txt").write_text("".join(f.path + "\n" for f in part))

    total = sum(r["tokens_used"] for r in rows.values())
    sides = Counter()
    for side, part in (("train", train), ("test", test)):
        for f in part:
            sides[(f.project, side)] += 1
    for r in rows.values():
        r["share"] = f"{100 * r['tokens_used'] / total:.1f}%"
        on = [s for s in ("train", "test") if sides[(r["project"], s)]]
        r["split"] = "+".join(on) if on else "-"

    cols = ["project", "files_kept", "tokens_kept", "files_used", "tokens_used", "share", "split", "dropped"]
    with STATS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows.values())

    print(f"{'project':24} {'files':>5} {'tokens':>7} {'share':>6}  split")
    for r in rows.values():
        print(f"{r['project']:24} {r['files_used']:5} {r['tokens_used']:7} {r['share']:>6}  {r['split']}")
    train_tok, test_tok = sum(f.tokens for f in train), sum(f.tokens for f in test)
    print(f"\nsplit by {args.unit}, seed {args.seed}")
    print(f"train: {len(train)} files, {train_tok} tokens ({100 * train_tok / total:.1f}%)")
    print(f"test:  {len(test)} files, {test_tok} tokens ({100 * test_tok / total:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
