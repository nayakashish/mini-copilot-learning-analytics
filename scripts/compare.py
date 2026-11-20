"""Side-by-side table of the AST model, word-4 and the best word and letter n-gram.

Reads results/metrics/ngram.csv and ast.csv, writes results/metrics/comparison.csv
and prints the same table as markdown. Run from the repo root after both
evaluations:
    python3 scripts/compare.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
from copilot.compare import COLUMNS, comparison, read_rows, to_markdown  # noqa: E402

METRICS = Path("results/metrics")


def main() -> int:
    for name in ("ngram.csv", "ast.csv"):
        if not (METRICS / name).exists():
            print(f"missing {METRICS / name}, run scripts/evaluate.py first "
                  f"(add --model ast for ast.csv)", file=sys.stderr)
            return 1
    table = comparison(read_rows(METRICS / "ngram.csv"), read_rows(METRICS / "ast.csv"))
    with (METRICS / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    print(to_markdown(table))
    print(f"\nwrote {METRICS}/comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
