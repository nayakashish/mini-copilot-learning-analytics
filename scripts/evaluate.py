"""Evaluate all 20 n-gram models (word and letter, n = 1..10) on the test split.

Writes:
    results/metrics/ngram.csv             one row per model and target set
    results/metrics/ngram_by_project.csv  same, split up per test project

Needs the models from scripts/train.py. Run from the repo root:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --per-project 200   # quick run
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from copilot.evaluate import LAYOUT, TARGETS, load_files, pick_cut_points, run  # noqa: E402
from copilot.metrics import summarize  # noqa: E402
from copilot.predictors import MODELS, LetterPredictor, WordPredictor, load  # noqa: E402

TEST = Path("data/splits/test.txt")
OUT = Path("results/metrics")
PREDICTORS = {"word": WordPredictor, "letter": LetterPredictor}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-project", type=int, default=1000, help="cut points per test project")
    parser.add_argument("--seed", type=int, default=329)
    parser.add_argument("--max-n", type=int, default=10)
    args = parser.parse_args()

    files = load_files(TEST)
    cuts = {t: pick_cut_points(files, args.per_project, args.seed, t) for t in TARGETS}
    print(f"{len(files)} test files; cut points: " + ", ".join(f"{t} {len(c)}" for t, c in cuts.items()))

    rows, by_project = [], []
    for kind, make in PREDICTORS.items():
        model = load(MODELS / f"{kind}.pkl")
        for n in range(1, args.max_n + 1):
            predictor = make(model, n)
            start = time.time()
            for targets, points in cuts.items():
                drop = frozenset(LAYOUT) if targets == "words" else frozenset()
                scores = run(predictor, files, points, drop=drop)
                rows.append({"model": kind, "n": n, "targets": targets, **summarize(scores)})

                per = defaultdict(list)
                for cp, s in zip(points, scores):
                    per[cp.project].append(s)
                for project, s in sorted(per.items()):
                    by_project.append({"model": kind, "n": n, "targets": targets,
                                       "project": project, **summarize(s)})
            main_row = rows[-2]  # the "words" row
            print(f"{predictor.name:10} acc {main_row['accuracy']:.3f}  top5 {main_row['top5_accuracy']:.3f}  "
                  f"edit {main_row['edit_distance']:.2f} ({main_row['normalized_edit_distance']:.2f})  "
                  f"{time.time() - start:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("ngram.csv", rows), ("ngram_by_project.csv", by_project)):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            for r in data:
                w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"wrote {OUT}/ngram.csv and {OUT}/ngram_by_project.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
