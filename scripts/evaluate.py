"""Evaluate the models on the test split.

Default: all 20 n-gram models (word and letter, n = 1..10). Writes
    results/metrics/ngram.csv             one row per model and target set
    results/metrics/ngram_by_project.csv  same, split up per test project

With --model ast: the AST model on top of word-4 (change with --ast-base). Writes
    results/metrics/ast.csv and results/metrics/ast_by_project.csv

Needs the models from scripts/train.py. Run from the repo root:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --model ast
    python3 scripts/evaluate.py --model ast --per-project 200   # quick run
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
from copilot.ast_model import AstPredictor  # noqa: E402
from copilot.predictors import MODELS, LetterPredictor, WordPredictor, load  # noqa: E402

TEST = Path("data/splits/test.txt")
OUT = Path("results/metrics")
PREDICTORS = {"word": WordPredictor, "letter": LetterPredictor}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-project", type=int, default=1000, help="cut points per test project")
    parser.add_argument("--seed", type=int, default=329)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--model", choices=("ngram", "ast"), default="ngram")
    parser.add_argument("--ast-base", type=int, default=4, help="word model n that the AST model reorders")
    args = parser.parse_args()

    files = load_files(TEST)
    cuts = {t: pick_cut_points(files, args.per_project, args.seed, t) for t in TARGETS}
    print(f"{len(files)} test files; cut points: " + ", ".join(f"{t} {len(c)}" for t, c in cuts.items()))

    if args.model == "ast":
        word = load(MODELS / "word.pkl")
        # the AST model, and the word model it reorders, on the same spots
        jobs = [("ast", args.ast_base, AstPredictor(WordPredictor(word, args.ast_base))),
                ("word", args.ast_base, WordPredictor(word, args.ast_base))]
        prefix = "ast"
    else:
        jobs = [(kind, n, make(load(MODELS / f"{kind}.pkl"), n))
                for kind, make in PREDICTORS.items() for n in range(1, args.max_n + 1)]
        prefix = "ngram"

    rows, by_project = [], []
    for kind, n, predictor in jobs:
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
    for name, data in ((f"{prefix}.csv", rows), (f"{prefix}_by_project.csv", by_project)):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            for r in data:
                w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"wrote {OUT}/{prefix}.csv and {OUT}/{prefix}_by_project.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
