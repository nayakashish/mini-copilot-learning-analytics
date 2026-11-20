"""Show where the AST model changes the top-1 guess compared to word-4.

Counts how often it helps (word-4 wrong, AST right), hurts (the other way) or
changes it without either being right, and prints a few real examples. Same
cut points as the evaluation ("words" spots, seed 329). Run from the repo root:
    python3 scripts/ast_examples.py
    python3 scripts/ast_examples.py --per-project 250   # smaller sample
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
from copilot.ast_model import AstPredictor  # noqa: E402
from copilot.evaluate import LAYOUT, load_files, pick_cut_points  # noqa: E402
from copilot.predictors import MODELS, WordPredictor, load  # noqa: E402

DROP = frozenset(LAYOUT)


def top1(predictor, tokens):
    guesses = [g for g in predictor.predict_tokens(tokens, 1 + len(DROP)) if g not in DROP]
    return guesses[0] if guesses else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-project", type=int, default=250)
    parser.add_argument("--examples", type=int, default=4, help="examples to print per group")
    args = parser.parse_args()

    files = load_files(Path("data/splits/test.txt"))
    cuts = pick_cut_points(files, args.per_project, 329, "words")
    base = WordPredictor(load(MODELS / "word.pkl"), 4)
    ast_model = AstPredictor(base)

    groups = {"helps": [], "hurts": [], "changes, both wrong": []}
    same = 0
    for cp in cuts:
        tokens = files[cp.path]
        before, after = top1(base, tokens[:cp.index]), top1(ast_model, tokens[:cp.index])
        target = tokens[cp.index]
        if before == after:
            same += 1
        elif after == target:
            groups["helps"].append((cp, before, after))
        elif before == target:
            groups["hurts"].append((cp, before, after))
        else:
            groups["changes, both wrong"].append((cp, before, after))

    print(f"{len(cuts)} cut points, top-1 unchanged on {same}")
    for name, items in groups.items():
        print(f"\n{name}: {len(items)}")
        for cp, before, after in items[:args.examples]:
            tokens = files[cp.path]
            context = " ".join(tokens[max(0, cp.index - 8):cp.index])
            print(f"  ...{context}  [real: {tokens[cp.index]}]  word-4: {before}  ast: {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
