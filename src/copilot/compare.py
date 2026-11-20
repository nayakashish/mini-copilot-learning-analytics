"""The side-by-side table of the models, built straight from the results csvs.

No numbers are typed in by hand: it reads results/metrics/ngram.csv and
ast.csv, so rerunning the evaluation and this gives the table again.
"""

import csv
from pathlib import Path

COLUMNS = ["model", "accuracy", "top5_accuracy", "edit_distance", "normalized_edit_distance"]


def read_rows(path: Path, targets: str = "words") -> list[dict]:
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r["targets"] == targets]


def name(row: dict) -> str:
    return f"{row['model']}-{row['n']}"


def comparison(ngram_rows: list[dict], ast_rows: list[dict]) -> list[dict]:
    """The AST model, the word model it sits on, and the best n of each family.

    "Best" is the highest top-1 accuracy. Each model shows up once.
    """
    chosen = [r for r in ast_rows if r["model"] == "ast"]
    chosen += [r for r in ast_rows if r["model"] == "word"]  # the AST model's base
    for family in ("word", "letter"):
        mine = [r for r in ngram_rows if r["model"] == family]
        chosen.append(max(mine, key=lambda r: float(r["accuracy"])))

    table, seen = [], set()
    for r in chosen:
        if name(r) not in seen:
            seen.add(name(r))
            table.append({"model": name(r), **{c: r[c] for c in COLUMNS[1:]}})
    return sorted(table, key=lambda r: -float(r["accuracy"]))


def to_markdown(table: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(r[c] for c in COLUMNS) + " |" for r in table]
    return "\n".join(lines)
