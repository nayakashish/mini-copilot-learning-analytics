"""Running the models on the test projects and scoring them.

For each test project I pick the same number of random spots (cut points) in
its files, seeded so it's the same spots every run. At each spot the model
sees the code before it and has to guess the next word, and I score the guess
with the metrics. Every model gets the exact same spots, so they're compared
fairly.

Two sets of spots:
- "words": the next word is a real word (name, keyword, symbol, <STR>, <NUM>).
  This is the main score, since that's what an editor would suggest.
- "all": the next word can also be <NEWLINE>, <INDENT> or <DEDENT>. These are
  generally easy to guess, so these ones were tested just for my interest in the results.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from copilot.metrics import Score, score
from copilot.tokenizers import DEDENT, INDENT, NEWLINE, word_tokens

LAYOUT = {NEWLINE, INDENT, DEDENT}
TARGETS = ("words", "all")


@dataclass(frozen=True)
class CutPoint:
    project: str
    path: str
    index: int  # the target is tokens[index], the context is tokens[:index]


def project_of(path: str) -> str:
    """data/raw/<project>/... -> <project>"""
    return Path(path).parts[2]


def load_files(split_file: Path) -> dict[str, list[str]]:
    """Word tokens for every file in a split, keyed by path."""
    files = {}
    for path in split_file.read_text().split():
        files[path] = word_tokens(Path(path).read_text(encoding="utf-8", errors="replace"))
    return files


def pick_cut_points(files: dict[str, list[str]], per_project: int, seed: int,
                    targets: str = "words") -> list[CutPoint]:
    """per_project random spots from each project (all of them if it has fewer).

    Spot 0 is skipped so there's always at least one word of context.
    """
    if targets not in TARGETS:
        raise ValueError(f"targets has to be one of {TARGETS}, got {targets!r}")
    spots = defaultdict(list)
    for path in sorted(files):
        for i, tok in enumerate(files[path]):
            if i > 0 and (targets == "all" or tok not in LAYOUT):
                spots[project_of(path)].append(CutPoint(project_of(path), path, i))

    picked = []
    for project in sorted(spots):
        rng = random.Random(f"{seed}:{project}:{targets}")
        options = spots[project]
        picked += rng.sample(options, min(per_project, len(options)))
    return picked


def run(predictor, files: dict[str, list[str]], cut_points: list[CutPoint],
        k: int = 5, drop: frozenset = frozenset()) -> list[Score]:
    """Ask the predictor at every cut point and score what it says.

    Anything in `drop` gets taken out of the suggestions first. For the "words"
    score I drop the layout tokens, because the editor would never suggest a
    newline either.
    """
    scores = []
    for cp in cut_points:
        tokens = files[cp.path]
        # ask for a few extra so there are still k left after dropping
        suggestions = predictor.predict_tokens(tokens[:cp.index], k + len(drop))
        suggestions = [s for s in suggestions if s not in drop][:k]
        scores.append(score(suggestions, tokens[cp.index], k))
    return scores
