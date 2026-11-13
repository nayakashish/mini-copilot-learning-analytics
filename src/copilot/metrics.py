"""The two metrics: accuracy and edit distance.

Both compare what a model suggested to the word that really came next.

- accuracy: top-1 = the first suggestion is exactly right,
  top-5 = the right word is anywhere in the 5 suggestions
- edit distance: Levenshtein distance between the first suggestion and the
  right word (inserts, deletes, swaps of single letters), raw and divided by
  the length of the right word

No suggestion at all counts as a miss, with distance = length of the right word
(as if I had to type the whole thing myself).
"""

from dataclasses import dataclass


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, the usual dynamic programming way, one row at a time."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,               # delete
                cur[j - 1] + 1,            # insert
                prev[j - 1] + (ca != cb),  # swap (free if same letter)
            ))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class Score:
    top1: bool
    top5: bool
    distance: int
    normalized: float


def score(suggestions: list[str], target: str, k: int = 5) -> Score:
    """Score one prediction: the model's suggestions (best first) vs the real next word."""
    if not target:
        raise ValueError("target can't be empty")
    best = suggestions[0] if suggestions else ""
    dist = edit_distance(best, target)
    return Score(
        top1=best == target,
        top5=target in suggestions[:k],
        distance=dist,
        normalized=dist / len(target),
    )


def summarize(scores: list[Score]) -> dict[str, float]:
    """Averages over a bunch of predictions, which is what goes in the results table."""
    n = len(scores)
    if n == 0:
        return {"count": 0, "accuracy": 0.0, "top5_accuracy": 0.0,
                "edit_distance": 0.0, "normalized_edit_distance": 0.0}
    return {
        "count": n,
        "accuracy": sum(s.top1 for s in scores) / n,
        "top5_accuracy": sum(s.top5 for s in scores) / n,
        "edit_distance": sum(s.distance for s in scores) / n,
        "normalized_edit_distance": sum(s.normalized for s in scores) / n,
    }
