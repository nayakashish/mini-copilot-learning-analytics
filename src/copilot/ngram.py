"""The n-gram model, used for both the word and the letter models.

One model holds the counts for every context length from 0 up to max_n - 1,
so the n = 1..10 models all come from the same counts. Asking for order n just
means using at most the last n - 1 tokens as context.

Smoothing is backoff + add-k:
- backoff: if the context was never seen in training, drop the oldest token
  and try again, all the way down to no context (plain frequencies)
- add-k: at the level I end up on, every token gets k added to its count, so
  nothing ever has probability 0
- tokens seen fewer than min_count times in training become <UNK>, so the
  model has learned something about where unknown names show up
"""

from collections import Counter, defaultdict
from typing import Iterable, Sequence

UNK = "<UNK>"


class NGramModel:
    def __init__(self, max_n: int = 10, k: float = 0.1, min_count: int = 2):
        self.max_n = max_n
        self.k = k
        self.min_count = min_count
        self.vocab: set[str] = set()
        # counts[m][context of m tokens][next token]
        self.counts: list[dict[tuple, Counter]] = []
        self.totals: list[dict[tuple, int]] = []

    def fit(self, sequences: Sequence[Sequence[str]]) -> "NGramModel":
        """Count everything. Each sequence is one file, so contexts never run across files."""
        freq = Counter(tok for seq in sequences for tok in seq)
        self.vocab = {t for t, c in freq.items() if c >= self.min_count} | {UNK}

        counts = [defaultdict(Counter) for _ in range(self.max_n)]
        for seq in sequences:
            seq = self.map_unk(seq)
            for i, tok in enumerate(seq):
                for m in range(min(i, self.max_n - 1) + 1):
                    counts[m][tuple(seq[i - m:i])][tok] += 1

        self.counts = [dict(level) for level in counts]
        self.totals = [{ctx: sum(c.values()) for ctx, c in level.items()} for level in self.counts]
        return self

    def map_unk(self, tokens: Iterable[str]) -> list[str]:
        return [t if t in self.vocab else UNK for t in tokens]

    def backoff(self, context: Sequence[str], n: int):
        """Yield (m, counts after that context) from the longest usable context down to none."""
        if not 1 <= n <= self.max_n:
            raise ValueError(f"n has to be 1..{self.max_n}, got {n}")
        ctx = self.map_unk(context[-(n - 1):]) if n > 1 else []
        for m in range(len(ctx), -1, -1):
            key = tuple(ctx[len(ctx) - m:])
            if key in self.counts[m]:
                yield m, key

    def distribution(self, context: Sequence[str], n: int) -> tuple[Counter, int]:
        """Counts after the longest context I've seen, and their total."""
        m, key = next(self.backoff(context, n))
        return self.counts[m][key], self.totals[m][key]

    def prob(self, token: str, context: Sequence[str], n: int) -> float:
        nexts, total = self.distribution(context, n)
        tok = token if token in self.vocab else UNK
        return (nexts.get(tok, 0) + self.k) / (total + self.k * len(self.vocab))

    def top_k(self, context: Sequence[str], n: int, k: int = 5,
              skip: frozenset = frozenset({UNK})) -> list[tuple[str, float]]:
        """The k most likely next tokens with their probabilities.

        Starts with the longest seen context and fills up from shorter ones if
        it had fewer than k different next tokens.
        """
        picked: list[str] = []
        for m, key in self.backoff(context, n):
            for tok, _ in self.counts[m][key].most_common():
                if tok not in skip and tok not in picked:
                    picked.append(tok)
                    if len(picked) == k:
                        break
            if len(picked) == k:
                break
        return [(tok, self.prob(tok, context, n)) for tok in picked]
