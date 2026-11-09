"""The models the evaluation and the editor actually talk to.

Every predictor has the same two methods:
- predict(code, k): code typed so far -> k suggestions for the next word
- predict_tokens(tokens, k): same thing when I already have the word tokens

so the editor and the evaluation can swap models without caring which one it is.
"""

import heapq
import math
import pickle
from pathlib import Path

from copilot.ngram import NGramModel
from copilot.tokenizers import tokens_to_text, word_tokens

MODELS = Path("models")


class WordPredictor:
    def __init__(self, model: NGramModel, n: int):
        self.model, self.n = model, n
        self.name = f"word-{n}"

    def predict_tokens(self, tokens: list[str], k: int = 5) -> list[str]:
        return [tok for tok, _ in self.model.top_k(tokens, self.n, k)]

    def predict(self, code: str, k: int = 5) -> list[str]:
        return self.predict_tokens(word_tokens(code), k)


class LetterPredictor:
    """Builds the next word one letter at a time until it writes a space.

    Uses a small beam search so it can give k different words, not just one.
    """

    def __init__(self, model: NGramModel, n: int, beam: int = 8,
                 chars_per_step: int = 8, max_len: int = 40):
        self.model, self.n = model, n
        self.beam, self.chars_per_step, self.max_len = beam, chars_per_step, max_len
        self.name = f"letter-{n}"

    def next_chars(self, context: list[str]) -> list[tuple[str, float]]:
        """The most likely next letters and their log probabilities.

        top_k fills up from shorter contexts, so even when the long context only
        ever had one next letter there are still other options (with small
        smoothed probabilities) and the beam doesn't run out of words.
        """
        return [(c, math.log(p)) for c, p in self.model.top_k(context, self.n, self.chars_per_step)]

    def predict_tokens(self, tokens: list[str], k: int = 5) -> list[str]:
        start = list(tokens_to_text(tokens))  # ends in a space, so we're at the start of a word
        beams = [(0.0, "")]
        done: list[tuple[float, str]] = []
        for _ in range(self.max_len):
            grown = []
            for logp, word in beams:
                for c, lp in self.next_chars(start + list(word)):
                    if c == " ":
                        if word:
                            done.append((logp + lp, word))
                    else:
                        grown.append((logp + lp, word + c))
            beams = heapq.nlargest(self.beam, grown)
            # adding letters only lowers the score, so once k finished words beat
            # every unfinished one, nothing can change the answer
            if not beams or (len(done) >= k and heapq.nlargest(k, done)[-1][0] >= beams[0][0]):
                break
        return [w for _, w in heapq.nlargest(k, done)]

    def predict(self, code: str, k: int = 5) -> list[str]:
        return self.predict_tokens(word_tokens(code), k)


def save(model: NGramModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> NGramModel:
    # only ever load pickles I made myself, loading one can run code
    with path.open("rb") as f:
        return pickle.load(f)


def get_predictor(kind: str, n: int, models_dir: Path = MODELS):
    """E.g. get_predictor("word", 3). Needs scripts/train.py to have been run."""
    if kind == "word":
        return WordPredictor(load(models_dir / "word.pkl"), n)
    if kind == "letter":
        return LetterPredictor(load(models_dir / "letter.pkl"), n)
    raise ValueError(f"kind has to be 'word' or 'letter', got {kind!r}")
