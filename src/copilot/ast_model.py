"""The AST model: word model suggestions, reordered by whether the code still parses.

It isn't a new model from scratch. The word model gives its best guesses, then
for each guess I write out the code with that guess added, repair the cut-off
ending (repair.py) and ask Python's parser if it's still valid. Guesses that
keep the code valid go first, guesses that break it go last. Inside each group
the word model's order stays, so any difference from the word model comes from
the syntax check alone.
"""

from copilot.predictors import WordPredictor
from copilot.repair import repair
from copilot.tokenizers import DEDENT, INDENT, NEWLINE, NUM, STR, word_tokens

STAND_IN = {STR: '"s"', NUM: "0"}  # real text for the placeholder tokens
INDENT_SIZE = 4


def tokens_to_code(tokens: list[str]) -> str:
    """Write word tokens back out as code the parser can read.

    The layout tokens turn into real newlines and indents, <STR> and <NUM> into
    a small string and number, and every other token is written as it is.
    """
    out, depth, line_start = [], 0, True
    for tok in tokens:
        if tok == NEWLINE:
            out.append("\n")
            line_start = True
        elif tok == INDENT:
            depth += 1
        elif tok == DEDENT:
            depth = max(depth - 1, 0)
        else:
            if line_start:
                out.append(" " * INDENT_SIZE * depth)
                line_start = False
            out.append(STAND_IN.get(tok, tok) + " ")
    return "".join(out)


class AstPredictor:
    def __init__(self, base: WordPredictor, k_candidates: int = 20):
        self.base, self.k_candidates = base, k_candidates
        self.name = f"ast-{base.n}"

    def keeps_valid(self, tokens: list[str], candidate: str) -> bool:
        return repair(tokens_to_code(tokens + [candidate])) is not None

    def predict_tokens(self, tokens: list[str], k: int = 5) -> list[str]:
        candidates = self.base.predict_tokens(tokens, max(k, self.k_candidates))
        # if the code so far can't be repaired, every guess would look bad
        # for the same reason, so there is nothing to rerank with
        if repair(tokens_to_code(tokens)) is None:
            return candidates[:k]
        good = [c for c in candidates if self.keeps_valid(tokens, c)]
        bad = [c for c in candidates if c not in good]
        return (good + bad)[:k]

    def predict(self, code: str, k: int = 5) -> list[str]:
        return self.predict_tokens(word_tokens(code), k)
