"""Train the word and letter n-gram models on the train split and pickle them.

Each pickle has the counts for every n from 1 to 10, so this makes all 20
models in one go. They end up in models/ (gitignored).

Run from the repo root:
    python3 scripts/train.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from copilot.ngram import NGramModel  # noqa: E402
from copilot.predictors import MODELS, save  # noqa: E402
from copilot.tokenizers import tokens_to_text, word_tokens  # noqa: E402

TRAIN = Path("data/splits/train.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--k", type=float, default=0.1, help="add-k smoothing")
    parser.add_argument("--min-count", type=int, default=2, help="rarer than this -> <UNK>")
    args = parser.parse_args()

    files = TRAIN.read_text().split()
    words = [word_tokens(Path(p).read_text(encoding="utf-8", errors="replace")) for p in files]
    letters = [list(tokens_to_text(toks)) for toks in words]
    print(f"{len(files)} train files, {sum(map(len, words))} words, {sum(map(len, letters))} letters")

    for name, seqs in (("word", words), ("letter", letters)):
        start = time.time()
        model = NGramModel(args.max_n, args.k, args.min_count).fit(seqs)
        path = MODELS / f"{name}.pkl"
        save(model, path)
        contexts = sum(len(level) for level in model.counts)
        print(f"{name}: vocab {len(model.vocab)}, {contexts} contexts, "
              f"{time.time() - start:.1f}s, {path} {path.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
