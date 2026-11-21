# Mini-Copilot (COSC 329 Learning Analytics project)

Next-token code suggestion for Python using letter n-grams, word n-grams
(n = 1..10) and an AST-based model. The models are evaluated with accuracy
and edit distance, and the best one is demoed in a small text editor.

## Requirements

- Python 3.10
- git 2.27 or newer (for sparse checkout)

Setup:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repo root.

## Dataset

I'm using 12 Python projects: some libraries, some apps, a games repo,
some teaching code, part of the CPython standard library, and one of my own
projects (`hledger-wrapper`). They're all in [`data/sources.csv`](data/sources.csv)
with the repo URL, the commit I pinned, the license and what kind of project
it is. The `paths` column is for repos that are too big to use whole (CPython
and TheAlgorithms), so I only take some folders/files from them.

To download them into `data/raw/` (gitignored):

```sh
python3 scripts/fetch_data.py                 # every project
python3 scripts/fetch_data.py flask sherlock  # only some
```

It grabs each project at its pinned commit (no history) and only checks out
the `.py` files. If a project is already downloaded it gets skipped, so it's
fine to run again.


### Cleaning and the split

```sh
python3 scripts/make_splits.py
```

I only keep code that looks like what someone would actually be writing:
tests, docs, examples, `setup.py`, empty files, duplicates and files that
don't parse on Python 3.10 all get dropped (because I use v3.10 on my machine currently). Then I cap every project at 80k
tokens (random files, fixed seed) so one big project like glances doesn't take
over the dataset and is more balanced with the other smaller projects.

The split is by project: whole projects go to test until test is about 30%
of the tokens, so the model gets tested on code it has never seen. With seed
329 that's 8 projects in train and 4 in test (flask, python-patterns, sherlock,
thealgorithms), 70.2% / 29.8% of the tokens. `--unit file` splits by file
instead.

The file lists end up in `data/splits/train.txt` and `data/splits/test.txt`,
and the numbers per project in `data/stats.csv`.

## Models

```sh
python3 scripts/train.py
```

This trains the word and letter n-gram models on the train split and pickles
them into `models/` (gitignored, takes about 30 seconds). Each pickle has the
counts for every n from 1 to 10, so that's all 20 models at once.

- **Word tokens** come from Python's `tokenize`. Comments are dropped, strings
  and numbers become `<STR>` and `<NUM>`, and newlines/indents are kept as
  `<NEWLINE>`, `<INDENT>`, `<DEDENT>`.
- **The letter model** reads the same tokens written out as text with a space
  between each one, and it predicts the next word by spelling it out letter
  by letter until it writes a space (with a small beam search so it can give
  more than one suggestion).
- **Smoothing** is backoff + add-k. If a context was never seen in training,
  the model drops the oldest token and tries again, down to no context at all.
  At the level it lands on, every count gets k = 0.1 added so nothing ever has
  probability 0. Words seen only once in training become `<UNK>`.

Every model has the same `predict(code, k)` method that gives k suggestions,
so the evaluation and the editor can swap them around.

### The AST model

This one isn't a new model, it sits on top of word-4. Word-4 gives its 20 best
guesses, and I check each one: if I add this word, can Python still read the
code? The ones that still parse go first. The rest go after. The order inside
each group stays like word-4 had it.

The important catch is that code at a cut point is almost never valid (open bracket,
half a string, an `if x:` with nothing under it), and `ast.parse` just errors
on all of that. So `src/copilot/repair.py` fixes the ending first. I use
parso's tokenizer to see what's still open (parso doesn't give up on broken
code), close it, and add a `pass` or a dummy value where something is missing.
Then `ast.parse` can give me a yes or no. If the code can't be fixed at all,
the AST model just uses word-4's order.

Tests: `python3 -m pytest`

## Evaluation

```sh
python3 scripts/evaluate.py
```

For each of the 4 test projects I pick 1,000 random spots (seed 329). At each spot the model sees the code before it and
guesses the next word. The main score only uses spots where the next word is a real word, and newline/indent tokens are taken out of the guesses since the editor won't suggest them. A second score includes them, just out of interest.

Results go to `results/metrics/ngram.csv` and, split up per test project, `results/metrics/ngram_by_project.csv`. Takes about 5 minutes.

The AST model gets the same spots, run next to word-4 since that's what it
reorders:

```sh
python3 scripts/evaluate.py --model ast
python3 scripts/ast_examples.py
```

Results go to `results/metrics/ast.csv` and `ast_by_project.csv`. This one is
slow (about 30 minutes) because every guess gets repaired and parsed.
`ast_examples.py` counts the spots where the AST model changes the top guess
(better, worse or neither) and prints a few real ones.

## Charts

```sh
python3 scripts/plot.py
```

Makes the charts in `results/figures/`: accuracy vs n, edit distance vs n, and
accuracy vs n for each test project. If `results/metrics/ast.csv` exists, the
AST model shows up as a star at n = 4.

## Comparing the models

```sh
python3 scripts/compare.py
```

Puts the AST model, word-4, the best word n and the best letter n side by side
in `results/metrics/comparison.csv` (and prints it). It needs both evaluations
run first.

## What the files do

`src/copilot/` (each one has a test file in `tests/`):
- `dataset.py`: cleaning and the split by project
- `tokenizers.py`: code to word tokens (and to letters)
- `ngram.py`: the n-gram model with backoff and add-k
- `predictors.py`: word and letter predictors, save/load, `get_predictor`
- `repair.py`: closes open brackets and strings in cut-off code
- `ast_model.py`: the AST model (word-4 guesses reordered by whether the code parses)
- `metrics.py`: accuracy and edit distance
- `evaluate.py`: picks the cut points and scores a model on them
- `compare.py`: builds the comparison table from the csvs

`scripts/` (short, just run them from the repo root):
- `fetch_data.py`, `make_splits.py`, `train.py`, `evaluate.py`: the pipeline, in that order
- `ast_examples.py`: where the AST model helps or hurts vs word-4
- `plot.py`: the charts
- `compare.py`: the comparison table
