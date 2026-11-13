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

Tests: `python3 -m pytest`

## Evaluation

```sh
python3 scripts/evaluate.py
```

For each of the 4 test projects I pick 1,000 random spots (seed 329). At each spot the model sees the code before it and
guesses the next word. The main score only uses spots where the next word is a real word, and newline/indent tokens are taken out of the guesses since the editor won't suggest them. A second score includes them, just out of interest.

Results go to `results/metrics/ngram.csv` and, split up per test project, `results/metrics/ngram_by_project.csv`. Takes about 5 minutes.
