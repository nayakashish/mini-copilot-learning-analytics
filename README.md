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
