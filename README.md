# Mini-Copilot (COSC 329 Learning Analytics project)

Next-token code suggestion for Python using letter n-grams, word n-grams
(n = 1..10) and an AST-based model. The models are evaluated with accuracy
and edit distance, and the best one is demoed in a small text editor.

## Requirements

- Python 3.10
- git 2.27 or newer (for sparse checkout)

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

