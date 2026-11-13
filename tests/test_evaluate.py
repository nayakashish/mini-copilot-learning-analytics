import pytest

from copilot.evaluate import LAYOUT, CutPoint, pick_cut_points, project_of, run

FILES = {
    "data/raw/proj_a/x.py": ["def", "f", "(", ")", ":", "<NEWLINE>", "<INDENT>", "return", "<NUM>", "<NEWLINE>"],
    "data/raw/proj_a/y.py": ["x", "=", "<NUM>", "<NEWLINE>"],
    "data/raw/proj_b/z.py": ["import", "os", "<NEWLINE>"] * 20,
}


def test_project_of():
    assert project_of("data/raw/flask/src/flask/app.py") == "flask"


def test_same_number_per_project():
    cuts = pick_cut_points(FILES, per_project=5, seed=1)
    assert sum(c.project == "proj_a" for c in cuts) == 5
    assert sum(c.project == "proj_b" for c in cuts) == 5


def test_takes_everything_when_a_project_is_small():
    cuts = pick_cut_points(FILES, per_project=1000, seed=1)
    # proj_a has 9 + 3 spots after the first word, minus the layout tokens
    real_a = [t for toks in list(FILES.values())[:2] for t in toks[1:] if t not in LAYOUT]
    assert sum(c.project == "proj_a" for c in cuts) == len(real_a)


def test_words_skips_layout_but_all_keeps_it():
    words = pick_cut_points(FILES, per_project=1000, seed=1, targets="words")
    every = pick_cut_points(FILES, per_project=1000, seed=1, targets="all")
    assert all(FILES[c.path][c.index] not in LAYOUT for c in words)
    assert any(FILES[c.path][c.index] in LAYOUT for c in every)


def test_never_cuts_at_the_very_start():
    assert all(c.index > 0 for c in pick_cut_points(FILES, per_project=1000, seed=1, targets="all"))


def test_same_seed_same_spots():
    assert pick_cut_points(FILES, 5, seed=7) == pick_cut_points(FILES, 5, seed=7)
    assert pick_cut_points(FILES, 5, seed=7) != pick_cut_points(FILES, 5, seed=8)


def test_bad_targets():
    with pytest.raises(ValueError):
        pick_cut_points(FILES, 5, seed=1, targets="nope")


class Echo:
    """Fake model that always suggests the last word it saw."""
    def predict_tokens(self, tokens, k=5):
        return [tokens[-1]]


def test_run_scores_against_the_real_next_word():
    files = {"data/raw/p/a.py": ["a", "a", "b"]}
    cuts = [CutPoint("p", "data/raw/p/a.py", 1), CutPoint("p", "data/raw/p/a.py", 2)]
    scores = run(Echo(), files, cuts)
    assert [s.top1 for s in scores] == [True, False]


class Fixed:
    def __init__(self, words):
        self.words = words

    def predict_tokens(self, tokens, k=5):
        return self.words[:k]


def test_run_drops_layout_suggestions():
    files = {"data/raw/p/a.py": ["x", "y"]}
    cuts = [CutPoint("p", "data/raw/p/a.py", 1)]
    model = Fixed(["<NEWLINE>", "y", "<INDENT>", "a", "b", "c", "d", "e"])
    assert not run(model, files, cuts)[0].top1
    s = run(model, files, cuts, drop=frozenset(LAYOUT))[0]
    assert s.top1 and s.top5
