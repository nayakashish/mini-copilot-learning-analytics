import pytest

from copilot.metrics import Score, edit_distance, score, summarize


@pytest.mark.parametrize("a, b, d", [
    ("", "", 0),
    ("len", "len", 0),
    ("", "len", 3),
    ("len", "", 3),
    ("setting", "settings", 1),   # one insert
    ("self", "elf", 1),           # one delete
    ("args", "argv", 1),          # one swap
    ("kitten", "sitting", 3),     # the classic example
    ("len", "self", 3),           # s->l, l->n, drop f
])
def test_edit_distance(a, b, d):
    assert edit_distance(a, b) == d
    assert edit_distance(b, a) == d


def test_exact_hit():
    s = score(["len", "self", "x"], "len")
    assert s == Score(top1=True, top5=True, distance=0, normalized=0.0)


def test_right_word_lower_in_the_list():
    s = score(["self", "len"], "len")
    assert not s.top1 and s.top5
    assert s.distance == 3  # distance only looks at the first suggestion


def test_right_word_past_top_5():
    s = score(["a", "b", "c", "d", "e", "len"], "len")
    assert not s.top5


def test_close_miss():
    s = score(["setting"], "settings")
    assert not s.top1
    assert s.distance == 1
    assert s.normalized == pytest.approx(1 / 8)


def test_no_suggestion_is_a_full_miss():
    s = score([], "settings")
    assert not s.top1 and not s.top5
    assert s.distance == 8 and s.normalized == 1.0


def test_empty_target_is_an_error():
    with pytest.raises(ValueError):
        score(["x"], "")


def test_summarize():
    scores = [score(["len"], "len"), score(["self", "len"], "len"), score([], "abcd")]
    out = summarize(scores)
    assert out["count"] == 3
    assert out["accuracy"] == pytest.approx(1 / 3)
    assert out["top5_accuracy"] == pytest.approx(2 / 3)
    assert out["edit_distance"] == pytest.approx((0 + 3 + 4) / 3)
    assert out["normalized_edit_distance"] == pytest.approx((0 + 3 / 3 + 1) / 3)


def test_summarize_nothing():
    assert summarize([])["count"] == 0
