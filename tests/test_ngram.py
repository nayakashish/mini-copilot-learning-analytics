import pickle

import pytest

from copilot.ngram import UNK, NGramModel

DOCS = [
    "for i in range ( n ) :".split(),
    "for x in range ( len ( a ) ) :".split(),
    "for i in range ( len ( b ) ) :".split(),
    "print ( len ( a ) )".split(),
]


def model(**kw):
    return NGramModel(**{"max_n": 4, "k": 0.1, "min_count": 1, **kw}).fit(DOCS)


def test_seen_context_predicts_what_followed():
    m = model()
    assert m.top_k(["in", "range", "("], n=4, k=1)[0][0] == "len"


def test_probabilities_add_up_to_one():
    m = model()
    for ctx in (["range", "("], ["never", "seen"], []):
        total = sum(m.prob(tok, ctx, n=3) for tok in m.vocab)
        assert total == pytest.approx(1.0)


def test_unseen_stuff_is_never_zero():
    m = model()
    assert m.prob("print", ["range", "("], n=3) > 0
    assert m.prob("totally_new_name", ["range", "("], n=3) > 0


def test_unseen_context_backs_off():
    m = model()
    # "zzz (" was never seen, so it should back off to just "("
    assert m.top_k(["zzz", "("], n=3, k=1) == m.top_k(["("], n=2, k=1)
    # nothing seen at all falls back to plain frequencies
    assert m.top_k(["zzz", "yyy"], n=3, k=1)[0][0] == "("


def test_n_limits_the_context():
    m = model()
    assert m.top_k(["in", "range", "("], n=1, k=1) == m.top_k([], n=1, k=1)


def test_top_k_fills_up_from_shorter_contexts():
    m = model()
    # only "n" and "len" ever followed "range (", the rest comes from shorter contexts
    preds = [tok for tok, _ in m.top_k(["range", "("], n=3, k=4)]
    assert preds[:2] == ["len", "n"]
    assert len(preds) == 4 and len(set(preds)) == 4


def test_rare_tokens_become_unk_and_are_not_suggested():
    m = model(min_count=2)
    assert "n" not in m.vocab and "b" not in m.vocab and UNK in m.vocab
    for ctx in (["range", "("], ["len", "("], []):
        assert UNK not in [tok for tok, _ in m.top_k(ctx, n=3, k=10)]


def test_bad_n():
    with pytest.raises(ValueError):
        model().top_k([], n=5)


def test_pickles():
    m = model()
    again = pickle.loads(pickle.dumps(m))
    assert again.top_k(["range", "("], n=3) == m.top_k(["range", "("], n=3)
