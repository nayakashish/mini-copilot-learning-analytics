from copilot.ngram import NGramModel
from copilot.predictors import LetterPredictor, WordPredictor, load, save
from copilot.tokenizers import tokens_to_text, word_tokens

CODE = [
    "for i in range(len(items)):\n    print(items[i])\n",
    "for j in range(len(names)):\n    print(names[j])\n",
    "total = len(items)\nprint(total)\n",
] * 3


def word_model():
    return NGramModel(max_n=4, min_count=1).fit([word_tokens(c) for c in CODE])


def letter_model():
    return NGramModel(max_n=8, min_count=1).fit([list(tokens_to_text(word_tokens(c))) for c in CODE])


def test_word_predictor():
    p = WordPredictor(word_model(), n=3)
    assert p.predict("for i in range(")[0] == "len"
    assert p.predict("for i in range(", k=3)[:1] == ["len"]
    assert len(p.predict("x = ", k=5)) <= 5


def test_letter_predictor_spells_out_words():
    p = LetterPredictor(letter_model(), n=8)
    assert p.predict("for i in range(")[0] == "len"
    preds = p.predict("for i in ", k=3)
    assert preds[0] == "range"
    assert len(preds) == len(set(preds))
    assert all(" " not in w and w for w in preds)


def test_same_interface():
    for p in (WordPredictor(word_model(), 2), LetterPredictor(letter_model(), 4)):
        out = p.predict("total = ", k=2)
        assert isinstance(out, list) and all(isinstance(w, str) for w in out)
        assert p.predict_tokens(word_tokens("total = "), k=2) == out


def test_save_and_load(tmp_path):
    m = word_model()
    save(m, tmp_path / "m.pkl")
    assert WordPredictor(load(tmp_path / "m.pkl"), 3).predict("for i in range(") == \
        WordPredictor(m, 3).predict("for i in range(")
