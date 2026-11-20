from copilot.compare import comparison, to_markdown


def row(model, n, acc):
    return {"model": model, "n": str(n), "targets": "words", "accuracy": acc, "top5_accuracy": "0.5",
            "edit_distance": "3.0", "normalized_edit_distance": "0.9"}


NGRAM = [row("word", 3, "0.30"), row("word", 4, "0.35"), row("letter", 9, "0.28"), row("letter", 10, "0.31")]
AST = [row("ast", 4, "0.37"), row("word", 4, "0.35")]


def test_picks_best_of_each_family_and_sorts_by_accuracy():
    table = comparison(NGRAM, AST)
    assert [r["model"] for r in table] == ["ast-4", "word-4", "letter-10"]


def test_base_model_is_not_listed_twice():
    names = [r["model"] for r in comparison(NGRAM, AST)]
    assert len(names) == len(set(names))


def test_base_shown_even_when_not_the_best_word_model():
    ngram = [row("word", 4, "0.35"), row("word", 5, "0.36"), row("letter", 10, "0.31")]
    names = [r["model"] for r in comparison(ngram, AST)]
    assert "word-4" in names and "word-5" in names


def test_markdown_has_a_line_per_model():
    out = to_markdown(comparison(NGRAM, AST))
    assert out.count("\n") == 2 + 3 - 1
