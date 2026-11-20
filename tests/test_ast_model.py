from copilot.ast_model import AstPredictor, tokens_to_code
from copilot.predictors import get_predictor, save
from copilot.tokenizers import word_tokens
from test_predictors import word_model


class FakeBase:
    """A word model that always says the same thing, so I control the order."""
    n = 4

    def __init__(self, guesses):
        self.guesses = guesses

    def predict_tokens(self, tokens, k=5):
        return self.guesses[:k]


def test_tokens_to_code_layout():
    code = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    out = tokens_to_code(word_tokens(code))
    compile(out, "x", "exec")  # it is real code again
    assert out.count("\n") == 4


def test_tokens_to_code_placeholders():
    out = tokens_to_code(word_tokens("x = 'abc' + 5"))
    assert out.strip() == 'x = "s" + 0'


def test_breaking_guess_goes_below_valid_one():
    # the n-gram likes "def" most, but `foo(a def` can never be valid
    p = AstPredictor(FakeBase(["def", ",", "+"]))
    assert p.predict("foo(a", k=3) == [",", "+", "def"]


def test_order_inside_a_group_stays():
    p = AstPredictor(FakeBase(["+", "-", "def", "class"]))
    assert p.predict("x = foo(a", k=4) == ["+", "-", "def", "class"]
    assert AstPredictor(FakeBase(["class", "def", "-", "+"])).predict("x = foo(a", k=4) == ["-", "+", "class", "def"]


def test_unfixable_context_keeps_the_original_order():
    p = AstPredictor(FakeBase(["def", ",", "+"]))
    assert p.predict("foo(a, b def", k=3) == ["def", ",", "+"]


def test_closing_bracket_with_nothing_open_is_demoted():
    p = AstPredictor(FakeBase([")", "x"]))
    assert p.predict("y = 1\nz = ", k=2) == ["x", ")"]


def test_same_interface_as_the_others():
    from copilot.predictors import WordPredictor
    p = AstPredictor(WordPredictor(word_model(), 3))
    out = p.predict("for i in range(", k=3)
    assert isinstance(out, list) and len(out) <= 3
    assert p.predict_tokens(word_tokens("for i in range("), k=3) == out


def test_get_predictor_ast(tmp_path):
    save(word_model(), tmp_path / "word.pkl")
    p = get_predictor("ast", 3, models_dir=tmp_path)
    assert p.name == "ast-3"
