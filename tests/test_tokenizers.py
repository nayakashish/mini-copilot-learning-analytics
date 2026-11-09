from copilot.tokenizers import (
    DEDENT, INDENT, NEWLINE, NUM, STR, char_tokens, tokens_to_text, word_tokens,
)


def test_simple_function():
    src = "def f(x):\n    return x + 1\n"
    assert word_tokens(src) == [
        "def", "f", "(", "x", ")", ":", NEWLINE, INDENT, "return", "x", "+", NUM, NEWLINE,
    ]


def test_comments_and_blank_lines_are_gone():
    assert word_tokens("x = 1  # a comment\n\n\n# another\ny = 2\n") == \
        ["x", "=", NUM, NEWLINE, "y", "=", NUM, NEWLINE]


def test_strings_and_numbers_become_placeholders():
    src = 's = f"hi {name}"\nt = """two\nlines"""\nn = 3.5e2\n'
    assert word_tokens(src) == ["s", "=", STR, NEWLINE, "t", "=", STR, NEWLINE, "n", "=", NUM, NEWLINE]


def test_blocks_keep_their_dedents_but_not_the_ones_at_the_end():
    src = "if a:\n    b()\nc()\nif d:\n    e()\n"
    toks = word_tokens(src)
    assert toks.count(INDENT) == 2
    assert toks.count(DEDENT) == 1
    assert toks[-1] == NEWLINE


def test_cut_off_code_does_not_crash():
    assert word_tokens("print(foo(a,") == ["print", "(", "foo", "(", "a", ","]
    assert word_tokens("x = [1,\n  2") == ["x", "=", "[", NUM, ",", NUM]
    assert word_tokens('msg = "half a str') == ["msg", "=", STR]


def test_cut_off_mid_line_has_no_fake_newline():
    assert word_tokens("x = y") == ["x", "=", "y"]


def test_letter_text_matches_the_words():
    toks = word_tokens("a = b(1)\n")
    assert tokens_to_text(toks) == "a = b ( <NUM> ) <NEWLINE> "
    assert char_tokens("a = b(1)\n") == list("a = b ( <NUM> ) <NEWLINE> ")
