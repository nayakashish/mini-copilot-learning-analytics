from copilot.repair import open_things, parses_so_far, repair


def test_open_bracket_gets_closed():
    assert repair("foo(a, b") == "foo(a, b)"


def test_nested_brackets_close_in_order():
    assert repair("x = {'a': (1,") == "x = {'a': (1,)}"


def test_half_written_string_gets_closed():
    assert repair("s = 'hel") == "s = 'hel'"


def test_open_triple_quote_string():
    assert repair("s = '''ab\n") == "s = '''ab\n'''"


def test_block_with_no_body_gets_pass():
    assert "pass" in repair("if x:\n")
    assert "pass" in repair("class A:\n    def f(self):\n")


def test_unfinished_def_header():
    assert repair("def f(x") == "def f(x):\n    pass"


def test_dangling_operator():
    assert repair("x =") == "x = 0"


def test_import_with_no_name():
    assert repair("from a import") == "from a import x"


def test_valid_code_is_left_alone():
    assert repair("x = 1\n") == "x = 1\n"


def test_cannot_be_fixed_gives_none():
    assert repair("foo(a, b def") is None
    assert not parses_so_far("else:")


def test_open_things():
    assert open_things("f(a, [b") == ("", ["(", "["])
    assert open_things("f(a)") == ("", [])
    assert open_things("s = 'hel") == ("'", [])


def test_keywords_that_need_more_after_them():
    for code in ("def", "for i", "from a", "lambda x", "async", "x = 1\nfor"):
        assert parses_so_far(code), code
