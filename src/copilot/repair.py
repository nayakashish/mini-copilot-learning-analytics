"""Fixing code that is cut off halfway so ast.parse can read it.

Code at a cut point is almost never valid Python: an open bracket, half a
string, an `if x:` with no body. ast.parse gives up on all of those. parso
doesn't (it keeps going and marks the bad bits), so I use its tokenizer to
see what is still open, close it, then try a few small endings until the
stdlib parser is happy.
"""

import ast

from parso.python import tokenize as ptokenize
from parso.utils import parse_version_string

VERSION = parse_version_string("3.10")
CLOSERS = {"(": ")", "[": "]", "{": "}"}
QUOTES = ('"""', "'''", '"', "'")


def open_things(code: str) -> tuple[str, list[str]]:
    """The string still open at the end ('' if none) and what is open inside it.

    The list holds brackets, and the quote of an f-string that isn't closed yet.
    """
    stack, string = [], ""
    for t in ptokenize.tokenize(code, version_info=VERSION):
        name = t.type.name
        if name == "FSTRING_START":
            stack.append(t.string.lstrip("rbfuRBFU"))
        elif name == "FSTRING_END" and stack and stack[-1] in QUOTES:
            stack.pop()
        elif t.string in CLOSERS:
            stack.append(t.string)
        elif t.string in CLOSERS.values() and stack and stack[-1] in CLOSERS:
            stack.pop()
        elif name == "ERRORTOKEN" and t.string.lstrip("rbfuRBFU").startswith(QUOTES):
            string = next(q for q in QUOTES if t.string.lstrip("rbfuRBFU").startswith(q))
    return string, stack


# small endings for a statement that is cut off halfway. Tried in this order
# (plainest first), so e.g. `for i` gets ` in 0` and `def` gets ` x()`
FILLS = ("", " 0", " x", " x()", "()", " in 0", " x in 0", " import x", " x import x",
         " x: 0", ": 0", " x\ndef f()", " def x()")


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def repair(code: str) -> str | None:
    """Code + the smallest ending I can find that makes it parse, or None if I can't."""
    string, stack = open_things(code)
    closers = "".join(CLOSERS.get(b, b) for b in reversed(stack))

    code = code.rstrip(" ")
    last = code.rstrip("\n").split("\n")[-1]
    pad = " " * (len(last) - len(last.lstrip()) + 4)
    base = " " * (len(pad) - 4)
    body = "\n" + pad + "pass"
    # a try with no except yet
    tried = body + "\n" + base + "except Exception:" + body
    # tried in order, so the plainest ending wins
    for fill in FILLS:
        for head in ("", ":"):
            for tail in ("", body, tried):
                attempt = code + string + fill + closers + head + tail
                if parses(attempt):
                    return attempt
    return None


def parses_so_far(code: str) -> bool:
    """True if the cut-off code can be completed into valid Python."""
    return repair(code) is not None
