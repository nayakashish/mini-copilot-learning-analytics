"""Turning Python source into tokens for the models.

Word tokens come from Python's own tokenizer. Comments are dropped, strings
become <STR> and numbers become <NUM>, and the layout tokens are kept as
<NEWLINE>, <INDENT> and <DEDENT> so the model knows where lines and blocks are.

The letter model reads the same tokens written out as text with a space
between each one, so both models see exactly the same code and a "word" for
the letter model is just everything up to the next space.
"""

import io
import tokenize

STR, NUM = "<STR>", "<NUM>"
NEWLINE, INDENT, DEDENT = "<NEWLINE>", "<INDENT>", "<DEDENT>"

SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
LAYOUT = {tokenize.NEWLINE: NEWLINE, tokenize.INDENT: INDENT, tokenize.DEDENT: DEDENT}


def word_tokens(src: str) -> list[str]:
    """Word tokens for a file, or for code cut off halfway (which is what the editor sees)."""
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in SKIP:
                continue
            if t.type == tokenize.NEWLINE and not t.string:
                continue  # tokenize adds this when the code doesn't end in a newline
            if t.type == tokenize.STRING:
                out.append(STR)
            elif t.type == tokenize.NUMBER:
                out.append(NUM)
            elif t.type in LAYOUT:
                out.append(LAYOUT[t.type])
            elif t.type == tokenize.ERRORTOKEN and t.string.isspace():
                continue
            elif t.type == tokenize.ERRORTOKEN and t.string in {'"', "'"}:
                out.append(STR)  # a string I'm still typing
                break
            else:
                out.append(t.string)
    except (tokenize.TokenError, IndentationError):
        pass  # cut-off code (open bracket, half a string...), keep what I got so far

    # the DEDENTs at the very end are tokenize closing every open block at end
    # of input, which isn't something anyone typed, so they go
    while out and out[-1] == DEDENT:
        out.pop()
    return out


def tokens_to_text(tokens: list[str]) -> str:
    """The text the letter model trains on: the tokens with a space after each one."""
    return "".join(t + " " for t in tokens)


def char_tokens(src: str) -> list[str]:
    return list(tokens_to_text(word_tokens(src)))
