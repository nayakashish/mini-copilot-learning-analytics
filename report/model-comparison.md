# Model comparison: notes for the report

From `results/metrics/comparison.csv` (made by `scripts/compare.py`) and
`results/metrics/ast*.csv`. Main score = real words only, 1,000 spots from
each of the 4 test projects (4,000 total).

| model | top-1 | top-5 | edit distance | normalized edit distance |
|---|---|---|---|---|
| ast-4 | 37.3% | 63.8% | 3.12 | 0.83 |
| word-4 | 35.4% | 59.6% | 3.16 | 0.88 |
| letter-10 | 31.9% | 56.2% | 3.07 | 0.77 |

## The AST model

It isn't a separate model. Word-4 gives 20 guesses, and I move the ones that
keep the code valid Python to the front. The code at a cut point is cut off
halfway, so I repair the ending first (close brackets and strings, add a
`pass`) and then ask `ast.parse` if it still parses.

- It's ahead of word-4 in all 4 test projects, so it isn't one project
  carrying it. Thealgorithms gained the most for top-1 (32.8% to 35.7%), then
  flask (30.1% to 32.1%).
- The gain is bigger for top-5 (+4.2 points) than top-1 (+1.9). That makes
  sense: the check pushes invalid guesses out of the top 5 and valid ones in,
  but it can't tell which valid guess is the right one.
- On 1,000 spots it changed the top guess on 60. It helped on 18, hurt on 3,
  and on 39 it changed it but both were wrong. Example of a help:
  `Union [ <STR> , str , bytes` then the real word is `,`. Word-4 said `)`,
  which can't close the bracket there since the `[` is still open, and AST
  fixed it.
- A first version had a bug: it could never repair code ending in `def` or
  `for`, so it always ranked those last. Fixing it took "hurts" from 15 to 3
  (top-1 0.369 to 0.373, top-5 0.623 to 0.638).
- The 3 spots left are all the same thing. After a keyword argument the real
  word is `help` or `path`, but my repair puts a bare name there, which Python
  doesn't allow, so AST ranks it down. It's a known limit.
- It costs time. About 0.15 seconds per suggestion (word-4 is almost
  instant), because every guess is repaired and parsed. That's fine for an
  editor.

## Which model is best

**ast-4.** It has the best top-1 and top-5, and a better edit distance than
word-4. The honest part is that the gain is small (about 2 points top-1), and
the edit distance is still behind letter-10.

**Runner-up: letter-10.** It's less accurate (31.9%), but its misses are
closest to the right word (0.77 normalized edit distance, best of the three),
since it can spell part of a word it never saw. It's also a different kind of
model (types letter by letter), so swapping it into the editor in week 12 is
a real contrast. Word-4 would be a bad runner-up for that, since the AST model
is just word-4 plus a check.
