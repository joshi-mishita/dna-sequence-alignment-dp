# DNA Sequence Alignment Using Dynamic Programming

An interactive learning tool that finds the optimal alignment of two DNA sequences with the
**Needleman–Wunsch** algorithm (global alignment) and **Smith–Waterman** (local alignment), and
shows every step of the dynamic-programming computation.

Project Based Learning — *26B12MA312 Optimization Methods and Applications*, Module 3: Dynamic Programming.

## Features

| # | Feature | Where |
|---|---------|-------|
| 1 | Sequence input: upper-cases input, strips spaces, rejects anything but A/T/G/C, shows lengths, Clear button | both apps |
| 2 | Custom scoring (match / mismatch / gap) with validation and warnings | both apps |
| 3 | DP matrix with base labels, indices, stored pointers (↖ ↑ ←), traceback path and legend | both apps |
| 4 | Step-by-step fill: Start, Previous, Next, Play/Pause, Reset, speed; each step shows the cell, both bases, the diagonal/up/left candidates, the maximum and why | both apps |
| 5 | Traceback walkthrough with the alignment growing column by column | both apps |
| 6 | Colour-coded alignment (match / mismatch / gap), scrolls sideways for long sequences | both apps |
| 7 | Statistics: score, matches, mismatches, gap positions, alignment length, identity | both apps |
| 8 | Six one-click examples, each with an explanation | both apps |
| 9 | Random DNA generator (0–500 bases, one or both sequences) | both apps |
| 10 | Global vs local comparison (two independent algorithms) | both apps |
| 11 | Counts every co-optimal alignment and lists up to 12 | both apps |
| 12 | Exports: alignment `.txt`, statistics `.txt`/`.csv`, DP matrix `.csv`, full report `.txt` | Streamlit downloads; browser copy blocks |
| 13 | “About the algorithm” section with a worked example built from your current input | both apps |

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py        # opens http://localhost:8501
python -m pytest -q         # runs the full test suite (181 tests)
```

**Browser version (no install):** open `web/index.html` in any browser. It is built from
`web/template.html` + `web/dna_align.js`; after editing either, run `python web/build.py`.

## Project structure

```
dna-sequence-alignment-dp/
├── app.py                     Streamlit interface
├── dna_alignment/             Algorithm package (no UI code, fully testable)
│   ├── validation.py          clean + validate sequences and scoring
│   ├── scoring.py             scoring scheme, re-scores finished alignments
│   ├── alignment.py           Needleman–Wunsch, Smith–Waterman, traceback, tie counting
│   ├── statistics.py          matches / mismatches / gaps / identity
│   ├── visualization.py       HTML for the DP matrix and alignment
│   ├── export.py              text, CSV and report exports
│   └── examples.py            built-in examples, random DNA
├── web/
│   ├── dna_align.js           JavaScript port of alignment.py
│   ├── template.html          browser UI
│   ├── build.py               inlines the JS → index.html
│   └── index.html             self-contained browser version
├── tests/
│   ├── test_alignment.py      hand-checked cases + brute-force comparison
│   ├── test_validation_stats_export.py
│   ├── test_app.py            headless clicks through every Streamlit control
│   └── test_web_parity.py     JS port vs Python on 155 inputs (needs Node.js)
├── .streamlit/config.toml     light theme
└── requirements.txt
```

## The algorithm

Sequence 1 = rows (index *i*), sequence 2 = columns (index *j*). `F[i][j]` is the best score for
aligning the first *i* bases of sequence 1 with the first *j* bases of sequence 2.

```
F[0][0] = 0,  F[i][0] = i·gap,  F[0][j] = j·gap
F[i][j] = max( F[i-1][j-1] + s(Xi, Yj),   ↖ align Xi with Yj
               F[i-1][j]   + gap,         ↑ Xi with a gap
               F[i][j-1]   + gap )        ← a gap with Yj
s(a, b) = match if a = b else mismatch      (defaults: +2, −1, gap −2)
```

* **Traceback** starts at `F[m][n]` and follows stored pointers to `F[0][0]`.
* **Ties:** every tied direction is stored. The displayed alignment uses the rule
  diagonal > up > left; the number of optimal alignments is counted with a second DP over the pointers.
* **Smith–Waterman** adds a fourth option `0`, starts traceback at the highest cell (first in row
  order on a tie) and stops at the first `0`.
* **Complexity:** time O(mn), space O(mn); traceback O(m + n).

### Statistics convention

Each alignment column is exactly one of match, mismatch or gap, so
`matches + mismatches + gaps = alignment length`.
**Identity = matches ÷ alignment length × 100**, over the full aligned length (gap columns included),
not the original sequence lengths.

## Testing

`python -m pytest -q` → **181 passed**.

* Global alignment agrees with an exhaustive solver (every possible alignment) on 60 random cases:
  same score, same number of optimal alignments, same set of optimal alignments.
* Local alignment agrees with the best global score over all substring pairs on 60 random cases.
* 10 longer sequences (40–120 bases) checked against an independent memoised recursion.
* Hand-checked edge cases: identical, completely different, empty, one empty, single bases,
  insertion, deletion, ties, custom scoring, the full worked-example matrix.
* Invariants on every result: equal aligned lengths; removing gaps restores the input; re-scored
  alignment = reported score = `F[m][n]`; every traceback step follows a stored optimal pointer.
* Exports match on-screen values; every Streamlit button exercised headlessly; JS port identical to Python.

## Worked example

`AGTCA` vs `AGCA`, scoring +2 / −1 / −2:

```
          -    A    G    C    A
     -   0*   -2   -4   -6   -8
     A   -2   2*    0   -2   -4
     G   -4    0   4*    2    0
     T   -6   -2   2*    3    1
     C   -8   -4    0   4*    2
     A  -10   -6   -2    2   6*
```

Optimal alignment `AGTCA / AG-CA`, score 6 (4 × 2 + 1 × −2), identity 80%.

## Known limitations

* O(mn) memory; inputs capped at 500 bases to keep the interactive views responsive.
* Matrix and step views are drawn up to 50 × 50; larger results are still computed and exported.
* Linear gap penalty only (no affine gaps); simple match/mismatch scoring (no substitution matrix).
* Pairwise alignment only.
* The browser version cannot save files (its host blocks downloads), so it offers copy blocks instead.

## Explaining it in a viva

> We find the best way to line up two DNA sequences. Trying every alignment is impossible because
> the number grows exponentially. We model it as a dynamic program: the state is how many bases of
> each sequence have been used, and each decision adds one column — match/mismatch, or a gap in
> either sequence. Bellman's principle holds because removing the last column of an optimal
> alignment leaves an optimal alignment of shorter prefixes. So we fill an (m+1) × (n+1) table where
> each cell is the best of three neighbours plus a score, in O(mn) time, and trace the stored
> pointers back from the corner to recover the alignment. Smith–Waterman gives local alignment by
> flooring cells at 0. We verified everything with 181 tests, including brute-force comparison.

## References

1. R. Bellman, *Dynamic Programming*, Princeton Univ. Press, 1957.
2. S. B. Needleman and C. D. Wunsch, *J. Mol. Biol.*, 48(3):443–453, 1970.
3. T. F. Smith and M. S. Waterman, *J. Mol. Biol.*, 147(1):195–197, 1981.
4. H. A. Taha, *Operations Research: An Introduction*, 10th ed., Pearson, 2017.
