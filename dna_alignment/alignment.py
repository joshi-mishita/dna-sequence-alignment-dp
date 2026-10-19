"""Needleman-Wunsch (global) and Smith-Waterman (local) alignment.

Conventions used everywhere in the project
------------------------------------------
* Sequence 1 runs down the rows (index i), sequence 2 runs across the columns (index j).
* F[i][j] is the best score for aligning seq1[:i] with seq2[:j].
* Moves into cell (i, j):
    DIAG  from (i-1, j-1): seq1[i-1] aligned with seq2[j-1]   (match or mismatch)
    UP    from (i-1, j)  : seq1[i-1] aligned with a gap         (gap in sequence 2)
    LEFT  from (i, j-1)  : a gap aligned with seq2[j-1]         (gap in sequence 1)
* Tie-breaking during traceback: DIAG > UP > LEFT. Every tied predecessor is still
  recorded in `pointers`, so the number of optimal alignments can be counted and all of
  them enumerated.

Time complexity O(mn); space complexity O(mn) (score matrix + pointer matrix).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .scoring import GAP_CHAR, ScoringScheme, DEFAULT_SCORING

DIAG, UP, LEFT = "D", "U", "L"
PREFERENCE = (DIAG, UP, LEFT)
ARROWS = {DIAG: "↖", UP: "↑", LEFT: "←"}
MOVE_NAMES = {DIAG: "diagonal", UP: "up", LEFT: "left"}

GLOBAL = "Needleman-Wunsch"
LOCAL = "Smith-Waterman"


@dataclass
class FillStep:
    """Everything needed to explain how one cell of the matrix was computed."""
    index: int
    i: int
    j: int
    char1: str
    char2: str
    substitution: int
    diag: int
    up: int
    left: int
    best: int
    chosen: tuple[str, ...]
    gap: int
    zero_floor: bool = False   # Smith-Waterman only: 0 beat every candidate

    @property
    def explanation(self) -> str:
        """Built on demand (cheap to store, so 500 x 500 matrices stay fast)."""
        return _explain_fill(self)


@dataclass
class TracebackStep:
    index: int
    i: int
    j: int
    cell_score: int
    move: str | None          # None means the traceback stops here
    options: tuple[str, ...]  # every optimal predecessor of this cell
    col1: str
    col2: str
    partial1: str             # alignment built so far (left-to-right order)
    partial2: str
    explanation: str


@dataclass
class AlignmentResult:
    algorithm: str
    seq1: str
    seq2: str
    scoring: ScoringScheme
    matrix: list[list[int]]
    pointers: list[list[tuple[str, ...]]]
    score: int
    aligned1: str
    aligned2: str
    path: list[tuple[int, int]]               # from start cell to end cell
    fill_steps: list[FillStep]
    traceback_steps: list[TracebackStep]
    start: tuple[int, int]
    end: tuple[int, int]
    optimal_count: int | None = None          # number of co-optimal alignments (global only)
    max_cells: list[tuple[int, int]] = field(default_factory=list)  # local only

    @property
    def is_local(self) -> bool:
        return self.algorithm == LOCAL

    @property
    def region1(self) -> tuple[int, int]:
        """1-based inclusive positions in seq1 covered by the alignment (0, 0 if empty)."""
        (si, _), (ei, _) = self.start, self.end
        return (si + 1, ei) if ei > si else (0, 0)

    @property
    def region2(self) -> tuple[int, int]:
        (_, sj), (_, ej) = self.start, self.end
        return (sj + 1, ej) if ej > sj else (0, 0)


# --------------------------------------------------------------------------- fill
def _fill(seq1: str, seq2: str, scoring: ScoringScheme, local: bool):
    m, n = len(seq1), len(seq2)
    g = scoring.gap
    F = [[0] * (n + 1) for _ in range(m + 1)]
    P: list[list[tuple[str, ...]]] = [[() for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialisation. Global: cumulative gap penalties (prefix aligned to nothing).
    # Local: zeros (an alignment may start anywhere).
    for i in range(1, m + 1):
        F[i][0] = 0 if local else i * g
        P[i][0] = () if local else (UP,)
    for j in range(1, n + 1):
        F[0][j] = 0 if local else j * g
        P[0][j] = () if local else (LEFT,)

    steps: list[FillStep] = []
    for i in range(1, m + 1):
        a = seq1[i - 1]
        for j in range(1, n + 1):
            b = seq2[j - 1]
            s = scoring.substitution(a, b)
            cand = {DIAG: F[i - 1][j - 1] + s, UP: F[i - 1][j] + g, LEFT: F[i][j - 1] + g}
            best = max(cand.values())
            zero_floor = False
            if local and best <= 0:
                chosen: tuple[str, ...] = ()
                zero_floor = True
                best = 0
            else:
                chosen = tuple(d for d in PREFERENCE if cand[d] == best)
            F[i][j] = best
            P[i][j] = chosen
            steps.append(FillStep(len(steps) + 1, i, j, a, b, s, cand[DIAG], cand[UP],
                                  cand[LEFT], best, chosen, g, zero_floor))
    return F, P, steps


def _explain_fill(st: "FillStep") -> str:
    i, j, a, b, s, g = st.i, st.j, st.char1, st.char2, st.substitution, st.gap
    best, chosen, zero_floor = st.best, st.chosen, st.zero_floor
    kind = "match" if a == b else "mismatch"
    parts = [
        f"Comparing {a} (seq 1, position {i}) with {b} (seq 2, position {j}): {kind}, "
        f"substitution score {s:+d}.",
        f"Diagonal = F[{i-1}][{j-1}] + ({s:+d}) = {st.diag - s} + ({s:+d}) = {st.diag}.",
        f"Up = F[{i-1}][{j}] + ({g:+d}) = {st.up - g} + ({g:+d}) = {st.up}.",
        f"Left = F[{i}][{j-1}] + ({g:+d}) = {st.left - g} + ({g:+d}) = {st.left}.",
    ]
    if zero_floor:
        parts.append(f"All candidates are ≤ 0, so Smith-Waterman resets the cell to 0 "
                     f"(a local alignment may start fresh here).")
    elif len(chosen) == 1:
        d = chosen[0]
        parts.append(f"The maximum is {best}, coming only from the {MOVE_NAMES[d]} cell "
                     f"({_move_meaning(d, a, b)}).")
    else:
        names = " and ".join(MOVE_NAMES[d] for d in chosen)
        parts.append(f"The maximum is {best}, reached by a tie between {names}; all tied "
                     f"directions are kept as valid traceback pointers.")
    return " ".join(parts)


def _move_meaning(d: str, a: str, b: str) -> str:
    if d == DIAG:
        return f"align {a} with {b}"
    if d == UP:
        return f"align {a} with a gap"
    return f"align a gap with {b}"


# ---------------------------------------------------------------------- traceback
def _traceback(seq1, seq2, F, P, end, local: bool):
    i, j = end
    cols1: list[str] = []
    cols2: list[str] = []
    path = [(i, j)]
    steps: list[TracebackStep] = []

    while True:
        options = P[i][j]
        stop = (i == 0 and j == 0) if not local else (F[i][j] == 0 or not options)
        if stop:
            reason = ("Reached the top-left cell F[0][0]; the traceback is complete."
                      if not local else
                      f"F[{i}][{j}] = {F[i][j]}, so the local alignment starts here; "
                      f"the traceback stops.")
            steps.append(TracebackStep(len(steps) + 1, i, j, F[i][j], None, options, "", "",
                                       "".join(reversed(cols1)), "".join(reversed(cols2)),
                                       reason))
            break
        move = options[0]  # PREFERENCE order: DIAG > UP > LEFT
        if move == DIAG:
            c1, c2, ni, nj = seq1[i - 1], seq2[j - 1], i - 1, j - 1
        elif move == UP:
            c1, c2, ni, nj = seq1[i - 1], GAP_CHAR, i - 1, j
        else:
            c1, c2, ni, nj = GAP_CHAR, seq2[j - 1], i, j - 1
        cols1.append(c1)
        cols2.append(c2)
        if len(options) > 1:
            tie = (f" Tied options: {', '.join(MOVE_NAMES[o] for o in options)}; "
                   f"choosing {MOVE_NAMES[move]} by the rule diagonal > up > left.")
        else:
            tie = ""
        expl = (f"At F[{i}][{j}] = {F[i][j]}, the pointer is {MOVE_NAMES[move]} "
                f"{ARROWS[move]} to F[{ni}][{nj}] = {F[ni][nj]}. Add column "
                f"{c1}/{c2}.{tie}")
        steps.append(TracebackStep(len(steps) + 1, i, j, F[i][j], move, options, c1, c2,
                                   "".join(reversed(cols1)), "".join(reversed(cols2)), expl))
        i, j = ni, nj
        path.append((i, j))

    path.reverse()
    return "".join(reversed(cols1)), "".join(reversed(cols2)), path, steps


def count_optimal_paths(P, m: int, n: int) -> int:
    """Number of distinct optimal global alignments = number of pointer paths (0,0)->(m,n)."""
    C = [[0] * (n + 1) for _ in range(m + 1)]
    C[0][0] = 1
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            total = 0
            for d in P[i][j]:
                pi, pj = (i - 1, j - 1) if d == DIAG else (i - 1, j) if d == UP else (i, j - 1)
                total += C[pi][pj]
            C[i][j] = total
    return C[m][n]


def enumerate_global_alignments(result: AlignmentResult, limit: int = 20) -> list[tuple[str, str]]:
    """List up to `limit` co-optimal global alignments by following every tied pointer."""
    seq1, seq2, P = result.seq1, result.seq2, result.pointers
    out: list[tuple[str, str]] = []

    def walk(i, j, a1, a2):
        if len(out) >= limit:
            return
        if i == 0 and j == 0:
            out.append(("".join(reversed(a1)), "".join(reversed(a2))))
            return
        for d in P[i][j]:
            if d == DIAG:
                walk(i - 1, j - 1, a1 + [seq1[i - 1]], a2 + [seq2[j - 1]])
            elif d == UP:
                walk(i - 1, j, a1 + [seq1[i - 1]], a2 + [GAP_CHAR])
            else:
                walk(i, j - 1, a1 + [GAP_CHAR], a2 + [seq2[j - 1]])

    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 4 * (len(seq1) + len(seq2)) + 100))
    try:
        walk(len(seq1), len(seq2), [], [])
    finally:
        sys.setrecursionlimit(old)
    return out


# ----------------------------------------------------------------------- public API
def needleman_wunsch(seq1: str, seq2: str,
                     scoring: ScoringScheme = DEFAULT_SCORING) -> AlignmentResult:
    """Global alignment. The optimal score is F[m][n]."""
    m, n = len(seq1), len(seq2)
    F, P, fill_steps = _fill(seq1, seq2, scoring, local=False)
    a1, a2, path, tb = _traceback(seq1, seq2, F, P, (m, n), local=False)
    return AlignmentResult(GLOBAL, seq1, seq2, scoring, F, P, F[m][n], a1, a2, path,
                           fill_steps, tb, start=(0, 0), end=(m, n),
                           optimal_count=count_optimal_paths(P, m, n))


def smith_waterman(seq1: str, seq2: str,
                   scoring: ScoringScheme = DEFAULT_SCORING) -> AlignmentResult:
    """Local alignment. Cells are floored at 0; traceback starts from the highest cell
    (first one in row-major order if several tie) and stops at the first 0."""
    m, n = len(seq1), len(seq2)
    F, P, fill_steps = _fill(seq1, seq2, scoring, local=True)
    best, end = 0, (0, 0)
    max_cells: list[tuple[int, int]] = []
    for i in range(m + 1):
        for j in range(n + 1):
            if F[i][j] > best:
                best, end, max_cells = F[i][j], (i, j), [(i, j)]
            elif F[i][j] == best and best > 0:
                max_cells.append((i, j))
    if best == 0:
        tb = [TracebackStep(1, 0, 0, 0, None, (), "", "", "", "",
                            "Every cell is 0: the sequences share no positively scoring "
                            "region, so the local alignment is empty.")]
        return AlignmentResult(LOCAL, seq1, seq2, scoring, F, P, 0, "", "", [(0, 0)],
                               fill_steps, tb, start=(0, 0), end=(0, 0), max_cells=[])
    a1, a2, path, tb = _traceback(seq1, seq2, F, P, end, local=True)
    return AlignmentResult(LOCAL, seq1, seq2, scoring, F, P, best, a1, a2, path,
                           fill_steps, tb, start=path[0], end=end, max_cells=max_cells)


def align(seq1: str, seq2: str, scoring: ScoringScheme = DEFAULT_SCORING,
          algorithm: str = GLOBAL) -> AlignmentResult:
    if algorithm == GLOBAL:
        return needleman_wunsch(seq1, seq2, scoring)
    if algorithm == LOCAL:
        return smith_waterman(seq1, seq2, scoring)
    raise ValueError(f"Unknown algorithm: {algorithm}")
