"""Plain-text and CSV exports. The app shows exactly these values, so a download always
matches the screen (verified by tests/test_export.py)."""
from __future__ import annotations

import csv
import io
from datetime import datetime

from .alignment import AlignmentResult, ARROWS, enumerate_global_alignments
from .statistics import AlignmentStats, compute_stats, match_line


def stats_for(result: AlignmentResult) -> AlignmentStats:
    return compute_stats(result.aligned1, result.aligned2, result.scoring, result.score)


def format_alignment(aligned1: str, aligned2: str, width: int = 60,
                     start1: int = 1, start2: int = 1) -> str:
    """Block-wrapped alignment with position numbers; wrapping never shifts columns."""
    if not aligned1:
        return "(empty alignment)"
    mid = match_line(aligned1, aligned2)
    lines = []
    p1, p2 = start1, start2
    for k in range(0, len(aligned1), width):
        b1, b2, bm = aligned1[k:k + width], aligned2[k:k + width], mid[k:k + width]
        e1 = p1 + sum(c != "-" for c in b1) - 1
        e2 = p2 + sum(c != "-" for c in b2) - 1
        lines.append(f"Seq 1 {p1:>5}  {b1}  {max(e1, p1 - 1)}")
        lines.append(f"            {bm}")
        lines.append(f"Seq 2 {p2:>5}  {b2}  {max(e2, p2 - 1)}")
        lines.append("")
        p1, p2 = e1 + 1, e2 + 1
    return "\n".join(lines).rstrip() + "\n"


def alignment_text(result: AlignmentResult) -> str:
    s1, s2 = (result.region1[0] or 1), (result.region2[0] or 1)
    return (f"{result.algorithm} alignment\n"
            f"Sequence 1: {result.seq1 or '(empty)'}\n"
            f"Sequence 2: {result.seq2 or '(empty)'}\n"
            f"Legend: | match   . mismatch   (blank) gap\n\n"
            + format_alignment(result.aligned1, result.aligned2, start1=s1, start2=s2))


def stats_text(result: AlignmentResult) -> str:
    st = stats_for(result)
    sc = result.scoring
    rows = [f"{k}: {v}" for k, v in st.as_rows()]
    return "\n".join([
        f"{result.algorithm} alignment statistics",
        f"Scoring: match {sc.match:+d}, mismatch {sc.mismatch:+d}, gap {sc.gap:+d}",
        *rows,
        "Identity = matches / alignment length x 100 (gaps counted in the length).",
    ]) + "\n"


def stats_csv(result: AlignmentResult) -> str:
    st = stats_for(result)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["metric", "value"])
    for k, v in [("algorithm", result.algorithm), ("score", st.score),
                 ("matches", st.matches), ("mismatches", st.mismatches),
                 ("gap_positions", st.gaps), ("gaps_in_seq1", st.gaps_in_seq1),
                 ("gaps_in_seq2", st.gaps_in_seq2), ("alignment_length", st.length),
                 ("identity_percent", f"{st.identity:.2f}")]:
        w.writerow([k, v])
    return buf.getvalue()


def matrix_headers(result: AlignmentResult) -> tuple[list[str], list[str]]:
    """Column labels and row labels, including the leading gap row/column '-'."""
    return ["-"] + list(result.seq2), ["-"] + list(result.seq1)


def matrix_csv(result: AlignmentResult) -> str:
    cols, rows = matrix_headers(result)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([""] + cols)
    for label, row in zip(rows, result.matrix):
        w.writerow([label] + row)
    return buf.getvalue()


def matrix_text(result: AlignmentResult, with_arrows: bool = True) -> str:
    cols, rows = matrix_headers(result)
    width = max(4, max(len(str(v)) for row in result.matrix for v in row) + 2)
    on_path = set(result.path)
    out = ["      " + "".join(f"{c:>{width}}" for c in cols)]
    for i, (label, row) in enumerate(zip(rows, result.matrix)):
        cells = []
        for j, v in enumerate(row):
            mark = "*" if (i, j) in on_path else ""
            cells.append(f"{str(v) + mark:>{width}}")
        out.append(f"{label:>6}" + "".join(cells))
    if with_arrows:
        out.append("(* = cell on the traceback path)")
    return "\n".join(out) + "\n"


def full_report(result: AlignmentResult, other: AlignmentResult | None = None) -> str:
    st = stats_for(result)
    sc = result.scoring
    m, n = len(result.seq1), len(result.seq2)
    lines = [
        "DNA SEQUENCE ALIGNMENT USING DYNAMIC PROGRAMMING - ANALYSIS REPORT",
        "=" * 66,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Algorithm: {result.algorithm} ({'local' if result.is_local else 'global'} alignment)",
        "",
        "INPUT",
        f"  Sequence 1 ({m} bp): {result.seq1 or '(empty)'}",
        f"  Sequence 2 ({n} bp): {result.seq2 or '(empty)'}",
        f"  Scoring: match {sc.match:+d}, mismatch {sc.mismatch:+d}, gap {sc.gap:+d}",
        "",
        "OPTIMAL ALIGNMENT",
        alignment_text(result).split("\n\n", 1)[1].rstrip(),
        "",
        "STATISTICS",
        *[f"  {k}: {v}" for k, v in st.as_rows()],
        f"  Score re-computed column by column: {st.recomputed_score}",
        "",
    ]
    if result.is_local:
        r1, r2 = result.region1, result.region2
        lines += ["LOCAL REGION",
                  f"  Sequence 1 positions {r1[0]}-{r1[1]}, sequence 2 positions {r2[0]}-{r2[1]}",
                  f"  Highest cell F[{result.end[0]}][{result.end[1]}] = {result.score}", ""]
    else:
        lines += ["CO-OPTIMAL ALIGNMENTS",
                  f"  Number of optimal alignments: {result.optimal_count}"]
        if result.optimal_count and result.optimal_count > 1:
            for k, (a1, a2) in enumerate(enumerate_global_alignments(result, 10), 1):
                lines += [f"  #{k}: {a1}", f"      {match_line(a1, a2)}", f"      {a2}"]
        lines.append("")
    lines += ["DP MATRIX (rows = sequence 1, columns = sequence 2)",
              matrix_text(result).rstrip(), "",
              "TRACEBACK"]
    for t in result.traceback_steps:
        arrow = ARROWS.get(t.move, "■") if t.move else "■"
        lines.append(f"  {t.index:>3}. {arrow} {t.explanation}")
    if other is not None:
        so = stats_for(other)
        lines += ["", f"COMPARISON WITH {other.algorithm.upper()}",
                  f"  Score: {other.score}",
                  f"  Alignment: {other.aligned1 or '(empty)'} / {other.aligned2 or '(empty)'}",
                  f"  Identity: {so.identity:.2f}% over {so.length} columns"]
    lines += ["", "Complexity: time O(mn), space O(mn) "
              f"-> {(m + 1) * (n + 1)} cells for m={m}, n={n}."]
    return "\n".join(lines) + "\n"
