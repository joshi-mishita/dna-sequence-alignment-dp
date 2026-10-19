"""HTML builders for the DP matrix and alignment views.

They return plain HTML strings (no Streamlit import), so the visuals can be unit-tested
and reused by any front end. Styling lives in `CSS`.
"""
from __future__ import annotations

from html import escape

from .alignment import ARROWS, DIAG, LEFT, UP, AlignmentResult
from .statistics import GAP, MATCH, MISMATCH, classify_columns

CSS = """
<style>
.dp-wrap{display:inline-block;max-width:100%;overflow:auto;max-height:640px;border:1px solid #e3e6ea;border-radius:10px;background:#fff}
table.dp{border-collapse:separate;border-spacing:0;font-family:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace;font-size:13px;margin:0}
table.dp th{position:sticky;background:#f6f7f9;color:#4b5563;font-weight:600;padding:6px 8px;text-align:center;border-bottom:1px solid #e3e6ea;z-index:2}
table.dp thead th{top:0}
table.dp tbody th{left:0;border-right:1px solid #e3e6ea;border-bottom:none;z-index:1}
table.dp thead th.corner{left:0;z-index:3}
table.dp th small{display:block;font-weight:400;color:#9aa1ab;font-size:10px}
table.dp td{min-width:44px;height:38px;text-align:center;padding:2px 4px;border:1px solid #f0f1f3;position:relative;color:#1f2933;transition:background .2s}
table.dp td .ar{position:absolute;top:1px;left:3px;font-size:10px;color:#9aa1ab;letter-spacing:-1px}
table.dp td.empty{color:#c7ccd3}
table.dp td.path{background:#d6f5ee;font-weight:700;color:#0b5c4c;box-shadow:inset 0 0 0 2px #14a38b}
table.dp td.tbdone{background:#d6f5ee;font-weight:700;color:#0b5c4c}
table.dp td.cur{background:#fff3b0;font-weight:700;box-shadow:inset 0 0 0 3px #d4a106}
table.dp td.pd{background:#e3edff;box-shadow:inset 0 0 0 2px #2f6fdb}
table.dp td.pu{background:#efe8ff;box-shadow:inset 0 0 0 2px #7a4fd8}
table.dp td.pl{background:#ffe6f1;box-shadow:inset 0 0 0 2px #cc3d7f}
table.dp td.maxcell{outline:2px dashed #d4a106;outline-offset:-4px}
table.dp td .tag{position:absolute;bottom:1px;right:3px;font-size:9px;font-weight:700}
td.pd .tag{color:#2f6fdb} td.pu .tag{color:#7a4fd8} td.pl .tag{color:#cc3d7f}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:#4b5563;margin:10px 0 4px}
.legend span.sw{display:inline-block;width:16px;height:16px;border-radius:4px;vertical-align:-3px;margin-right:6px}
.aln-wrap{overflow-x:auto;padding:14px 16px;background:#fff;border:1px solid #e3e6ea;border-radius:10px}
.aln{font-family:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace;font-size:16px;line-height:1.2;white-space:nowrap}
.aln .row{display:flex;align-items:center}
.aln .lab{width:92px;flex:none;color:#6b7280;font-size:12px;font-family:Inter,system-ui,sans-serif}
.aln .c{display:inline-block;width:24px;text-align:center;flex:none;border-radius:4px;margin:1px 1px}
.aln .c.m{background:#ddf3e6;color:#11643d;font-weight:600}
.aln .c.x{background:#fde8cc;color:#8a4a00;font-weight:600}
.aln .c.g{background:#eef0f3;color:#8a929c}
.aln .c.s{color:#4b5563;font-weight:700}
.aln .ruler .c{font-size:10px;color:#9aa1ab}
.cand{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0}
.cand div{flex:1 1 110px;border:1px solid #e3e6ea;border-radius:10px;padding:10px 12px;background:#fff}
.cand div.win{border:2px solid #14a38b;background:#effbf8}
.cand b{font-size:22px;display:block;font-family:'JetBrains Mono',ui-monospace,monospace}
.cand small{color:#6b7280}
</style>
"""

PRED_CLASS = {DIAG: "pd", UP: "pu", LEFT: "pl"}
PRED_TAG = {DIAG: "D", UP: "U", LEFT: "L"}


def legend_html(mode: str = "full") -> str:
    items = [("#fff3b0", "#d4a106", "Current cell"),
             ("#e3edff", "#2f6fdb", "Diagonal predecessor (D)"),
             ("#efe8ff", "#7a4fd8", "Upper predecessor (U)"),
             ("#ffe6f1", "#cc3d7f", "Left predecessor (L)"),
             ("#d6f5ee", "#14a38b", "Optimal traceback path")]
    if mode == "result":
        items = items[4:]
    parts = [f'<span><span class="sw" style="background:{bg};box-shadow:inset 0 0 0 2px {bd}"></span>{escape(t)}</span>'
             for bg, bd, t in items]
    parts.append('<span>Arrows ↖ ↑ ← = stored traceback pointers (several arrows = tie)</span>')
    parts.append('<span>Row/column “-” = empty prefix (gap)</span>')
    return '<div class="legend">' + "".join(parts) + "</div>"


def matrix_html(result: AlignmentResult, *, filled_upto: int | None = None,
                current: tuple[int, int] | None = None, show_path: bool = True,
                path_cells: set[tuple[int, int]] | None = None,
                show_arrows: bool = True, show_max: bool = False,
                pred_filter: set[str] | None = None) -> str:
    """Render the DP matrix.

    filled_upto: number of fill steps already executed (None = everything). Cells not
      yet computed are shown as '·'. Row 0 and column 0 always count as initialised.
    current: cell being computed (fill) or visited (traceback); its predecessors are coloured.
    path_cells: cells to mark as traceback path (defaults to the full path when show_path).
    """
    m, n = len(result.seq1), len(result.seq2)
    F, P = result.matrix, result.pointers
    total = m * n
    k = total if filled_upto is None else filled_upto

    def filled(i, j):
        return i == 0 or j == 0 or (i - 1) * n + j <= k

    preds: dict[tuple[int, int], str] = {}
    if current and current != (0, 0):
        ci, cj = current
        if ci > 0 and cj > 0:
            preds = {(ci - 1, cj - 1): DIAG, (ci - 1, cj): UP, (ci, cj - 1): LEFT}
        elif ci > 0:
            preds = {(ci - 1, cj): UP}
        else:
            preds = {(ci, cj - 1): LEFT}
    if pred_filter is not None:
        preds = {c: d for c, d in preds.items() if d in pred_filter}

    if path_cells is None:
        path_cells = set(result.path) if show_path else set()
    max_cells = set(result.max_cells) if show_max else set()

    head = ['<th class="corner"></th>', '<th>-<small>j=0</small></th>']
    head += [f"<th>{escape(c)}<small>j={j}</small></th>" for j, c in enumerate(result.seq2, 1)]
    rows = []
    labels = ["-"] + list(result.seq1)
    for i in range(m + 1):
        cells = [f"<th>{escape(labels[i])}<small>i={i}</small></th>"]
        for j in range(n + 1):
            cls = []
            if not filled(i, j):
                cells.append('<td class="empty">·</td>')
                continue
            tag = ""
            if (i, j) == current:
                cls.append("cur")
            elif (i, j) in preds:
                cls.append(PRED_CLASS[preds[(i, j)]])
                tag = f'<span class="tag">{PRED_TAG[preds[(i, j)]]}</span>'
            elif (i, j) in path_cells:
                cls.append("path")
            if (i, j) in max_cells:
                cls.append("maxcell")
            arrows = ""
            if show_arrows and P[i][j]:
                arrows = '<span class="ar">' + "".join(ARROWS[d] for d in P[i][j]) + "</span>"
            title = f"F[{i}][{j}] = {F[i][j]}"
            cells.append(f'<td class="{" ".join(cls)}" title="{title}">{arrows}{F[i][j]}{tag}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="dp-wrap"><table class="dp"><thead><tr>' + "".join(head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def alignment_html(aligned1: str, aligned2: str, label1: str = "Sequence 1",
                   label2: str = "Sequence 2") -> str:
    """Colour-coded, horizontally scrollable alignment (one fixed-width box per column)."""
    if not aligned1:
        return '<div class="aln-wrap"><em>Empty alignment.</em></div>'
    kinds = classify_columns(aligned1, aligned2)
    cls = {MATCH: "m", MISMATCH: "x", GAP: "g"}
    sym = {MATCH: "|", MISMATCH: "·", GAP: "&nbsp;"}
    title = {MATCH: "match", MISMATCH: "mismatch", GAP: "gap"}

    def row(label, chars):
        return (f'<div class="row"><span class="lab">{escape(label)}</span>'
                + "".join(f'<span class="c {cls[k]}" title="{title[k]}">{escape(c)}</span>'
                          for c, k in zip(chars, kinds)) + "</div>")

    mid = ('<div class="row"><span class="lab"></span>'
           + "".join(f'<span class="c s">{sym[k]}</span>' for k in kinds) + "</div>")
    ruler = ('<div class="row ruler"><span class="lab">column</span>'
             + "".join(f'<span class="c">{p if p % 5 == 0 or p == 1 else ""}</span>'
                       for p in range(1, len(kinds) + 1)) + "</div>")
    return ('<div class="aln-wrap"><div class="aln">' + ruler + row(label1, aligned1) + mid
            + row(label2, aligned2) + "</div></div>"
            + '<div class="legend"><span><span class="sw" style="background:#ddf3e6"></span>Match ( | )</span>'
            '<span><span class="sw" style="background:#fde8cc"></span>Mismatch ( · )</span>'
            '<span><span class="sw" style="background:#eef0f3"></span>Gap ( - )</span></div>')


def candidates_html(step) -> str:
    winners = set(step.chosen)

    def box(key, label, formula, value):
        w = ' class="win"' if key in winners else ""
        return f"<div{w}><small>{label}</small><b>{value}</b><small>{escape(formula)}</small></div>"

    zero = ""
    if step.zero_floor:
        zero = '<div class="win"><small>Floor (local only)</small><b>0</b><small>start a new local alignment</small></div>'
    return ('<div class="cand">'
            + box(DIAG, "↖ Diagonal", f"F[{step.i-1}][{step.j-1}] {step.substitution:+d}", step.diag)
            + box(UP, "↑ Up", f"F[{step.i-1}][{step.j}] + gap", step.up)
            + box(LEFT, "← Left", f"F[{step.i}][{step.j-1}] + gap", step.left)
            + zero + "</div>")
