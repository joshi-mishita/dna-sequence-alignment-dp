"""Streamlit front end: DNA Sequence Alignment Using Dynamic Programming.

Run with:  streamlit run app.py
All computation lives in the `dna_alignment` package; this file only handles UI state.
"""
from __future__ import annotations

import streamlit as st

from dna_alignment import (EXAMPLES, EXAMPLES_BY_KEY, GLOBAL, LOCAL, ScoringScheme,
                           enumerate_global_alignments, needleman_wunsch, random_dna,
                           smith_waterman, validate_scoring, validate_sequence)
from dna_alignment.alignment import ARROWS, MOVE_NAMES
from dna_alignment.export import (alignment_text, full_report, matrix_csv, stats_csv,
                                  stats_for, stats_text)
from dna_alignment.statistics import match_line
from dna_alignment.visualization import (CSS, alignment_html, candidates_html, legend_html,
                                         matrix_html)

MAX_MATRIX_CELLS = 2601          # 51 x 51 cells drawn as an HTML table
DEFAULTS = {"match_in": 2, "mismatch_in": -1, "gap_in": -2}

st.set_page_config(page_title="DNA Alignment · Dynamic Programming", page_icon="🧬",
                   layout="wide")
st.markdown(CSS + """
<style>
.block-container{padding-top:2.2rem;max-width:1250px}
h1{font-weight:700;letter-spacing:-.02em}
.hero{color:#4b5563;font-size:1.02rem;max-width:860px}
.pill{display:inline-block;background:#e6f4f1;color:#0b5c4c;border-radius:999px;padding:2px 10px;
      font-size:12px;font-weight:600;margin-right:6px}
.small{color:#6b7280;font-size:13px}
.formula{font-family:'JetBrains Mono',ui-monospace,monospace;background:#f6f7f9;border:1px solid #e3e6ea;
         border-radius:8px;padding:10px 14px;font-size:13.5px;white-space:pre;overflow-x:auto}
div[data-testid="stMetricValue"]{font-size:1.7rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state & callbacks
def init_state():
    ss = st.session_state
    for k, v in {"seq1_in": "AGTCA", "seq2_in": "AGCA", "algo_in": GLOBAL,
                 "result": None, "fill_step": 0, "fill_playing": False,
                 "tb_step": 0, "tb_playing": False, "len1": 10, "len2": 10,
                 "example_key": EXAMPLES[3].key, "speed": 0.8, **DEFAULTS}.items():
        ss.setdefault(k, v)


@st.cache_resource(max_entries=24, show_spinner=False)
def compute(seq1: str, seq2: str, match: int, mismatch: int, gap: int):
    sc = ScoringScheme(match, mismatch, gap)
    return needleman_wunsch(seq1, seq2, sc), smith_waterman(seq1, seq2, sc)


def current_inputs():
    ss = st.session_state
    v1 = validate_sequence(ss.seq1_in, "Sequence 1")
    v2 = validate_sequence(ss.seq2_in, "Sequence 2")
    errs, warns = validate_scoring(ss.match_in, ss.mismatch_in, ss.gap_in)
    return v1, v2, errs, warns


def run_alignment():
    ss = st.session_state
    v1, v2, errs, _ = current_inputs()
    ss.fill_step = ss.tb_step = 0
    ss.fill_playing = ss.tb_playing = False
    if not (v1.ok and v2.ok) or errs:
        ss.result = {"error": True}
        return
    nw, sw = compute(v1.sequence, v2.sequence, ss.match_in, ss.mismatch_in, ss.gap_in)
    ss.seq1_in, ss.seq2_in = v1.sequence, v2.sequence  # show the cleaned upper-case text
    ss.result = {"nw": nw, "sw": sw,
                 "key": (v1.sequence, v2.sequence, ss.match_in, ss.mismatch_in, ss.gap_in)}


def load_example():
    ex = EXAMPLES_BY_KEY[st.session_state.example_key]
    st.session_state.seq1_in, st.session_state.seq2_in = ex.seq1, ex.seq2
    run_alignment()


def gen_random(which: str):
    ss = st.session_state
    if which in ("1", "both"):
        ss.seq1_in = random_dna(int(ss.len1))
    if which in ("2", "both"):
        ss.seq2_in = random_dna(int(ss.len2))
    run_alignment()


def clear_all():
    ss = st.session_state
    ss.seq1_in = ss.seq2_in = ""
    ss.result = None
    ss.fill_step = ss.tb_step = 0
    ss.fill_playing = ss.tb_playing = False


def reset_scoring():
    st.session_state.update(DEFAULTS)


def set_step(key: str, value: int):
    st.session_state[key] = value
    st.session_state["fill_playing" if key == "fill_step" else "tb_playing"] = False


init_state()
ss = st.session_state
if "booted" not in ss:          # show a worked result on first load
    ss.booted = True
    run_alignment()

# ------------------------------------------------------------------ 1. header
st.markdown('<span class="pill">Dynamic Programming</span><span class="pill">Bioinformatics</span>'
            '<span class="pill">26B12MA312 · PBL</span>', unsafe_allow_html=True)
st.title("DNA Sequence Alignment Using Dynamic Programming")
st.markdown('<p class="hero">Two DNA sequences can differ by substitutions, insertions and deletions. '
            'The <b>Needleman–Wunsch</b> algorithm finds the best possible global alignment by solving '
            'small sub-problems (aligning prefixes) once, storing them in a matrix, and building the '
            'answer from them — Bellman’s principle of optimality in action. Enter two sequences, run '
            'the alignment, then step through the matrix to see exactly how the answer is reached.</p>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 2-4. inputs
v1, v2, score_errs, score_warns = current_inputs()
left, right = st.columns([3, 2], gap="large")

with left:
    with st.container(border=True):
        st.subheader("1 · DNA sequences")
        c1, c2 = st.columns(2)
        for col, key, v, name in ((c1, "seq1_in", v1, "Sequence 1 (rows)"),
                                  (c2, "seq2_in", v2, "Sequence 2 (columns)")):
            with col:
                st.text_area(name, key=key, height=96, placeholder="e.g. AGTCA",
                             help="A, T, G, C only. Lower case is converted; spaces and line breaks are ignored.")
                if v.ok:
                    st.caption(f"✅ Valid · length **{len(v.sequence)}**"
                               + (" · empty (aligns to all gaps)" if not v.sequence else ""))
                else:
                    for e in v.errors:
                        st.error(e, icon="⚠️")
        e1, e2 = st.columns([3, 1], vertical_alignment="bottom")
        with e1:
            st.selectbox("Example", [e.key for e in EXAMPLES], key="example_key",
                         format_func=lambda k: EXAMPLES_BY_KEY[k].title)
        with e2:
            st.button("Load example", on_click=load_example, use_container_width=True)
        st.caption("💡 " + EXAMPLES_BY_KEY[ss.example_key].description)

        st.markdown("**Random DNA generator**")
        r1, r2, r3, r4, r5 = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom")
        r1.number_input("Length 1", 0, 500, key="len1", step=1)
        r2.number_input("Length 2", 0, 500, key="len2", step=1)
        r3.button("🎲 Seq 1", on_click=gen_random, args=("1",), use_container_width=True)
        r4.button("🎲 Seq 2", on_click=gen_random, args=("2",), use_container_width=True)
        r5.button("🎲 Both", on_click=gen_random, args=("both",), use_container_width=True)

with right:
    with st.container(border=True):
        st.subheader("2 · Scoring")
        s1, s2, s3 = st.columns(3)
        s1.number_input("Match", -100, 100, key="match_in", step=1)
        s2.number_input("Mismatch", -100, 100, key="mismatch_in", step=1)
        s3.number_input("Gap", -100, 100, key="gap_in", step=1)
        for e in score_errs:
            st.error(e, icon="⚠️")
        for w in score_warns:
            st.warning(w, icon="ℹ️")
        st.button("Restore defaults (+2 / −1 / −2)", on_click=reset_scoring)

    with st.container(border=True):
        st.subheader("3 · Algorithm")
        st.radio("Method", [GLOBAL, LOCAL], key="algo_in", horizontal=True,
                 format_func=lambda a: f"{a} ({'global' if a == GLOBAL else 'local'})",
                 label_visibility="collapsed")
        b1, b2 = st.columns([2, 1])
        can_run = v1.ok and v2.ok and not score_errs
        b1.button("▶ Run alignment", type="primary", on_click=run_alignment,
                  disabled=not can_run, use_container_width=True)
        b2.button("Clear", on_click=clear_all, use_container_width=True)
        if not can_run:
            st.caption("Fix the highlighted problems to enable the alignment.")

res = ss.result
if not res or res.get("error"):
    st.info("Load an example, generate random sequences or type your own, then press **Run alignment**.")
    st.stop()

nw, sw = res["nw"], res["sw"]
main = nw if ss.algo_in == GLOBAL else sw
stats = stats_for(main)
live_key = (v1.sequence, v2.sequence, ss.match_in, ss.mismatch_in, ss.gap_in)
if live_key != res["key"]:
    st.warning("Inputs changed since the last run — press **Run alignment** to recalculate. "
               "Results below are for the previous inputs.", icon="🔄")

# ------------------------------------------------------------------ 5. results
st.header(f"Results · {main.algorithm}")
m_ = st.columns(6)
m_[0].metric("Optimal score", stats.score)
m_[1].metric("Matches", stats.matches)
m_[2].metric("Mismatches", stats.mismatches)
m_[3].metric("Gap positions", stats.gaps)
m_[4].metric("Alignment length", stats.length)
m_[5].metric("Identity", f"{stats.identity:.2f}%")
st.caption("Identity = matches ÷ alignment length × 100 (gap columns count toward the length). "
           f"Score re-computed column by column from the alignment: **{stats.recomputed_score}**"
           + (" ✓ equals the bottom-right cell F[m][n]." if not main.is_local else
              " ✓ equals the highest matrix cell."))
st.markdown(alignment_html(main.aligned1, main.aligned2), unsafe_allow_html=True)
if main.is_local and main.aligned1:
    (a, b), (c, d) = main.region1, main.region2
    st.caption(f"Local region: sequence 1 positions {a}–{b}, sequence 2 positions {c}–{d}.")
if not main.is_local and main.optimal_count and main.optimal_count > 1:
    with st.expander(f"This input has {main.optimal_count} equally optimal alignments — show them"):
        st.caption("Ties in the matrix (cells with more than one arrow) create alternative paths. "
                   "The main result uses the rule diagonal > up > left at every tie.")
        for k, (a1, a2) in enumerate(enumerate_global_alignments(main, 12), 1):
            st.code(f"#{k}\n{a1}\n{match_line(a1, a2)}\n{a2}", language=None)
        if main.optimal_count > 12:
            st.caption(f"Showing 12 of {main.optimal_count}.")

m, n = len(main.seq1), len(main.seq2)
too_big = (m + 1) * (n + 1) > MAX_MATRIX_CELLS
tabs = st.tabs(["DP matrix", "Step-by-step fill", "Traceback", "Global vs local",
                "Export", "About the algorithm"])

# ------------------------------------------------------------------ 6. matrix
with tabs[0]:
    st.markdown(f"**{m + 1} × {n + 1} matrix** — rows: sequence 1, columns: sequence 2. "
                "Hover a cell to see its coordinates.")
    if too_big:
        st.info(f"The matrix has {(m + 1) * (n + 1):,} cells, too many to draw clearly. "
                "Use sequences up to 50 bases to view it, or download it as CSV from the Export tab.")
    else:
        st.markdown(matrix_html(main, show_max=main.is_local), unsafe_allow_html=True)
        st.markdown(legend_html("result"), unsafe_allow_html=True)
        if main.is_local:
            st.caption("Dashed outline = cell(s) holding the maximum score, where the local traceback starts.")


# ------------------------------------------------------------------ 7. step-by-step fill
def fill_walkthrough(r):
    K = len(r.fill_steps)
    if K == 0:
        st.info("One sequence is empty, so there are no inner cells: the matrix is just the "
                "initialised first row/column.")
        st.markdown(matrix_html(r, filled_upto=0, show_path=False), unsafe_allow_html=True)
        return

    @st.fragment(run_every=ss.speed if ss.fill_playing else None)
    def view():
        if ss.fill_playing:
            if ss.fill_step < K:
                ss.fill_step += 1
            else:
                ss.fill_playing = False
                st.rerun()
        c = st.columns([1, 1, 1, 1.2, 1, 2.2], vertical_alignment="bottom")
        c[0].button("⏮ Start", on_click=set_step, args=("fill_step", 1), use_container_width=True,
                    help="Jump to the first inner cell F[1][1]")
        c[1].button("◀ Prev", on_click=set_step, args=("fill_step", max(0, ss.fill_step - 1)),
                    use_container_width=True, disabled=ss.fill_step == 0)
        c[2].button("Next ▶", on_click=set_step, args=("fill_step", min(K, ss.fill_step + 1)),
                    use_container_width=True, disabled=ss.fill_step == K)
        if c[3].button("⏸ Pause" if ss.fill_playing else "⏵ Play", use_container_width=True,
                       type="primary"):
            ss.fill_playing = not ss.fill_playing
            if ss.fill_playing and ss.fill_step >= K:
                ss.fill_step = 0
            st.rerun()
        c[4].button("↺ Reset", on_click=set_step, args=("fill_step", 0), use_container_width=True)
        c[5].select_slider("Speed", options=[1.5, 0.8, 0.4, 0.15], key="speed",
                           format_func=lambda s: {1.5: "slow", 0.8: "normal", 0.4: "fast", 0.15: "very fast"}[s])
        st.slider("Step", 0, K, key="fill_step", label_visibility="collapsed")

        k = ss.fill_step
        cols = st.columns([3, 2], gap="large")
        with cols[1]:
            if k == 0:
                g = r.scoring.gap
                st.markdown(f"#### Step 0 of {K} · Initialisation")
                if r.is_local:
                    st.markdown("Smith–Waterman sets the first row and column to **0**: a local "
                                "alignment may begin anywhere, so no prefix has to be paid for.")
                else:
                    st.markdown(f"First row: F[0][j] = j × gap = j × ({g:+d}). First column: "
                                f"F[i][0] = i × ({g:+d}).\n\nAligning a prefix of length *k* "
                                "with an empty sequence needs exactly *k* gaps, so these are the "
                                "only possible scores — they are the **base cases** of the recursion.")
                st.caption("Press Next or Play to start filling the matrix row by row.")
            else:
                s = r.fill_steps[k - 1]
                st.markdown(f"#### Step {k} of {K} · cell F[{s.i}][{s.j}]")
                kind = "match ✅" if s.char1 == s.char2 else "mismatch ✖"
                st.markdown(f"Comparing **{s.char1}** (seq 1, pos {s.i}) with **{s.char2}** "
                            f"(seq 2, pos {s.j}) → {kind}, substitution score **{s.substitution:+d}**")
                st.markdown(candidates_html(s), unsafe_allow_html=True)
                chosen = ", ".join(f"{ARROWS[d]} {MOVE_NAMES[d]}" for d in s.chosen) or "reset to 0"
                st.markdown(f"**Selected maximum: {s.best}** (from {chosen})")
                st.caption(s.explanation)
        with cols[0]:
            if too_big:
                st.info("Matrix too large to draw; the calculations on the right are still exact.")
            else:
                cur = None if k == 0 else (r.fill_steps[k - 1].i, r.fill_steps[k - 1].j)
                st.markdown(matrix_html(r, filled_upto=k, current=cur, show_path=False),
                            unsafe_allow_html=True)
                st.markdown(legend_html(), unsafe_allow_html=True)

    view()


with tabs[1]:
    st.markdown(f"Watch **{main.algorithm}** fill the matrix one cell at a time, row by row. "
                "Each cell needs only its three already-computed neighbours.")
    fill_walkthrough(main)


# ------------------------------------------------------------------ 8. traceback
def traceback_walkthrough(r):
    T = len(r.traceback_steps)

    @st.fragment(run_every=ss.speed if ss.tb_playing else None)
    def view():
        if ss.tb_playing:
            if ss.tb_step < T - 1:
                ss.tb_step += 1
            else:
                ss.tb_playing = False
                st.rerun()
        c = st.columns([1, 1, 1, 1.2, 1], vertical_alignment="bottom")
        c[0].button("⏮ Start", key="tb_start", on_click=set_step, args=("tb_step", 0),
                    use_container_width=True)
        c[1].button("◀ Prev", key="tb_prev", on_click=set_step,
                    args=("tb_step", max(0, ss.tb_step - 1)), use_container_width=True,
                    disabled=ss.tb_step == 0)
        c[2].button("Next ▶", key="tb_next", on_click=set_step,
                    args=("tb_step", min(T - 1, ss.tb_step + 1)), use_container_width=True,
                    disabled=ss.tb_step >= T - 1)
        if c[3].button("⏸ Pause" if ss.tb_playing else "⏵ Play", key="tb_play",
                       use_container_width=True, type="primary"):
            ss.tb_playing = not ss.tb_playing
            if ss.tb_playing and ss.tb_step >= T - 1:
                ss.tb_step = 0
            st.rerun()
        c[4].button("↺ Reset", key="tb_reset", on_click=set_step, args=("tb_step", 0),
                    use_container_width=True)
        if T > 1:
            st.slider("Traceback step", 0, T - 1, key="tb_step", label_visibility="collapsed")

        t = r.traceback_steps[ss.tb_step]
        visited = {(x.i, x.j) for x in r.traceback_steps[:ss.tb_step]}
        cols = st.columns([3, 2], gap="large")
        with cols[1]:
            st.markdown(f"#### Traceback step {ss.tb_step + 1} of {T}")
            st.markdown(f"Current cell **F[{t.i}][{t.j}] = {t.cell_score}**")
            if t.move:
                st.markdown(f"Move: **{ARROWS[t.move]} {MOVE_NAMES[t.move]}** → adds column "
                            f"`{t.col1}` / `{t.col2}`")
                if len(t.options) > 1:
                    st.info("Tie: " + ", ".join(MOVE_NAMES[o] for o in t.options)
                            + " all give the optimum. Any of them yields an optimal alignment; "
                            "we follow diagonal > up > left.", icon="⚖️")
            st.caption(t.explanation)
            st.markdown("**Alignment built so far** (grows from the right end):")
            st.markdown(alignment_html(t.partial1, t.partial2), unsafe_allow_html=True)
        with cols[0]:
            if too_big:
                st.info("Matrix too large to draw; the traceback text on the right is still exact.")
            else:
                st.markdown(matrix_html(r, current=(t.i, t.j), path_cells=visited,
                                        pred_filter=set(t.options) if t.move else set(),
                                        show_max=r.is_local), unsafe_allow_html=True)
                st.markdown(legend_html(), unsafe_allow_html=True)
                st.caption("Coloured predecessor = the pointer(s) stored in the current cell. "
                           "Green cells = path already walked.")

    view()


with tabs[2]:
    start = "the bottom-right cell F[m][n]" if not main.is_local else "the highest-scoring cell"
    stop = "F[0][0]" if not main.is_local else "the first cell containing 0"
    st.markdown(f"Traceback starts at **{start}** and follows the stored arrows back to **{stop}**. "
                "Each diagonal move aligns two bases; an up move puts a gap in sequence 2; a left "
                "move puts a gap in sequence 1.")
    traceback_walkthrough(main)

# ------------------------------------------------------------------ 9. comparison
with tabs[3]:
    st.markdown("**Global** alignment (Needleman–Wunsch) must use *every* base of both sequences. "
                "**Local** alignment (Smith–Waterman) looks for the most similar *region*: cells "
                "never drop below 0, so a bad stretch can be abandoned and a new alignment started. "
                "Both use the same scoring and are computed independently here.")
    ca, cb = st.columns(2, gap="large")
    for col, r in ((ca, nw), (cb, sw)):
        with col, st.container(border=True):
            s_ = stats_for(r)
            st.subheader(f"{r.algorithm}  ·  {'global' if not r.is_local else 'local'}")
            k1, k2, k3 = st.columns(3)
            k1.metric("Score", r.score)
            k2.metric("Identity", f"{s_.identity:.1f}%")
            k3.metric("Length", s_.length)
            st.markdown(alignment_html(r.aligned1, r.aligned2), unsafe_allow_html=True)
            if r.is_local:
                if r.aligned1:
                    (a, b), (c, d) = r.region1, r.region2
                    st.caption(f"Region: seq 1 [{a}–{b}] of {len(r.seq1)}, seq 2 [{c}–{d}] of "
                               f"{len(r.seq2)}. Ends at F[{r.end[0]}][{r.end[1]}] = {r.score}.")
                else:
                    st.caption("No positively scoring region exists, so the local alignment is empty.")
            else:
                st.caption(f"Covers all {len(r.seq1)} + {len(r.seq2)} bases. "
                           f"Score = F[{len(r.seq1)}][{len(r.seq2)}].")
            if not too_big:
                with st.expander("Show matrix"):
                    st.markdown(matrix_html(r, show_max=r.is_local), unsafe_allow_html=True)
    st.caption("The local score is always ≥ the global score, because the full-length global "
               "alignment is one of the candidates local alignment could pick (and 0 is always allowed).")

# ------------------------------------------------------------------ 10. export
with tabs[4]:
    st.markdown(f"Downloads reflect the **{main.algorithm}** result shown above — the same "
                "functions generate both the screen values and the files.")
    tag = "global" if not main.is_local else "local"
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button("Alignment (.txt)", alignment_text(main), f"alignment_{tag}.txt",
                       "text/plain", use_container_width=True)
    d2.download_button("Statistics (.txt)", stats_text(main), f"statistics_{tag}.txt",
                       "text/plain", use_container_width=True)
    d3.download_button("Statistics (.csv)", stats_csv(main), f"statistics_{tag}.csv",
                       "text/csv", use_container_width=True)
    d4.download_button("DP matrix (.csv)", matrix_csv(main), f"dp_matrix_{tag}.csv",
                       "text/csv", use_container_width=True)
    other = sw if main is nw else nw
    d5.download_button("Full report (.txt)", full_report(main, other), f"report_{tag}.txt",
                       "text/plain", use_container_width=True, type="primary")
    with st.expander("Preview full report"):
        st.code(full_report(main, other), language=None)

# ------------------------------------------------------------------ 11. about
with tabs[5]:
    g, mt, mm = nw.scoring.gap, nw.scoring.match, nw.scoring.mismatch
    st.markdown("### What is sequence alignment?")
    st.markdown("Aligning two DNA sequences means writing them one above the other, inserting gaps "
                "(“-”) so that similar bases line up. Each column is a **match**, a **mismatch** "
                "(substitution) or a **gap** (insertion/deletion). A scoring scheme rewards matches "
                "and penalises the rest; the optimal alignment is the one with the highest total score.")
    st.markdown("### Why dynamic programming?")
    st.markdown("Two sequences of length 100 have more than 10⁷⁵ possible alignments, so trying them "
                "all is impossible. DP works because the problem has:\n"
                "- **Optimal substructure** — the best alignment of `X[1..i]` and `Y[1..j]` must end in "
                "one of three columns (X_i/Y_j, X_i/−, −/Y_j), and what precedes that column must itself "
                "be an optimal alignment of the shorter prefixes. This is Bellman’s principle of optimality.\n"
                "- **Overlapping sub-problems** — the same prefix pair (i, j) is needed by three "
                "different larger problems, so each is solved once and stored in the matrix.")
    st.markdown("### Recurrence")
    st.markdown(f'<div class="formula">F[0][0] = 0,   F[i][0] = i·gap,   F[0][j] = j·gap\n'
                f'F[i][j] = max( F[i-1][j-1] + s(Xᵢ, Yⱼ),   ← diagonal: align Xᵢ with Yⱼ\n'
                f'               F[i-1][j]   + gap,         ← up: Xᵢ against a gap\n'
                f'               F[i][j-1]   + gap )        ← left: a gap against Yⱼ\n'
                f's(a,b) = {mt:+d} if a = b, {mm:+d} otherwise;   gap = {g:+d}\n'
                f'Smith–Waterman: same, plus a 4th option 0; answer = max cell, not F[m][n].</div>',
                unsafe_allow_html=True)
    st.markdown("### Why cumulative gap penalties in the first row and column?")
    st.markdown("F[i][0] is the score of aligning the first *i* bases of sequence 1 with nothing. "
                "The only way is *i* gaps, so F[i][0] = i × gap. These base cases are what force a "
                "*global* alignment to account for every base.")
    st.markdown("### Traceback and multiple optimal alignments")
    st.markdown("While filling, each cell remembers which neighbour(s) gave its maximum. Starting at "
                "F[m][n] and following these pointers back to F[0][0] reconstructs the alignment in "
                "reverse. When two or three candidates tie, each tied pointer leads to a different but "
                "equally optimal alignment; this app counts them all and shows up to 12.")
    st.markdown("### Complexity")
    st.markdown("There are (m+1)(n+1) cells and each takes O(1) work (three additions and a max), so "
                "**time is O(mn)**. Storing the score and pointer matrices needs **O(mn) space**. "
                "Traceback adds O(m + n). (Hirschberg’s algorithm can reduce space to O(m + n); it is "
                "outside the scope of this project.)")

    st.markdown("### Worked example from your current input")
    if nw.fill_steps and not too_big:
        f1, fl = nw.fill_steps[0], nw.fill_steps[-1]
        st.markdown(f"Sequences **{nw.seq1}** and **{nw.seq2}**, scoring {mt:+d}/{mm:+d}/{g:+d}.")
        st.markdown(f"1. **Initialise**: first row 0, {g}, {2 * g}, … and first column the same.\n"
                    f"2. **First cell** F[1][1] ({f1.char1} vs {f1.char2}): "
                    f"max(diag {f1.diag}, up {f1.up}, left {f1.left}) = **{f1.best}**.\n"
                    f"3. **Last cell** F[{fl.i}][{fl.j}] ({fl.char1} vs {fl.char2}): "
                    f"max(diag {fl.diag}, up {fl.up}, left {fl.left}) = **{fl.best}** — "
                    f"this is the optimal global score.\n"
                    f"4. **Traceback** takes {len(nw.traceback_steps) - 1} moves and gives:")
        st.code(f"{nw.aligned1}\n{match_line(nw.aligned1, nw.aligned2)}\n{nw.aligned2}", language=None)
        st.markdown(matrix_html(nw), unsafe_allow_html=True)
    else:
        st.caption("Run a non-empty alignment of up to 50 bases to see a worked example here.")

st.divider()
st.caption("DNA Sequence Alignment Using Dynamic Programming · Needleman–Wunsch & Smith–Waterman · "
           "Course project for Optimization Methods and Applications (Dynamic Programming module).")
