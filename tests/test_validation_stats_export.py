import csv
import io
import random

import pytest

from dna_alignment import (ScoringScheme, compute_stats, match_line, needleman_wunsch,
                           random_dna, smith_waterman, validate_scoring, validate_sequence)
from dna_alignment.export import (alignment_text, format_alignment, full_report, matrix_csv,
                                  stats_csv, stats_for, stats_text)


# ------------------------------------------------------------------ validation
def test_lowercase_and_whitespace_are_cleaned():
    v = validate_sequence(" ag tc\na ")
    assert v.ok and v.sequence == "AGTCA"


@pytest.mark.parametrize("raw,bad", [("AGXT", "'X'"), ("ACGU", "'U'"), ("AC1G", "'1'"),
                                     ("NNN", "'N'"), ("AC-G", "'-'")])
def test_invalid_characters(raw, bad):
    v = validate_sequence(raw, "Sequence 1")
    assert not v.ok and bad in v.errors[0] and "Sequence 1" in v.errors[0]


def test_empty_allowed_by_default_but_can_be_rejected():
    assert validate_sequence("").ok
    assert not validate_sequence("", allow_empty=False).ok


def test_too_long():
    assert not validate_sequence("A" * 501).ok


def test_scoring_validation():
    assert validate_scoring(2, -1, -2) == ([], [])
    assert validate_scoring(1, 1, -2)[0]          # match must beat mismatch
    assert validate_scoring(2, -1, 3)[0]          # positive gap rejected
    assert validate_scoring(2.5, -1, -2)[0]       # non-integer rejected
    assert validate_scoring(2, -1, 0)[1]          # free gaps -> warning only
    assert validate_scoring(2, -5, -1)[1]         # cheap gaps -> warning only


# ------------------------------------------------------------------ statistics
def test_stats_convention():
    # columns: A/A match, G/C mismatch, -/G gap, T/T match, C/- gap, A/A match
    st = compute_stats("AG-TCA", "ACGT-A", ScoringScheme(), 0)
    assert (st.matches, st.mismatches, st.gaps, st.length) == (3, 1, 2, 6)
    assert (st.gaps_in_seq1, st.gaps_in_seq2) == (1, 1)
    assert st.matches + st.mismatches + st.gaps == st.length
    assert st.identity == 50.0                   # 3 / 6 aligned columns, not / original length
    assert st.recomputed_score == 3 * 2 + 1 * -1 + 2 * -2
    assert match_line("AG-TCA", "ACGT-A") == "|. | |"


def test_match_line_symbols():
    assert match_line("ACT-", "AGTA") == "|.| "


def test_stats_score_recomputed_from_alignment():
    for seed in range(20):
        rng = random.Random(seed)
        r = needleman_wunsch(random_dna(15, rng), random_dna(12, rng))
        st = stats_for(r)
        assert st.recomputed_score == st.score == r.matrix[-1][-1]


# ------------------------------------------------------------------ exports
def test_matrix_csv_matches_matrix():
    r = needleman_wunsch("GATTACA", "GCATGCT")
    rows = list(csv.reader(io.StringIO(matrix_csv(r))))
    assert rows[0] == ["", "-"] + list("GCATGCT")
    assert [row[0] for row in rows[1:]] == ["-"] + list("GATTACA")
    assert [[int(x) for x in row[1:]] for row in rows[1:]] == r.matrix


def test_stats_exports_match_displayed_stats():
    r = needleman_wunsch("GATTACAGT", "GCATGCAT")
    st = stats_for(r)
    d = dict(list(csv.reader(io.StringIO(stats_csv(r))))[1:])
    assert int(d["score"]) == st.score and int(d["matches"]) == st.matches
    assert int(d["mismatches"]) == st.mismatches and int(d["gap_positions"]) == st.gaps
    assert int(d["alignment_length"]) == st.length
    assert float(d["identity_percent"]) == st.identity
    txt = stats_text(r)
    for label, value in st.as_rows():
        assert f"{label}: {value}" in txt


def test_alignment_text_contains_alignment_and_wraps_without_shifting():
    rng = random.Random(7)
    r = needleman_wunsch(random_dna(150, rng), random_dna(140, rng))
    block = format_alignment(r.aligned1, r.aligned2, width=60)
    lines = block.splitlines()
    seq1_rows = [ln for ln in lines if ln.startswith("Seq 1")]
    seq2_rows = [ln for ln in lines if ln.startswith("Seq 2")]
    rebuilt1 = "".join(ln.split()[3] if len(ln.split()) > 3 else "" for ln in seq1_rows)
    rebuilt2 = "".join(ln.split()[3] if len(ln.split()) > 3 else "" for ln in seq2_rows)
    assert rebuilt1 == r.aligned1 and rebuilt2 == r.aligned2
    assert r.aligned1[:60] in alignment_text(r)


def test_full_report_consistent():
    r = needleman_wunsch("ATTA", "ATA")
    sw = smith_waterman("ATTA", "ATA")
    rep = full_report(r, sw)
    st = stats_for(r)
    assert f"Optimal alignment score: {st.score}" in rep
    assert f"Sequence identity: {st.identity:.2f}%" in rep
    assert "Number of optimal alignments: 2" in rep
    assert "COMPARISON WITH SMITH-WATERMAN" in rep
    assert "Highest cell" in full_report(sw)


def test_empty_exports_do_not_crash():
    r = needleman_wunsch("", "")
    assert "(empty alignment)" in alignment_text(r)
    assert matrix_csv(r).splitlines()[1] == "-,0"
    full_report(r)
    full_report(smith_waterman("AAA", "TTT"))
