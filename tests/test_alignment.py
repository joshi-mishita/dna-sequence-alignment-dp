"""Correctness tests for Needleman-Wunsch and Smith-Waterman.

Besides hand-checked cases, the algorithms are compared against an independent
brute-force solver that enumerates every possible alignment of short sequences.
"""
import itertools
import random
from functools import lru_cache

import pytest

from dna_alignment import (EXAMPLES, ScoringScheme, compute_stats, count_optimal_paths,
                           enumerate_global_alignments, needleman_wunsch, random_dna,
                           smith_waterman)
from dna_alignment.alignment import DIAG, LEFT, UP

DEFAULT = ScoringScheme()


# ------------------------------------------------------------------ brute force
def all_alignments(s1, s2):
    """Every global alignment of s1 and s2 (exponential: short inputs only)."""
    if not s1 and not s2:
        return [("", "")]
    out = []
    if s1 and s2:
        out += [(s1[0] + a, s2[0] + b) for a, b in all_alignments(s1[1:], s2[1:])]
    if s1:
        out += [(s1[0] + a, "-" + b) for a, b in all_alignments(s1[1:], s2)]
    if s2:
        out += [("-" + a, s2[0] + b) for a, b in all_alignments(s1, s2[1:])]
    return out


def brute_global(s1, s2, sc):
    scored = [(sc.alignment_score(a, b), a, b) for a, b in all_alignments(s1, s2)]
    best = max(x[0] for x in scored)
    return best, {(a, b) for s, a, b in scored if s == best}


def brute_local_score(s1, s2, sc):
    best = 0
    for i, k in itertools.combinations_with_replacement(range(len(s1) + 1), 2):
        for j, l in itertools.combinations_with_replacement(range(len(s2) + 1), 2):
            best = max(best, recursive_global(s1[i:k], s2[j:l], sc))
    return best


def recursive_global(s1, s2, sc):
    @lru_cache(None)
    def f(i, j):
        if i == 0:
            return j * sc.gap
        if j == 0:
            return i * sc.gap
        return max(f(i - 1, j - 1) + sc.substitution(s1[i - 1], s2[j - 1]),
                   f(i - 1, j) + sc.gap, f(i, j - 1) + sc.gap)
    return f(len(s1), len(s2))


def assert_valid_alignment(r):
    """Invariants every reconstructed alignment must satisfy."""
    assert len(r.aligned1) == len(r.aligned2)
    assert "--" not in {a + b for a, b in zip(r.aligned1, r.aligned2)}
    assert r.scoring.alignment_score(r.aligned1, r.aligned2) == r.score
    if not r.is_local:
        assert r.aligned1.replace("-", "") == r.seq1
        assert r.aligned2.replace("-", "") == r.seq2
        assert r.score == r.matrix[-1][-1]
        assert r.path[0] == (0, 0) and r.path[-1] == (len(r.seq1), len(r.seq2))
    else:
        (si, sj), (ei, ej) = r.start, r.end
        assert r.aligned1.replace("-", "") == r.seq1[si:ei]
        assert r.aligned2.replace("-", "") == r.seq2[sj:ej]
    # each step of the path follows a recorded optimal pointer
    for (pi, pj), (i, j) in zip(r.path, r.path[1:]):
        move = {(1, 1): DIAG, (1, 0): UP, (0, 1): LEFT}[(i - pi, j - pj)]
        assert move in r.pointers[i][j]
        g = r.scoring.gap
        expected = {DIAG: r.matrix[pi][pj] + (r.scoring.substitution(r.seq1[i - 1], r.seq2[j - 1])
                                              if i and j else 0),
                    UP: r.matrix[pi][pj] + g, LEFT: r.matrix[pi][pj] + g}[move]
        assert r.matrix[i][j] == expected


# ------------------------------------------------------------------ hand-checked cases
def test_brief_example_deletion():
    r = needleman_wunsch("AGTCA", "AGCA")
    assert (r.aligned1, r.aligned2) == ("AGTCA", "AG-CA")
    assert r.score == 6  # 4 matches * 2 + 1 gap * -2
    assert_valid_alignment(r)


def test_matrix_values_hand_computed():
    r = needleman_wunsch("AGTCA", "AGCA")
    assert r.matrix == [
        [0, -2, -4, -6, -8],
        [-2, 2, 0, -2, -4],
        [-4, 0, 4, 2, 0],
        [-6, -2, 2, 3, 1],
        [-8, -4, 0, 4, 2],
        [-10, -6, -2, 2, 6],
    ]


def test_identical():
    r = needleman_wunsch("ACGTACGT", "ACGTACGT")
    assert r.score == 16 and r.aligned1 == r.aligned2 == "ACGTACGT"
    assert r.optimal_count == 1
    st = compute_stats(r.aligned1, r.aligned2, r.scoring, r.score)
    assert st.identity == 100.0 and st.gaps == 0


def test_completely_different():
    r = needleman_wunsch("AAAA", "TTTT")
    assert r.score == -4  # four mismatches beat any gapped alternative
    assert (r.aligned1, r.aligned2) == ("AAAA", "TTTT")
    assert compute_stats(r.aligned1, r.aligned2, r.scoring, r.score).identity == 0.0


def test_both_empty():
    r = needleman_wunsch("", "")
    assert r.score == 0 and r.aligned1 == r.aligned2 == ""
    assert r.matrix == [[0]] and r.optimal_count == 1
    assert compute_stats("", "", DEFAULT, 0).identity == 0.0


@pytest.mark.parametrize("s1,s2", [("ACG", ""), ("", "ACG")])
def test_one_empty(s1, s2):
    r = needleman_wunsch(s1, s2)
    assert r.score == -6
    assert r.aligned1.replace("-", "") == s1 and r.aligned2.replace("-", "") == s2
    assert set(r.aligned1 if not s1 else r.aligned2) == {"-"}
    assert_valid_alignment(r)


@pytest.mark.parametrize("a,b,score", [("A", "A", 2), ("A", "G", -1)])
def test_single_character(a, b, score):
    r = needleman_wunsch(a, b)
    assert r.score == score and (r.aligned1, r.aligned2) == (a, b)


def test_single_character_gap_preferred_when_mismatch_is_expensive():
    sc = ScoringScheme(match=1, mismatch=-10, gap=-1)
    r = needleman_wunsch("A", "G", sc)
    assert r.score == -2 and len(r.aligned1) == 2  # A-/-G beats the -10 mismatch
    assert r.optimal_count == 2
    assert_valid_alignment(r)


def test_insertion():
    r = needleman_wunsch("AGTCA", "AGTTCA")
    assert r.score == 8 and r.aligned1.count("-") == 1 and r.aligned2.count("-") == 0
    assert_valid_alignment(r)


def test_multiple_optimal_alignments():
    r = needleman_wunsch("ATTA", "ATA")
    assert r.optimal_count == 2
    found = set(enumerate_global_alignments(r))
    assert found == {("ATTA", "A-TA"), ("ATTA", "AT-A")}
    # tie rule DIAG > UP > LEFT picks a deterministic one of them
    assert (r.aligned1, r.aligned2) in found


def test_custom_scoring_changes_result():
    s1, s2 = "GATTACA", "GCATGCU".replace("U", "T")
    a = needleman_wunsch(s1, s2, ScoringScheme(1, -1, -1))
    b = needleman_wunsch(s1, s2, ScoringScheme(5, -4, -10))
    assert a.score == brute_global(s1, s2, a.scoring)[0]
    assert b.score == brute_global(s1, s2, b.scoring)[0]
    assert b.aligned1.count("-") <= a.aligned1.count("-")  # expensive gaps -> fewer gaps


def test_fill_steps_match_matrix_and_explanations():
    r = needleman_wunsch("GATTACA", "GCATG")
    assert len(r.fill_steps) == 7 * 5
    for s in r.fill_steps:
        assert r.matrix[s.i][s.j] == s.best == max(s.diag, s.up, s.left)
        assert s.diag == r.matrix[s.i - 1][s.j - 1] + s.substitution
        assert s.up == r.matrix[s.i - 1][s.j] + r.scoring.gap
        assert s.left == r.matrix[s.i][s.j - 1] + r.scoring.gap
        assert set(s.chosen) == {d for d, v in (("D", s.diag), ("U", s.up), ("L", s.left))
                                 if v == s.best}
        assert f"The maximum is {s.best}" in s.explanation


def test_traceback_steps_rebuild_alignment():
    r = needleman_wunsch("GATTACA", "GCATGCT")
    last = r.traceback_steps[-1]
    assert last.move is None and (last.i, last.j) == (0, 0)
    assert (last.partial1, last.partial2) == (r.aligned1, r.aligned2)
    assert len(r.traceback_steps) == len(r.aligned1) + 1


# ------------------------------------------------------------------ brute-force agreement
SCHEMES = [ScoringScheme(), ScoringScheme(1, -1, -1), ScoringScheme(1, -3, -2),
           ScoringScheme(3, 0, -1), ScoringScheme(2, -1, 0)]


@pytest.mark.parametrize("seed", range(60))
def test_global_matches_brute_force(seed):
    rng = random.Random(seed)
    s1, s2 = random_dna(rng.randint(0, 5), rng), random_dna(rng.randint(0, 5), rng)
    sc = SCHEMES[seed % len(SCHEMES)]
    r = needleman_wunsch(s1, s2, sc)
    best, optimal = brute_global(s1, s2, sc)
    assert r.score == best
    assert r.optimal_count == len(optimal)
    assert set(enumerate_global_alignments(r, limit=10_000)) == optimal
    assert (r.aligned1, r.aligned2) in optimal
    assert_valid_alignment(r)


@pytest.mark.parametrize("seed", range(60))
def test_local_matches_brute_force(seed):
    rng = random.Random(1000 + seed)
    s1, s2 = random_dna(rng.randint(0, 7), rng), random_dna(rng.randint(0, 7), rng)
    sc = SCHEMES[seed % 4]  # skip the gap=0 scheme (local alignment needs real penalties)
    r = smith_waterman(s1, s2, sc)
    assert r.score == brute_local_score(s1, s2, sc)
    assert r.score == max(max(row) for row in r.matrix)
    assert min(min(row) for row in r.matrix) == 0
    assert_valid_alignment(r)


@pytest.mark.parametrize("seed", range(10))
def test_longer_random_sequences(seed):
    rng = random.Random(seed)
    s1, s2 = random_dna(rng.randint(40, 120), rng), random_dna(rng.randint(40, 120), rng)
    r = needleman_wunsch(s1, s2)
    assert r.score == recursive_global(s1, s2, r.scoring)
    assert_valid_alignment(r)
    assert_valid_alignment(smith_waterman(s1, s2))


# ------------------------------------------------------------------ Smith-Waterman specifics
def test_local_finds_shared_core():
    r = smith_waterman("TTTTGATTACATTTT", "CCGATTACACC")
    assert (r.aligned1, r.aligned2) == ("GATTACA", "GATTACA")
    assert r.score == 14 and r.region1 == (5, 11) and r.region2 == (3, 9)


def test_local_no_similarity():
    r = smith_waterman("AAAA", "TTTT")
    assert r.score == 0 and r.aligned1 == "" and r.region1 == (0, 0)


def test_local_score_never_below_global():
    for e in EXAMPLES:
        assert smith_waterman(e.seq1, e.seq2).score >= needleman_wunsch(e.seq1, e.seq2).score


def test_examples_behave_as_described():
    by = {e.key: needleman_wunsch(e.seq1, e.seq2) for e in EXAMPLES}
    assert by["identical"].aligned1 == by["identical"].aligned2
    st = compute_stats(by["substitution"].aligned1, by["substitution"].aligned2, DEFAULT,
                       by["substitution"].score)
    assert st.mismatches == 1 and st.gaps == 0
    assert by["insertion"].aligned1.count("-") == 1 and "-" not in by["insertion"].aligned2
    assert (by["deletion"].aligned1, by["deletion"].aligned2) == ("AGTCA", "AG-CA")
    assert by["ties"].optimal_count > 1
    assert count_optimal_paths(by["ties"].pointers, 4, 3) == 2
