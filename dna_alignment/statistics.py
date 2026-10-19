"""Alignment statistics, computed only from the reconstructed alignment.

Counting convention
-------------------
* Each alignment column is exactly one of: match, mismatch, gap.
* A gap position is a column containing '-' in either sequence.
  gaps = gaps_in_seq1 + gaps_in_seq2, and matches + mismatches + gaps = alignment length.
* Identity (%) = matches / alignment length x 100   (over the full aligned length,
  gaps included; NOT divided by the original sequence lengths). Empty alignment -> 0 %.
"""
from __future__ import annotations

from dataclasses import dataclass

from .scoring import GAP_CHAR, ScoringScheme

MATCH, MISMATCH, GAP = "match", "mismatch", "gap"


def classify_columns(aligned1: str, aligned2: str) -> list[str]:
    if len(aligned1) != len(aligned2):
        raise ValueError("Aligned sequences must have equal length.")
    kinds = []
    for a, b in zip(aligned1, aligned2):
        if a == GAP_CHAR or b == GAP_CHAR:
            kinds.append(GAP)
        elif a == b:
            kinds.append(MATCH)
        else:
            kinds.append(MISMATCH)
    return kinds


def match_line(aligned1: str, aligned2: str) -> str:
    """'|' under matches, '.' under mismatches, ' ' under gaps."""
    sym = {MATCH: "|", MISMATCH: ".", GAP: " "}
    return "".join(sym[k] for k in classify_columns(aligned1, aligned2))


@dataclass(frozen=True)
class AlignmentStats:
    score: int
    recomputed_score: int
    matches: int
    mismatches: int
    gaps: int
    gaps_in_seq1: int
    gaps_in_seq2: int
    length: int
    identity: float

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Optimal alignment score", str(self.score)),
            ("Matches", str(self.matches)),
            ("Mismatches", str(self.mismatches)),
            ("Gap positions", f"{self.gaps} (seq 1: {self.gaps_in_seq1}, seq 2: {self.gaps_in_seq2})"),
            ("Alignment length", str(self.length)),
            ("Sequence identity", f"{self.identity:.2f}%"),
        ]


def compute_stats(aligned1: str, aligned2: str, scoring: ScoringScheme,
                  reported_score: int) -> AlignmentStats:
    kinds = classify_columns(aligned1, aligned2)
    matches = kinds.count(MATCH)
    mismatches = kinds.count(MISMATCH)
    gaps = kinds.count(GAP)
    length = len(kinds)
    identity = (matches / length * 100) if length else 0.0
    return AlignmentStats(
        score=reported_score,
        recomputed_score=scoring.alignment_score(aligned1, aligned2),
        matches=matches, mismatches=mismatches, gaps=gaps,
        gaps_in_seq1=aligned1.count(GAP_CHAR), gaps_in_seq2=aligned2.count(GAP_CHAR),
        length=length, identity=round(identity, 2),
    )
