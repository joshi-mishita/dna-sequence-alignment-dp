"""Scoring scheme used by both alignment algorithms.

All three values are *signed* scores that are added to the running total:
    match     > 0   (reward, default +2)
    mismatch  <= 0  (penalty, default -1)
    gap       < 0   (penalty, default -2)  -- linear gap model: every gap position costs `gap`.
"""
from __future__ import annotations

from dataclasses import dataclass

GAP_CHAR = "-"


@dataclass(frozen=True)
class ScoringScheme:
    match: int = 2
    mismatch: int = -1
    gap: int = -2

    def substitution(self, a: str, b: str) -> int:
        """Score for aligning nucleotide `a` against nucleotide `b` (no gaps)."""
        return self.match if a == b else self.mismatch

    def column_score(self, a: str, b: str) -> int:
        """Score of one column of an alignment (either character may be a gap)."""
        if a == GAP_CHAR and b == GAP_CHAR:
            raise ValueError("A column cannot contain two gaps.")
        if a == GAP_CHAR or b == GAP_CHAR:
            return self.gap
        return self.substitution(a, b)

    def alignment_score(self, aligned1: str, aligned2: str) -> int:
        """Re-score a finished alignment column by column (used to verify the DP)."""
        if len(aligned1) != len(aligned2):
            raise ValueError("Aligned sequences must have equal length.")
        return sum(self.column_score(a, b) for a, b in zip(aligned1, aligned2))

    def as_dict(self) -> dict:
        return {"match": self.match, "mismatch": self.mismatch, "gap": self.gap}


DEFAULT_SCORING = ScoringScheme()
