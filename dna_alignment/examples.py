"""Predefined examples and the random DNA generator."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .validation import VALID_BASES


@dataclass(frozen=True)
class Example:
    key: str
    title: str
    seq1: str
    seq2: str
    description: str


EXAMPLES: list[Example] = [
    Example("identical", "Identical sequences", "AGTCAG", "AGTCAG",
            "Every column is a match, so the traceback runs straight down the diagonal "
            "and identity is 100%."),
    Example("substitution", "One substitution", "AGTCAG", "AGGCAG",
            "The 3rd base differs (T vs G). A single mismatch (-1) is cheaper than two gaps "
            "(-4), so the bases stay aligned."),
    Example("insertion", "An insertion", "AGTCA", "AGTTCA",
            "Sequence 2 has an extra T. The algorithm places one gap in sequence 1 to keep "
            "every other base matched."),
    Example("deletion", "A deletion", "AGTCA", "AGCA",
            "The T is missing from sequence 2 (the brief's example). One gap in sequence 2 "
            "gives A G T C A / A G - C A."),
    Example("multiple", "Multiple differences", "GATTACAGT", "GCATGCAT",
            "A mix of substitutions and an indel. Watch how the DP trades off mismatches "
            "against gaps."),
    Example("ties", "Several optimal alignments", "ATTA", "ATA",
            "One T must be removed, but either T can be the gapped one. Both alignments "
            "score the same, so the matrix has tied pointers."),
]

EXAMPLES_BY_KEY = {e.key: e for e in EXAMPLES}

_BASES = "".join(sorted(VALID_BASES))  # "ACGT"


def random_dna(length: int, rng: random.Random | None = None) -> str:
    """Uniformly random DNA sequence of the given length (only A, C, G, T)."""
    if length < 0:
        raise ValueError("Length must be non-negative.")
    rng = rng or random.Random()
    return "".join(rng.choice(_BASES) for _ in range(length))
