"""DNA sequence alignment with dynamic programming (Needleman-Wunsch & Smith-Waterman).

The package has no UI code, so every function can be tested on its own.
"""
from .alignment import (GLOBAL, LOCAL, AlignmentResult, align, count_optimal_paths,
                        enumerate_global_alignments, needleman_wunsch, smith_waterman)
from .examples import EXAMPLES, EXAMPLES_BY_KEY, random_dna
from .scoring import DEFAULT_SCORING, ScoringScheme
from .statistics import compute_stats, match_line
from .validation import clean_sequence, validate_scoring, validate_sequence

__all__ = [
    "GLOBAL", "LOCAL", "AlignmentResult", "align", "needleman_wunsch", "smith_waterman",
    "count_optimal_paths", "enumerate_global_alignments", "EXAMPLES", "EXAMPLES_BY_KEY",
    "random_dna", "ScoringScheme", "DEFAULT_SCORING", "compute_stats", "match_line",
    "clean_sequence", "validate_scoring", "validate_sequence",
]
