"""Input validation for DNA sequences and scoring parameters."""
from __future__ import annotations

from dataclasses import dataclass, field

VALID_BASES = frozenset("ATGC")
MAX_SEQUENCE_LENGTH = 500   # keeps the O(mn) matrix small enough for an interactive UI
SCORE_LIMIT = 100           # sanity bound for user-entered scores


@dataclass
class ValidationResult:
    sequence: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def clean_sequence(raw: str | None) -> str:
    """Upper-case the input and drop whitespace (spaces, tabs, newlines from pasted text)."""
    if raw is None:
        return ""
    return "".join(raw.split()).upper()


def validate_sequence(raw: str | None, name: str = "Sequence", allow_empty: bool = True,
                      max_length: int = MAX_SEQUENCE_LENGTH) -> ValidationResult:
    """Clean a sequence and report every problem found.

    Empty sequences are valid input for Needleman-Wunsch (the alignment is all gaps),
    so they are allowed unless `allow_empty=False`.
    """
    seq = clean_sequence(raw)
    errors: list[str] = []
    if not seq and not allow_empty:
        errors.append(f"{name} is empty.")
    invalid = sorted({ch for ch in seq if ch not in VALID_BASES})
    if invalid:
        positions = [i + 1 for i, ch in enumerate(seq) if ch not in VALID_BASES][:5]
        shown = ", ".join(repr(c) for c in invalid)
        pos = ", ".join(map(str, positions))
        errors.append(f"{name} contains invalid character(s) {shown} at position(s) {pos}"
                      f"{' ...' if len(positions) == 5 else ''}. Only A, T, G and C are allowed.")
    if len(seq) > max_length:
        errors.append(f"{name} is {len(seq)} bases long; the maximum is {max_length}.")
    return ValidationResult(seq, errors)


def validate_scoring(match, mismatch, gap) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a scoring configuration."""
    errors: list[str] = []
    warnings: list[str] = []
    values = {"Match score": match, "Mismatch score": mismatch, "Gap penalty": gap}
    for label, v in values.items():
        if isinstance(v, bool) or not isinstance(v, int):
            errors.append(f"{label} must be a whole number.")
        elif abs(v) > SCORE_LIMIT:
            errors.append(f"{label} must be between -{SCORE_LIMIT} and {SCORE_LIMIT}.")
    if errors:
        return errors, warnings
    if match <= mismatch:
        errors.append("Match score must be greater than the mismatch score; otherwise "
                      "the algorithm has no reason to prefer identical bases.")
    if gap > 0:
        errors.append("Gap penalty must be zero or negative; a positive gap score would "
                      "reward inserting gaps.")
    if match <= 0:
        warnings.append("Match score is not positive, so alignment scores will never increase.")
    if mismatch > 0:
        warnings.append("Mismatch score is positive, so mismatches are rewarded.")
    if gap == 0:
        warnings.append("Gap penalty is zero, so gaps are free and alignments may contain many gaps.")
    if gap < 0 and 2 * gap > mismatch:
        warnings.append("Two gaps cost less than one mismatch, so the algorithm may prefer "
                        "an insertion + deletion pair over a substitution.")
    return errors, warnings
