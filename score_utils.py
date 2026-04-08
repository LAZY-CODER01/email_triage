"""
Score normalization helpers.

The competition validator rejects task scores of exactly 0.0 or 1.0, so every
score we emit must remain strictly within the open interval (0, 1), including
when serialized to two decimal places.
"""

from __future__ import annotations

import math

MIN_OPEN_SCORE = 0.01
MAX_OPEN_SCORE = 0.99


def clamp_open_score(score: float) -> float:
    """Clamp any numeric score to the open interval (0, 1)."""
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        return MIN_OPEN_SCORE

    normalized = round(float(score), 4)
    if normalized <= 0.0:
        return MIN_OPEN_SCORE
    if normalized >= 1.0:
        return MAX_OPEN_SCORE

    return max(MIN_OPEN_SCORE, min(MAX_OPEN_SCORE, normalized))
