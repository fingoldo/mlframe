"""One-vs-rest targets for the warp operators on a multiclass target.

The warp of the rank-scaled target ``E[rank(y) | x]`` orders the classes by their label codes. When that order is not monotone in ``x``, a mixture of classes at the two ends of ``x`` averages
to the rank of a class that sits elsewhere, and the warp merges what the raw column separates. The one-vs-rest warp ``E[1{y=k} | x]`` of each class has no order to respect.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MULTICLASS_MODES", "validate_multiclass_mode", "ovr_indicators"]

MULTICLASS_MODES = ("rank", "ovr")


def validate_multiclass_mode(mode: str) -> str:
    """``mode`` itself when it names a multiclass warp mode, ``ValueError`` otherwise."""
    if mode not in MULTICLASS_MODES:
        raise ValueError(f"multiclass must be one of {MULTICLASS_MODES}; got {mode!r}")
    return mode


def ovr_indicators(codes: np.ndarray, ky: int) -> list[tuple[int, np.ndarray]]:
    """``(class, 0/1 indicator)`` of every class of the dense codes ``codes`` in ``[0, ky)``; empty unless the target has more than two classes."""
    if ky <= 2:
        return []
    return [(k, (codes == k).astype(np.float64)) for k in range(ky)]  # Intentional float64: 0/1 indicator column fed to the float64 MI path
