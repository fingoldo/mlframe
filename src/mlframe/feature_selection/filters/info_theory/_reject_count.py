"""Shared reject-count rule of the permutation noise gate."""

from __future__ import annotations

from numba import njit

_FLOAT_SLACK = 1e-9


@njit(cache=True, nogil=True)
def reject_count(n_perm: int, confidence: float) -> int:
    """Number of permuted scores allowed to reach the observed value: ``floor(n_perm * (1 - confidence))`` without float truncation.

    ``int(10 * (1 - 0.8))`` is 1 and ``int(100 * (1 - 0.9))`` is 9 because ``1 - confidence`` is not exact in binary; the slack restores the
    intended 2 and 10. Host code, the njit kernels and the GPU gate all call this one function so their verdicts agree.
    """
    return int(n_perm * (1.0 - confidence) + _FLOAT_SLACK)
