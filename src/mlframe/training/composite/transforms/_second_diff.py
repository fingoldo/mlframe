"""``second_diff`` composite transform.

``T = y - 2*b1 + b2``, inverse ``y = T_hat + 2*b1 - b2``, where ``b1`` is the
causal lag-1 anchor (``base_prev``) and ``b2`` the lag-2 anchor (``base_prev2``).

Motivation. For a strongly-trending / doubly-integrated (I(2)) series the FIRST
difference ``y - b1`` still trends (it is itself an I(1) random walk), so a
``diff``-reconstruction of ``y`` degenerates to a persistence forecast whose
error grows with the accumulated drift. The SECOND difference
``y - 2*b1 + b2`` cancels both the level AND the linear-drift term, leaving the
stationary innovation, which a downstream model can actually predict -- the
inverse then reconstructs ``y`` from the two real per-row lags with only the
innovation residual left over. The inverse is PURE ADDITIVE and in-range by
construction (the two lag columns are real observed rows, no learned scale), so
any ``T_hat`` extrapolation maps back to ``y`` by a single bounded combination,
keeping the composite MLP-friendly like ``diff`` / ``additive_residual``.

Base contract. ``base`` is the ``linear_residual_multi``-style ``(n, K)`` matrix
(wire it via ``base_column`` = lag-1 + ``extra_base_columns`` = [lag-2]):
column 0 is the lag-1 anchor ``b1`` (``y`` shifted by one causal step) and column
1 is the lag-2 anchor ``b2`` (shifted by two). Only the first two columns are
consulted; extra columns are ignored. A 1-D base (or a single-column matrix)
carries no lag-2 term, so the transform degenerates to ``T = y - 2*b1`` with the
exact additive inverse ``y = T_hat + 2*b1`` (still a valid, if weaker, detrend).
There are NO fitted parameters -- ``fit`` returns ``{}`` -- so the transform is
free of any train/predict scale drift.

cProfile (see ``_benchmarks/bench_second_diff.py``). fit / forward / inverse are
each a single fused AXPY over the ``(n, 2)`` base at the representative shape
(n=200k); there is no reduction, sort, or solve to accelerate. An ``@njit``
rewrite of the ``y - 2*b1 + b2`` combination was considered but rejected: numpy's
C loop already saturates memory bandwidth on the three-array read and numba adds
only JIT + dispatch overhead with no arithmetic to fuse. No actionable speedup;
the numpy path is the default and at the vectorised floor.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _second_diff_bases(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the base carrier into (b1 = lag-1, b2 = lag-2) float64 columns.

    A 1-D base / single-column matrix has no lag-2 term, so ``b2`` is all-zero
    (the transform degenerates to ``y - 2*b1`` with the same additive inverse).
    """
    base_f = np.asarray(base, dtype=np.float64)
    if base_f.ndim == 1:
        return base_f, np.zeros_like(base_f)
    b1 = base_f[:, 0]
    b2 = base_f[:, 1] if base_f.shape[1] >= 2 else np.zeros_like(b1)
    return b1, b2


def _second_diff_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """No fitted parameters: the two lag columns fully define the algebra."""
    return {}


def _second_diff_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Second-difference target transform: ``T = y - 2*b1 + b2``, cancelling both level and linear-drift so the residual is stationary."""
    b1, b2 = _second_diff_bases(base)
    return np.asarray(np.asarray(y, dtype=np.float64) - 2.0 * b1 + b2)


def _second_diff_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Additive inverse of the second-difference transform: ``y = T_hat + 2*b1 - b2``; bounded by construction since ``b1``/``b2`` are real observed lags."""
    b1, b2 = _second_diff_bases(base)
    return np.asarray(np.asarray(t_hat, dtype=np.float64) + 2.0 * b1 - b2)


def _second_diff_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Row validity mask: True where the two consulted lag columns (and ``y``, if given) are finite; a NaN in an ignored extra base column never drops an otherwise-valid row."""
    base_f = np.asarray(base, dtype=np.float64)
    if base_f.ndim == 1:
        base_ok = np.isfinite(base_f)
    else:
        # Only the two consulted lag columns must be finite; a NaN in an
        # ignored extra column must not drop an otherwise-valid row.
        cols = base_f[:, : min(2, base_f.shape[1])]
        base_ok = np.all(np.isfinite(cols), axis=1)
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64)))
