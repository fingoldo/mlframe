"""Scale-free guarded division for feature builders.

An additive ``den + 1e-12`` pad is harmless at unit scale but dominates the denominator once it is itself ~1e-12 (squared tiny amplitudes, tiny variances),
silently shrinking the ratio. ``safe_div`` divides exactly and returns 0.0 only where the denominator is exactly zero.
"""

from __future__ import annotations

import numpy as np


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den`` with 0.0 where ``den == 0``; NaN in either operand propagates."""
    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast(num_arr, den_arr).shape, dtype=np.float64)
    np.divide(num_arr, den_arr, out=out, where=den_arr != 0.0)
    return out


def safe_rel_err(abs_err, ref_abs) -> np.ndarray:
    """Elementwise ``abs_err / ref_abs`` for a relative-error diagnostic.

    Exact where ``ref_abs != 0``. Where the reference is exactly zero the error is 0.0 if the error is zero and ``+inf`` otherwise, so a
    mismatch against a zero reference is reported, never hidden by a pad. NaN in either operand propagates.
    """
    err = np.asarray(abs_err, dtype=np.float64)
    ref = np.asarray(ref_abs, dtype=np.float64)
    shape = np.broadcast(err, ref).shape
    out = np.divide(err, ref, out=np.zeros(shape, dtype=np.float64), where=ref != 0.0)
    zero_ref = np.broadcast_to(ref == 0.0, shape)
    mismatch = np.broadcast_to(err != 0.0, shape)
    out = np.where(zero_ref & mismatch, np.where(np.isnan(err), np.nan, np.inf), out)
    return out
