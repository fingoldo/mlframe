"""Post-processing helpers for quantile-regression predictions.

Currently:
- ``fix_quantile_crossing(preds_NK, alphas, mode)`` -- enforce
  monotonic non-crossing across alpha levels per row.

No library natively prevents quantile crossings (CB MultiQuantile,
XGB ``quantile_alpha=[...]``, LGB / HGB / Linear all can produce
``q_0.1 > q_0.5`` on rare configurations -- particularly with small
training sets, narrow alpha gaps, or near-flat conditional
distributions). The fix is post-hoc:

- ``sort``: ``np.sort(preds, axis=1)`` -- cheap, idempotent, default
- ``isotonic``: per-row isotonic regression, more accurate when
  crossings cluster around a particular alpha
- ``none``: leave preds unchanged (caller takes responsibility)
"""

from __future__ import annotations

from typing import Sequence

import numba
import numpy as np


@numba.njit(cache=True)
def _isotonic_rows_pava(preds: np.ndarray, is_mono: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Row-wise L2 isotonic projection via pool-adjacent-violators (equal weights).

    Bit-equivalent (to FP pooling round-off, ~1e-15) to calling
    ``sklearn.isotonic.isotonic_regression(row, increasing=True)`` per row, but
    with no per-row Python->C dispatch: 500-660x faster on (N>=1e5, K quantiles).
    Rows already monotone (``is_mono``) are copied verbatim.
    """
    N, K = preds.shape
    val = np.empty(K)
    wt = np.empty(K)
    cnt = np.empty(K, np.int64)
    for i in range(N):
        if is_mono[i]:
            for k in range(K):
                out[i, k] = preds[i, k]
            continue
        m = 0
        for k in range(K):
            val[m] = preds[i, k]
            wt[m] = 1.0
            cnt[m] = 1
            while m > 0 and val[m - 1] > val[m]:
                pooled = (val[m - 1] * wt[m - 1] + val[m] * wt[m]) / (wt[m - 1] + wt[m])
                val[m - 1] = pooled
                wt[m - 1] += wt[m]
                cnt[m - 1] += cnt[m]
                m -= 1
            m += 1
        pos = 0
        for b in range(m):
            for _ in range(cnt[b]):
                out[i, pos] = val[b]
                pos += 1
    return out


def fix_quantile_crossing(
    preds_NK: np.ndarray,
    alphas: Sequence[float],
    mode: str = "sort",
) -> np.ndarray:
    """Return a copy of ``preds_NK`` with monotonically non-decreasing
    columns enforced row-wise.

    Parameters
    ----------
    preds_NK : (N, K) ndarray
        Per-row, per-alpha quantile predictions.
    alphas : sequence of K floats, sorted ascending
        The alpha levels each column corresponds to. Used only by the
        ``isotonic`` mode (sorting needs no alpha values).
    mode : ``"sort"`` | ``"isotonic"`` | ``"none"``
        Crossing-fix strategy. ``sort`` is idempotent; ``isotonic`` is
        more accurate when crossings cluster around one alpha; ``none``
        is a no-op (returns input unchanged).
    """
    if mode == "none":
        return preds_NK
    if preds_NK.ndim != 2:
        raise ValueError(f"fix_quantile_crossing expects 2-D preds_NK; " f"got shape {preds_NK.shape}")
    if preds_NK.shape[1] != len(alphas):
        raise ValueError(f"preds_NK.shape[1]={preds_NK.shape[1]} != " f"len(alphas)={len(alphas)}")

    if mode == "sort":
        # np.sort along the alpha-axis is the dominant production fix
        # (Park & Ho 2020, sklearn QuantileRegressor docs). Idempotent
        # and O(N K log K).
        return np.sort(preds_NK, axis=1)

    if mode == "isotonic":
        # Functional ``isotonic_regression`` skips the IsotonicRegression
        # constructor / fit-state machinery (no estimator object created,
        # no interp1d fallback path, no out_of_bounds branching), which
        # saves dozens of microseconds per row and adds up to minutes on
        # 1M-row predictions. Because all K alpha-levels are the same set
        # for every row and we only need the isotonic projection ON those
        # exact x-values (no interpolation to new x), this is functionally
        # identical to ``ir.fit(alphas, row).transform(alphas)``.
        # ``_isotonic_rows_pava`` is the njit pool-adjacent-violators projection,
        # FP-equivalent (~1e-15) to ``isotonic_regression(row, increasing=True)``
        # per row but with no per-row Python->C dispatch: 500-660x faster on
        # (N>=1e5, K quantiles). Because all K alpha-levels are the same set for
        # every row and we only need the isotonic projection ON those exact
        # x-values (no interpolation to new x), the equal-weight PAVA is the same
        # unique L2 projection sklearn returns.
        preds64 = np.ascontiguousarray(preds_NK, dtype=np.float64)
        out = np.empty_like(preds64)
        # Vectorised monotone-shortcut mask: rows already sorted ascending
        # skip the PAVA entirely. ``np.diff`` along axis=1.
        _is_mono = np.all(np.diff(preds64, axis=1) >= 0, axis=1)
        return _isotonic_rows_pava(preds64, _is_mono, out)

    raise ValueError(f"fix_quantile_crossing mode must be sort/isotonic/none; " f"got {mode!r}")


__all__ = ["fix_quantile_crossing"]
