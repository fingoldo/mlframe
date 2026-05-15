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

import numpy as np


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
        raise ValueError(
            f"fix_quantile_crossing expects 2-D preds_NK; "
            f"got shape {preds_NK.shape}"
        )
    if preds_NK.shape[1] != len(alphas):
        raise ValueError(
            f"preds_NK.shape[1]={preds_NK.shape[1]} != "
            f"len(alphas)={len(alphas)}"
        )

    if mode == "sort":
        # np.sort along the alpha-axis is the dominant production fix
        # (Park & Ho 2020, sklearn QuantileRegressor docs). Idempotent
        # and O(N K log K).
        return np.sort(preds_NK, axis=1)

    if mode == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        alphas_arr = np.asarray(alphas, dtype=np.float64)
        out = np.empty_like(preds_NK)
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        for i in range(preds_NK.shape[0]):
            row = preds_NK[i].astype(np.float64)
            # Already monotone -> shortcut.
            if np.all(np.diff(row) >= 0):
                out[i] = row
                continue
            out[i] = ir.fit(alphas_arr, row).transform(alphas_arr)
        return out

    raise ValueError(
        f"fix_quantile_crossing mode must be sort/isotonic/none; "
        f"got {mode!r}"
    )


__all__ = ["fix_quantile_crossing"]
