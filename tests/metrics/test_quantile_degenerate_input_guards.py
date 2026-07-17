"""Regression tests for degenerate-input guards in mlframe.metrics.quantile.

Pre-fix:
- ``quantile_summary`` (EDGE1) did not validate ``preds_NK.ndim`` nor that the coverage-pair column index
  was within ``preds_NK.shape[1]`` -> 1-D preds or an out-of-range alpha raised an opaque IndexError.
- ``pit_values`` (EDGE-P2) accepted K==0/K==1 (matching the shape check) and then read sq[0]/sq[k-1] out of
  bounds inside the njit kernel (boundscheck off) -> garbage instead of a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.quantile import quantile_summary, pit_values


def test_quantile_summary_1d_preds_raises_clear_error():
    """Quantile summary 1d preds raises clear error."""
    y = np.array([1.0, 2.0, 3.0])
    preds_1d = np.array([1.0, 2.0, 3.0])  # 1-D, should be (N, K)
    with pytest.raises(ValueError, match="2-D"):
        quantile_summary(y, preds_1d, alphas=[0.1, 0.9], coverage_pairs=((0.1, 0.9),))


def test_quantile_summary_list_preds_no_opaque_type_error():
    # ``pinball_loss_per_alpha`` accepts a list (it does np.asarray internally), so a list ``preds_NK``
    # reaches the coverage loop. Pre-fix the loop indexed the raw list with ``preds_NK[:, col]`` ->
    # opaque "TypeError: list indices must be integers or slices, not tuple". Post-fix it indexes the
    # ndarray view and returns finite coverage metrics.
    """Quantile summary list preds no opaque type error."""
    y = np.array([1.0, 2.0, 3.0])
    preds = [[0.5, 1.5], [1.0, 2.0], [1.5, 2.5]]
    out = quantile_summary(y, preds, alphas=[0.1, 0.9], coverage_pairs=((0.1, 0.9),))
    assert np.isfinite(out["coverage_0.1_0.9"])


def test_pit_values_empty_alphas_raises_clear_error():
    """Pit values empty alphas raises clear error."""
    y = np.array([1.0, 2.0, 3.0])
    preds = np.empty((3, 0))
    with pytest.raises(ValueError, match="at least 2 quantile levels"):
        pit_values(y, preds, alphas=[])
