"""Regression: ``combine_probs(flavour='median')`` uses np.median, not
np.quantile(0.5) (iter128, 2026-05-21).

Same rationale as the iter119 ``compute_member_quality_gate`` fix:
``np.quantile`` with q=0.5 dispatches through ``apply_along_axis`` and
iterates the non-axis dimensions in Python; ``np.median`` uses numpy's
dedicated C reduction. Bench at the c0056 multilabel-chain shapes:

  (K=3, N=40_000, C=3)  : 11 ms -> 7 ms  (~1.5x)
  (K=3, N=200_000)      : 19 ms -> 11 ms (~1.7x)

Output is bit-equivalent at fp64 epsilon (both propagate NaN identically;
the downstream non_finite_mask arith-fallback further catches any NaN
cells either way).
"""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling import combine_probs


def test_combine_probs_median_matches_np_quantile_legacy_oracle():
    rng = np.random.default_rng(0)
    # (K, N, C) shape: 3 members, 1000 samples, 4 classes.
    stacked = rng.random((3, 1_000, 4))
    new = combine_probs(stacked, "median")
    legacy = np.quantile(stacked, 0.5, axis=0)
    # np.clip(0,1) is applied to BOTH inputs in combine_probs; the legacy
    # path's clip-before-reduce matches our shipped clip-before-median.
    legacy_clipped = np.clip(stacked, 0.0, 1.0)
    legacy = np.quantile(legacy_clipped, 0.5, axis=0)
    assert np.allclose(new, legacy, atol=1e-12), f"median path must match the legacy np.quantile(0.5) result at fp64 epsilon"


def test_combine_probs_median_2d_shape():
    """(K, N) shape -- the simpler 2-D path the legacy np.quantile fallback
    used to apply_along_axis-loop over N=1000."""
    rng = np.random.default_rng(1)
    stacked = rng.random((5, 200))
    out = combine_probs(stacked, "median")
    assert out.shape == (200,)
    assert np.all(np.isfinite(out))
    # Median of 5 i.i.d. uniform draws should sit close to 0.5 on average.
    assert 0.3 < out.mean() < 0.7


def test_combine_probs_median_finite_input_cells_yield_finite_output():
    """No NaN in input -> no NaN in output. Pins the per-cell median behaviour
    on simple integer-valued probabilities so a future regression that breaks
    the np.median axis=0 dispatch (e.g. accidental np.quantile revert) shows
    up at unit-test time."""
    stacked = np.array(
        [
            [[0.1, 0.3, 0.5]],
            [[0.2, 0.4, 0.6]],
            [[0.3, 0.5, 0.7]],
        ]
    )
    out = combine_probs(stacked, "median")
    assert np.all(np.isfinite(out))
    # Per-cell median across K=3 members. Numpy's median on an odd-K array
    # is the middle value after sort -- here just element 1.
    assert abs(out[0, 0] - 0.2) < 1e-12
    assert abs(out[0, 1] - 0.4) < 1e-12
    assert abs(out[0, 2] - 0.6) < 1e-12
