"""Regression: ``estimate_features_relevancy`` must not leave the caller's target columns scrambled.

The permutation loop shuffles target columns of ``arr = bins.to_numpy(allow_copy=True)``. When that
buffer aliases the caller's data (polars can return a zero-copy view for single-chunk numeric frames),
the pre-fix in-place ``_rng.shuffle(arr[:, idx])`` -- never restored -- permanently corrupts the
caller's target column. The fix shuffles a per-iteration copy and restores the original after each MI
call.

To exercise the aliasing case deterministically (independently of this polars build's copy
semantics), we patch ``DataFrame.to_numpy`` so it returns a VIEW into a backing array we own, then
assert that backing array is pristine after the call.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mlframe.feature_selection.general import estimate_features_relevancy


def test_estimate_features_relevancy_does_not_scramble_aliased_targets(monkeypatch):
    rng = np.random.default_rng(0)
    n = 200
    backing = np.column_stack(
        [
            rng.integers(0, 4, size=n),  # f0
            rng.integers(0, 4, size=n),  # f1
            rng.integers(0, 4, size=n),  # targ
        ]
    ).astype(np.float64)
    df = pl.DataFrame({"f0": backing[:, 0], "f1": backing[:, 1], "targ": backing[:, 2]})
    before = backing[:, 2].copy()

    # Make to_numpy alias our backing buffer so an in-place shuffle would be observable on it.
    monkeypatch.setattr(pl.DataFrame, "to_numpy", lambda self, *a, **k: backing, raising=True)

    def mi_stub(arr, target_indices):
        return np.zeros((len(target_indices), arr.shape[1]), dtype=float)

    estimate_features_relevancy(
        bins=df,
        target_columns=["targ"],
        mi_algorithms_ranking=[mi_stub],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=3,
        min_permuted_mi_evaluations=1,
        verbose=0,
    )

    assert np.array_equal(backing[:, 2], before), "aliased target column was left permuted by the relevancy permutation loop"
