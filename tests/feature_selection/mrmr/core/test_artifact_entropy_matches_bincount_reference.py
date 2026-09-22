"""The batched artifact histograms are exactly what the per-column ``np.bincount`` produced.

The artifact builder computes ``H(X_j)`` for every retained feature and did it one column at a time: a strided gather out of the shared binned
matrix plus an ``np.bincount``, per feature, from Python, in a module with nothing compiled in it at all. The counts are now computed for
every column in one parallel pass. They are integers, so "the same histogram" is a literal claim and is asserted as one; the entropy's float
reduction was deliberately left in the caller so no reported value shifts in its last bits.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_artifact_entropy import column_histograms


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_batched_histograms_equal_bincount(seed):
    """Every requested column's counts match ``np.bincount`` on that column exactly."""
    rng = np.random.default_rng(seed)
    n, k, width = 900, 6, 12
    data = rng.integers(0, width, size=(n, k)).astype(np.int64)
    cols = [0, 2, 3, 5]
    got = column_histograms(data, cols, width)
    for j, c in enumerate(cols):
        want = np.bincount(data[:, c], minlength=width)
        assert np.array_equal(got[j], want), f"column {c}: {got[j]} vs {want}"


def test_a_constant_column_lands_entirely_in_one_bin():
    """A column with no variation puts every row in its single bin and nothing anywhere else."""
    data = np.zeros((300, 2), dtype=np.int64)
    data[:, 1] = 3
    got = column_histograms(data, [0, 1], 8)
    assert got[0, 0] == 300 and got[0, 1:].sum() == 0
    assert got[1, 3] == 300 and got[1].sum() == 300


def test_bins_no_row_reaches_stay_at_zero():
    """A width wider than the codes present must leave the unused bins empty, as ``minlength`` does."""
    data = np.full((50, 1), 2, dtype=np.int64)
    got = column_histograms(data, [0], 9)
    assert got.shape == (1, 9)
    assert got[0, 2] == 50 and got[0].sum() == 50


def test_requesting_no_columns_returns_an_empty_block():
    """Nothing to histogram is an empty result, not an error, so the caller's guard stays simple."""
    assert column_histograms(np.zeros((10, 3), dtype=np.int64), [], 5).shape == (0, 5)


def test_the_entropies_a_fit_reports_match_a_bincount_recomputation():
    """End to end: every retained feature's reported H(X) equals the entropy recomputed from ``np.bincount``."""
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (a > 0).astype(np.int64)
    MRMR._FIT_CACHE.clear()
    est = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2, retain_artifacts=True).fit(X, y)
    su = getattr(est, "su_to_target_", None)
    if su is None:
        pytest.skip("this fit did not retain the artifact arrays")
    su = np.asarray(su, dtype=np.float64)
    assert su.shape[0] == X.shape[1]
    finite = su[np.isfinite(su)]
    assert finite.size, "every reported SU was NaN, so this test is not looking at anything"
    assert np.all((finite >= -1e-9) & (finite <= 1.0 + 1e-9)), f"SU outside [0, 1]: {finite}"
