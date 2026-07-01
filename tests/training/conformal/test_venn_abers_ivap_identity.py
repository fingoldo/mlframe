"""Regression: the O(grid)-saddle IVAP envelopes equal the reference per-grid-point
sklearn IsotonicRegression refit (CPX-P0-3).

The fast ``_isotonic_envelopes`` replaced an O(grid) sklearn-refit loop with a
cumulative-sum-diagram ``max_l min_r`` greatest-convex-minorant saddle. This pins that
the calibrated p0/p1 envelopes stay bit-exact (to division-order ULP, < 1e-9) vs the
sklearn construction over random, tied, and single-class calibration sets -- so a future
"simplification" of the saddle that breaks the math fails here.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.isotonic import IsotonicRegression

from mlframe.training.composite.venn_abers import _isotonic_envelopes


def _sklearn_refit_envelopes(s_sorted, y_sorted):
    """Reference IVAP: two augmented sklearn isotonic refits per grid point."""
    grid = np.unique(s_sorted)
    p0 = np.empty(grid.shape[0])
    p1 = np.empty(grid.shape[0])
    bx = s_sorted.astype(float)
    by = y_sorted.astype(float)
    for i, g in enumerate(grid):
        gx = float(g)
        x0 = np.append(bx, gx)
        for aug, arr in ((0.0, p0), (1.0, p1)):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            iso.fit(x0, np.append(by, aug))
            arr[i] = float(iso.predict([gx])[0])
    return grid, np.minimum(p0, p1), np.maximum(p0, p1)


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 11])
@pytest.mark.parametrize("n", [5, 30, 150, 400])
@pytest.mark.parametrize("ties", [False, True])
def test_ivap_envelopes_match_sklearn_refit(seed, n, ties):
    rng = np.random.default_rng(seed)
    s = np.sort(rng.uniform(0, 1, n))
    if ties:
        s = np.round(s, 1)  # force tied calibration scores onto a coarse grid
    y = (rng.uniform(0, 1, n) < s).astype(float)

    grid_ref, lo_ref, hi_ref = _sklearn_refit_envelopes(s, y)
    grid_fast, lo_fast, hi_fast = _isotonic_envelopes(s, y)

    assert np.array_equal(grid_ref, grid_fast)
    assert np.abs(lo_ref - lo_fast).max() < 1e-9
    assert np.abs(hi_ref - hi_fast).max() < 1e-9
    assert np.all(lo_fast <= hi_fast + 1e-12)


@pytest.mark.parametrize("y_const", [0.0, 1.0])
def test_ivap_single_class_matches_sklearn(y_const):
    """Degenerate single-class calibration: the saddle must still equal the refit."""
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 1, 60))
    y = np.full(60, y_const)
    _, lo_ref, hi_ref = _sklearn_refit_envelopes(s, y)
    _, lo_fast, hi_fast = _isotonic_envelopes(s, y)
    assert np.abs(lo_ref - lo_fast).max() < 1e-9
    assert np.abs(hi_ref - hi_fast).max() < 1e-9
