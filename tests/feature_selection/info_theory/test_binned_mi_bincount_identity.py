"""Pin the bincount joint-histogram rewrite of ``_wavelet_basis_fe._binned_mi`` as BIT-IDENTICAL to the
prior O(|fa|*|yb|*n) double-loop contingency build (perf commit 1dc05e37, 2.64x).

The rewrite (single ``np.bincount`` over the dense joint code) is bit-identical by construction -- same
plug-in counts, same count/n float64 probabilities, same ascending-unique summation order. The shipped
bench (_benchmarks/bench_binned_mi_hist.py) only PRINTS the max abs diff, so a future regression of either
the new kernel or its binning preamble would not fail CI. This test asserts EXACT equality (the MI feeds
selection ranking, so even a 1-ULP drift could reorder a borderline wavelet leg) across the edge cases the
random bench under-samples: single-unique feature, single-unique y, n=1, the exactly-``nbins`` boundary
(searchsorted branch) vs ``>nbins`` (quantile branch), ternary Haar-leg features, and discrete vs
continuous y -- plus a random battery.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._wavelet_basis_fe import _binned_mi


def _binned_mi_legacy(feat, y, nbins: int = 10) -> float:
    """The pre-1dc05e37 double-loop reference: one O(n) boolean mask per contingency cell."""
    feat = np.asarray(feat, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = feat.size
    if n == 0 or n != y.size:
        return 0.0
    uniq_f = np.unique(feat)
    if uniq_f.size <= nbins:
        fb = np.searchsorted(uniq_f, feat)
    else:
        fb = np.digitize(feat, np.quantile(feat, np.linspace(0.0, 1.0, nbins + 1)[1:-1]))
    if np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        yb = y.astype(np.int64)
    elif np.unique(y).size <= 20:
        yb = np.searchsorted(np.unique(y), y)
    else:
        yb = np.digitize(y, np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1]))
    mi = 0.0
    for a in np.unique(fb):
        pa = np.mean(fb == a)
        if pa <= 0:
            continue
        mask_a = fb == a
        for b in np.unique(yb):
            pab = np.mean(mask_a & (yb == b))
            if pab > 0:
                mi += pab * np.log(pab / (pa * np.mean(yb == b)))
    return float(max(mi, 0.0))


def _edge_cases():
    rng = np.random.default_rng(1)
    return [
        ("single_unique_feat", np.ones(50), rng.integers(0, 3, 50)),
        ("single_unique_y", rng.normal(size=50), np.zeros(50, dtype=int)),
        ("n_eq_1", np.array([1.0]), np.array([0])),
        ("exactly_nbins_unique_feat", np.arange(10.0), np.arange(10) % 3),
        ("gt_nbins_unique_feat_quantile", np.arange(11.0), np.arange(11) % 2),
        ("ternary_feat_continuous_y", rng.choice([-1.0, 0.0, 1.0], 300), rng.normal(size=300)),
        ("continuous_feat_discrete_y", rng.normal(size=400), rng.integers(0, 5, 400)),
    ]


@pytest.mark.parametrize("name,feat,y", _edge_cases(), ids=[c[0] for c in _edge_cases()])
def test_binned_mi_bincount_bit_identical_edge(name, feat, y):
    assert _binned_mi(feat, y) == _binned_mi_legacy(feat, y), f"{name}: bincount _binned_mi diverged from legacy"


def test_binned_mi_bincount_bit_identical_random_battery():
    rng = np.random.default_rng(0)
    max_abs = 0.0
    for _ in range(300):
        n = int(rng.integers(60, 2500))
        feat = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3]) if rng.random() < 0.5 else rng.normal(size=n)
        y = rng.integers(0, int(rng.integers(2, 8)), size=n) if rng.random() < 0.6 else rng.normal(size=n)
        max_abs = max(max_abs, abs(_binned_mi(feat, y) - _binned_mi_legacy(feat, y)))
    assert max_abs == 0.0, f"bincount _binned_mi not bit-identical over random battery: max abs diff {max_abs:.3e}"
