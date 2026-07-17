"""Wave-2 W8: pins the rejection of the 'degenerate single-class target-bin guard'.

These tests document WHY no guard was added: the classification rare-imbalance path
cannot produce a degenerate target bin, and continuous tied-value target bins already
yield finite/stable MI. See the bench-attempt-rejected note in
_unified_fe_gate._coerce_y_classes.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._unified_fe_gate import _coerce_y_classes
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch


def test_classification_rare_1pct_target_never_degenerates():
    # n=150, 1% positives (the literal decisive-test spec). Integer y -> factorize:
    # the coerced target keeps EXACTLY the genuine classes across every seed; no
    # single-class / degenerate target bin can form on the classification path.
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 150
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, 2, replace=False)] = 1
        yb = _coerce_y_classes(y)
        assert set(np.unique(yb).tolist()) == {0, 1}
        assert yb.min() == 0 and yb.max() == 1


def test_continuous_tied_target_bins_give_finite_stable_mi():
    # Continuous y whose quantile edges collapse (heavy ties + a 1% extreme tail)
    # produces FEWER than nbins realised bins -- this is documented np.unique(edges)
    # behaviour, not a silent bug -- and the plug-in MI is finite for every feature.
    rng = np.random.default_rng(1)
    n = 150
    y = np.r_[np.round(rng.normal(0, 0.3, n - 2), 1), [10.0, 11.0]]
    yb = _quantile_bin(y, nbins=10)
    assert len(np.unique(yb)) < 10  # collapse actually occurs (degenerate edges dropped)
    X = rng.normal(0, 1, (n, 5)).astype(np.float64)
    mi = _mi_classif_batch(X, yb.astype(np.int64), nbins=10)
    assert np.isfinite(mi).all()
    assert (mi >= 0).all()


def test_constant_target_quantile_bin_collapses_to_single_class_without_crash():
    # The genuine single-class case (constant target) collapses to one bin and does
    # not crash -- the existing edge handling, not a missing guard.
    yb = _quantile_bin(np.ones(150), nbins=10)
    assert len(np.unique(yb)) == 1
