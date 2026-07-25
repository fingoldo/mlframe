"""Regression tests for the FE-families / orth / screen / stability findings of the mrmr audit fix wave.

Pins the shared continuous-y encoding guard (never int-truncate a continuous target), the tie-corrected
Chatterjee Xi (unbiased on discrete/classification y), the group-blocked-MI "cannot decide" sentinel (no
group clears min_rows -> nan, not a spurious 0.0 leak), the SIS integer-nominal target handling, and the
tree-rescue filter-then-truncate ordering.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_encode_y_for_classif_mi_bins_continuous_not_truncates():
    """A continuous float target must be quantile-binned (many distinct codes preserved), never int-truncated to
    a couple of buckets; an integer target densifies directly."""
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    y_cont = np.linspace(0.0, 1.0, 500)  # 500 distinct floats in [0, 1)
    codes = encode_y_for_classif_mi(y_cont)
    assert codes.dtype == np.int64
    n_classes = int(np.unique(codes).size)
    assert 2 <= n_classes <= 12, f"continuous y must quantile-bin to ~10 classes, not truncate to 1-2; got {n_classes}"
    # pre-fix astype(int64) would collapse [0,1) to a single class
    assert n_classes > 1

    y_int = np.array([5, 5, 9, 9, 2], dtype=np.int64)
    ic = encode_y_for_classif_mi(y_int)
    assert int(np.unique(ic).size) == 3  # dense 0..2, order-preserving


def test_xi_correlation_tie_corrected_on_binary_y():
    """Xi must stay high when y is a measurable (step) function of x even though y is heavily tied (binary), where
    the no-ties formula deflated it; and must agree with the no-ties form when y has no ties."""
    from mlframe.feature_selection.filters._orthogonal_xi_fe import _xi_from_order, xi_correlation

    x = np.linspace(0.0, 1.0, 400)
    y = (x >= 0.5).astype(np.int64)  # measurable step function, heavily tied
    assert xi_correlation(x, y, random_state=0) >= 0.85, "tie-corrected Xi must detect the measurable step relation"

    # no-ties equivalence: on strictly-distinct y the tie-corrected form must match 1 - 3*sum|dr|/(n^2-1)
    rng = np.random.default_rng(0)
    y_dist = rng.permutation(50).astype(np.float64)  # distinct
    ranks = np.argsort(np.argsort(y_dist, kind="stable"), kind="stable").astype(np.float64) + 1.0
    legacy = 1.0 - 3.0 * float(np.sum(np.abs(np.diff(ranks)))) / (50**2 - 1)
    assert abs(_xi_from_order(y_dist) - legacy) < 1e-9


def test_group_blocked_mi_returns_nan_when_no_group_clears_min_rows():
    """group_blocked_mi must return nan ("cannot decide") when no group clears min_rows, so a leak-exempt caller
    cannot confuse a small-groups panel with a genuine 0.0 between-group-only leak."""
    from mlframe.feature_selection.filters.info_theory._group_mi import group_blocked_mi

    n = 10
    codes_x = np.array([0, 1] * 5, dtype=np.int32)
    codes_y = np.array([0, 1] * 5, dtype=np.int32)
    group_offsets = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)  # 5 groups of 2 rows
    sort_idx = np.arange(n, dtype=np.int64)
    mi_no_group = group_blocked_mi(codes_x, codes_y, sort_idx, group_offsets, 2, 2, min_rows=20)
    assert np.isnan(mi_no_group), "no group clears min_rows=20 -> nan (cannot decide), not 0.0"
    mi_ok = group_blocked_mi(codes_x, codes_y, sort_idx, group_offsets, 2, 2, min_rows=1)
    assert np.isfinite(mi_ok), "with min_rows=1 the groups clear -> a finite within-group MI"


def test_sis_screen_ranks_discriminator_for_integer_nominal_target():
    """SIS must not quantile-bin a high-cardinality integer NOMINAL classification target: a feature that cleanly
    separates the classes must out-rank pure noise (quantile-binning would merge classes and bury it)."""
    from mlframe.feature_selection.filters._mrmr_sis_screen import sis_screen

    rng = np.random.default_rng(0)
    n_classes = 40
    reps = 60
    y = np.repeat(np.arange(n_classes), reps).astype(np.int64)  # integer nominal, 40 classes
    n = y.shape[0]
    discriminator = y.astype(np.float64) + rng.normal(0, 0.01, size=n)  # near-perfect class separator
    noise = rng.normal(size=(n, 8))
    X = np.column_stack([noise[:, :4], discriminator, noise[:, 4:]])  # discriminator at column index 4
    survivors, scores = sis_screen(X, y, target_survivors=3, return_scores=True)
    assert 4 in set(int(s) for s in survivors), f"discriminator (col 4) must survive; got {survivors}"
    assert int(np.argmax(scores["fused"])) == 4, "the class separator must score highest under nominal-classification handling"


def test_target_encoding_multiclass_is_mean_class_index():
    """_compute_target_encoding's documented multi-class contract: the per-cell value is the mean class INDEX of
    the rows in that cell (naive single-pass). Pins the previously-untested nominal-multiclass semantics."""
    from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _compute_target_encoding

    factors_data = np.array([[0], [0], [1], [1], [2], [2], [2]], dtype=np.int64)
    classes_y = np.array([2, 2, 0, 0, 0, 1, 2], dtype=np.int64)  # cell 0 -> {2,2}, cell 1 -> {0,0}, cell 2 -> {0,1,2}
    te, _cell_means = _compute_target_encoding(
        factors_data=factors_data,
        idx_tuple=(0,),
        target_indices=np.arange(len(classes_y), dtype=np.int64),
        classes_y=classes_y,
        nbins=np.array([3], dtype=np.int64),
        n_oof_folds=0,
        smoothing=0.0,
        dtype=np.int64,
        allow_naive_leak=True,
    )
    assert te[0] == pytest.approx(2.0)  # cell 0 rows are all class 2
    assert te[2] == pytest.approx(0.0)  # cell 1 rows are all class 0
    assert te[4] == pytest.approx((0 + 1 + 2) / 3)  # cell 2 mixed -> mean class index


def test_tree_rescue_filters_allowed_before_truncation():
    """_apply_tree_rescue must filter by factors_to_use BEFORE the top-k cut, so an allowed feature ranked below
    the global top-k is still rescued (truncating first under-added)."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_selection.filters._mrmr_tree_rescue import MRMRTreeRescued

    rng = np.random.default_rng(0)
    n, p = 800, 30
    X = rng.normal(size=(n, p))
    # the target depends mostly on high-index columns; allowed pool is a few of them plus noise so the allowed
    # informative features sit BELOW the global top-k importances.
    y = (X[:, 25] + X[:, 27] + 0.3 * rng.normal(size=n) > 0).astype(int)
    est = MRMRTreeRescued(tree_rescue="always", tree_rescue_top_k=3, tree_rescue_min_p=1)
    est.n_features_in_ = p
    est.support_ = np.array([], dtype=np.int64)
    est.factors_to_use = np.array([25, 27], dtype=np.int64)  # both informative but not globally top-3 vs col 25 alone
    est._apply_tree_rescue(X, y)
    rescued = set(int(i) for i in np.asarray(est.support_, dtype=np.int64))
    assert {25, 27}.issubset(rescued), f"both allowed informative features must be rescued; got {rescued}"
