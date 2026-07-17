"""Bit-identity sensor for the P10 raw-per-bin memo on
``discovery/_screening_tiny.py`` + ``discovery/_tiny_rerank.py``.

P10: when the regime-aware gate is ON (``per_bin_n_bins>0``), the raw-y per-bin
baseline used to be recomputed per unique base via
``_tiny_cv_rmse_raw_y(..., bin_var=base_screen)``. The raw-y model is trained on
the full feature matrix and is INDEPENDENT of ``bin_var`` -- only the per-bin
re-aggregation differs across bases. The fix fits the K-fold raw-y model ONCE
(capturing per-fold predictions via ``return_fold_preds=True``) and re-bins
those cached predictions for each base via ``_per_bin_from_fold_preds``. This
test pins the memoised per-bin array EXACTLY equal to the per-base-refit array,
for several distinct bin_vars (bases) sharing the same raw-y fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._screening_tiny import (
    _per_bin_from_fold_preds,
    _tiny_cv_rmse_raw_y,
)


def _raw_dataset(n: int = 900, n_bases: int = 4, seed: int = 0):
    """Raw-y dataset plus several distinct continuous ``bin_var`` columns to
    play the role of distinct base columns in the rerank per-bin loop."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 0.7 * X[:, 0] - 1.1 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(0.0, 0.5, n)
    # Distinct bin_vars: correlated-but-not-identical to features so each base
    # quantile-bins the SAME fold predictions differently.
    bin_vars = [X[:, i % X.shape[1]] + 0.3 * rng.normal(size=n) for i in range(n_bases)]
    return y, X, bin_vars


_COMMON_KW = dict(
    family="linear",  # Ridge: deterministic by construction -> bit-exact.
    n_estimators=10,
    num_leaves=7,
    learning_rate=0.1,
    cv_folds=3,
    random_state=0,
    deterministic=True,
)


@pytest.mark.parametrize("n_bins", [8, 5])
@pytest.mark.parametrize("time_aware", [False, True])
def test_memoized_per_bin_equals_per_base_recompute(n_bins, time_aware):
    """P10: ``_per_bin_from_fold_preds`` over the ONE shared raw-y fit ==
    ``_tiny_cv_rmse_raw_y(..., bin_var=base)`` per base. Bit-identical."""
    y, X, bin_vars = _raw_dataset()

    # New path: fit ONCE, capture per-fold predictions, re-bin per base.
    rmse_once, fold_preds = _tiny_cv_rmse_raw_y(
        y_train=y,
        x_train_matrix=X,
        return_fold_preds=True,
        time_aware=time_aware,
        **_COMMON_KW,
    )
    assert np.isfinite(rmse_once)
    assert len(fold_preds) >= 1

    for bv in bin_vars:
        # Legacy path: per-base refit with bin_var set.
        rmse_ref, per_bin_ref = _tiny_cv_rmse_raw_y(
            y_train=y,
            x_train_matrix=X,
            return_per_bin=True,
            n_bins=n_bins,
            bin_var=bv,
            time_aware=time_aware,
            **_COMMON_KW,
        )
        # Memoised path: re-bin the shared fold predictions for THIS bin_var.
        per_bin_memo = _per_bin_from_fold_preds(fold_preds, bv, n_bins=n_bins)

        # Same overall RMSE (the fit is identical), and bit-identical per-bin.
        assert rmse_ref == pytest.approx(rmse_once, rel=0, abs=0)
        assert per_bin_memo.shape == per_bin_ref.shape == (n_bins,)
        np.testing.assert_array_equal(per_bin_memo, per_bin_ref)


def test_memo_handles_nonfinite_y_masking():
    """The fold ``val_idx`` indexes the isfinite(y)-masked space; passing the
    caller-masked bin_var to ``_per_bin_from_fold_preds`` reproduces the masked
    per-base recompute bit-for-bit."""
    y, X, bin_vars = _raw_dataset(seed=3)
    finite = np.ones(y.shape[0], dtype=bool)
    finite[::37] = False  # sprinkle some non-finite y rows
    y = y.copy()
    y[~finite] = np.nan

    rmse_once, fold_preds = _tiny_cv_rmse_raw_y(
        y_train=y,
        x_train_matrix=X,
        return_fold_preds=True,
        **_COMMON_KW,
    )
    assert np.isfinite(rmse_once)

    y_finite_mask = np.isfinite(y)
    for bv in bin_vars:
        _, per_bin_ref = _tiny_cv_rmse_raw_y(
            y_train=y,
            x_train_matrix=X,
            return_per_bin=True,
            n_bins=8,
            bin_var=bv,
            **_COMMON_KW,
        )
        # The rerank masks bin_var by isfinite(y_screen) before handing it to
        # the memo helper (val_idx lives in that masked space).
        per_bin_memo = _per_bin_from_fold_preds(
            fold_preds,
            bv[y_finite_mask],
            n_bins=8,
        )
        np.testing.assert_array_equal(per_bin_memo, per_bin_ref)


def test_return_fold_preds_does_not_change_legacy_returns():
    """Adding ``return_fold_preds`` must not perturb the existing return shapes
    (default False) -- a scalar RMSE and the (rmse, per_bin) tuple unchanged."""
    y, X, bin_vars = _raw_dataset(seed=7)

    scalar = _tiny_cv_rmse_raw_y(y_train=y, x_train_matrix=X, **_COMMON_KW)
    assert np.isscalar(scalar) or isinstance(scalar, float)

    rmse, per_bin = _tiny_cv_rmse_raw_y(
        y_train=y,
        x_train_matrix=X,
        return_per_bin=True,
        n_bins=8,
        bin_var=bin_vars[0],
        **_COMMON_KW,
    )
    assert np.isfinite(rmse)
    assert per_bin.shape == (8,)
