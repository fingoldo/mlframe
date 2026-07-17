"""Unit + biz_value tests for fuzzy-partition (POSP) membership encoding (PZAD fuzzy-set theory)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.fuzzy_features import (
    fuzzy_partition_encode,
    fuzzy_partition_fit,
    fuzzy_partition_names,
    fuzzy_partition_transform,
)


# ---------------------------------------------------------------- unit
def test_triangular_is_exact_ruspini_partition():
    """Triangular is exact ruspini partition."""
    x = np.linspace(0, 10, 200)
    M, _ = fuzzy_partition_encode(x, n_sets=5, kind="triangular")
    assert M.shape == (200, 5)
    assert np.allclose(M.sum(axis=1), 1.0)  # Ruspini: rows sum to 1 everywhere
    assert (M >= 0).all() and (M <= 1).all()
    # at any interior point at most two sets active
    assert (np.count_nonzero(M, axis=1) <= 2).all()


def test_gaussian_rows_normalized():
    """Gaussian rows normalized."""
    x = np.linspace(-3, 3, 150)
    M, _ = fuzzy_partition_encode(x, n_sets=4, kind="gaussian")
    assert M.shape == (150, 4)
    assert np.allclose(M.sum(axis=1), 1.0)  # normalized fuzzy partition
    assert (M > 0).all()  # gaussian has infinite support -> every set active


def test_shoulders_and_centers():
    """Shoulders and centers."""
    recipe = fuzzy_partition_fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), n_sets=5, strategy="uniform", kind="triangular")
    M = fuzzy_partition_transform(np.array([-5.0, 0.0, 2.0, 4.0, 9.0]), recipe)
    assert M[0, 0] == 1.0 and M[4, -1] == 1.0  # below-min and above-max saturate the shoulder sets
    assert M[1, 0] == 1.0 and M[2, 2] == 1.0 and M[3, -1] == 1.0  # exactly on a centre -> full membership


def test_nan_row_is_all_zero():
    """Nan row is all zero."""
    recipe = fuzzy_partition_fit(np.arange(10.0), n_sets=3, kind="gaussian")
    M = fuzzy_partition_transform(np.array([np.nan, 5.0]), recipe)
    assert M[0].sum() == 0.0
    assert abs(M[1].sum() - 1.0) < 1e-12


def test_leakage_safe_fit_transform_separate():
    """Leakage safe fit transform separate."""
    train = np.random.default_rng(0).normal(size=500)
    recipe = fuzzy_partition_fit(train, n_sets=5, kind="triangular")
    test = np.array([train.min() - 10, train.max() + 10, 0.0])
    M = fuzzy_partition_transform(test, recipe)  # centres from train only; out-of-range saturates, no error
    assert np.allclose(M.sum(axis=1), 1.0)


def test_guards():
    """Guards."""
    with pytest.raises(ValueError):
        fuzzy_partition_fit(np.arange(5.0), kind="sigmoid")
    with pytest.raises(ValueError):
        fuzzy_partition_fit(np.arange(5.0), n_sets=1)
    with pytest.raises(ValueError):
        fuzzy_partition_fit(np.full(5, np.nan))  # no finite values
    with pytest.raises(ValueError):
        fuzzy_partition_fit(np.arange(5.0), strategy="log")


def test_names():
    """Names."""
    assert fuzzy_partition_names("age", 3) == ["age_fuzzy_low", "age_fuzzy_medium", "age_fuzzy_high"]
    assert len(fuzzy_partition_names("v", 5)) == 5


def test_degenerate_constant_column():
    """Degenerate constant column."""
    recipe = fuzzy_partition_fit(np.full(20, 7.0), n_sets=5, kind="triangular")  # all identical
    M = fuzzy_partition_transform(np.array([7.0, 6.0, 8.0]), recipe)
    assert M.shape[1] >= 2  # well-defined partition even on a constant column


# ---------------------------------------------------------------- biz_value
def _ridge_rmse(F, y, F_test, y_test, alpha=1e-2):
    """Helper: Ridge rmse."""
    from sklearn.linear_model import Ridge

    m = Ridge(alpha=alpha).fit(F, y)
    pred = m.predict(F_test)
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))


def test_biz_val_fuzzy_partition_beats_hard_binning_and_raw_linear_on_smooth_target():
    """The fuzzification win: on a SMOOTH nonlinear target, a linear model on soft fuzzy-partition memberships
    interpolates across sets and beats (a) raw linear x and (b) hard one-hot binning with the SAME number of sets
    (hard binning is piecewise-constant and jumps at edges; fuzzy is piecewise-smooth). Isolates the soft-vs-hard
    membership effect: both binning models use n_sets features + Ridge, differing only in soft vs hard membership."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(-3, 3, size=800))
    y = np.sin(1.5 * x) + 0.3 * x  # smooth nonlinear
    xt = np.sort(rng.uniform(-3, 3, size=400))
    yt = np.sin(1.5 * xt) + 0.3 * xt
    n_sets = 8

    # fuzzy (gaussian) soft membership
    recipe = fuzzy_partition_fit(x, n_sets=n_sets, kind="gaussian")
    F_fuzzy = fuzzy_partition_transform(x, recipe)
    F_fuzzy_t = fuzzy_partition_transform(xt, recipe)
    rmse_fuzzy = _ridge_rmse(F_fuzzy, y, F_fuzzy_t, yt)

    # hard one-hot binning with the SAME centres/edges (piecewise constant)
    edges = np.quantile(x, np.linspace(0, 1, n_sets + 1))[1:-1]

    def hard_onehot(v):
        """Hard onehot."""
        idx = np.searchsorted(edges, v)
        H = np.zeros((v.shape[0], n_sets))
        H[np.arange(v.shape[0]), idx] = 1.0
        return H

    rmse_hard = _ridge_rmse(hard_onehot(x), y, hard_onehot(xt), yt)

    # raw linear
    rmse_raw = _ridge_rmse(x.reshape(-1, 1), y, xt.reshape(-1, 1), yt)

    assert rmse_fuzzy < rmse_raw * 0.6, f"fuzzy {rmse_fuzzy:.3f} should crush raw-linear {rmse_raw:.3f} on nonlinear target"
    assert rmse_fuzzy < rmse_hard * 0.85, f"soft fuzzy {rmse_fuzzy:.3f} should beat hard one-hot {rmse_hard:.3f} at same n_sets"
