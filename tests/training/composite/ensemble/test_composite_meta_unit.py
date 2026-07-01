"""Unit tests for CompositeOrRawStacker: OOF leakage-free, weight sanity, degenerate fallback, predict shape, clone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.meta import CompositeOrRawStacker, _fit_nnls_2col


def _make_diff_data(n=300, seed=0):
    """y = base + signal(X) + noise -> a 'diff' composite (T = y - base) is the right transform."""
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 2.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = base + 2.0 * f + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f": f})
    return X, y


def test_predict_shape_and_finite():
    X, y = _make_diff_data()
    est = CompositeOrRawStacker(base_estimator=LinearRegression(), transform_name="diff", base_column="base", n_splits=4)
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_weights_non_negative_sum_to_one():
    X, y = _make_diff_data()
    est = CompositeOrRawStacker(base_estimator=LinearRegression(), transform_name="diff", base_column="base", n_splits=4)
    est.fit(X, y)
    assert est.weights_.shape == (2,)
    assert (est.weights_ >= -1e-9).all(), "NNLS weights must be non-negative"
    assert est.weights_.sum() == pytest.approx(1.0, abs=1e-6)


def test_oof_matrix_leakage_free():
    """Every OOF row must be filled by a fold model that never trained on it (no NaN -> every row predicted out-of-fold)."""
    X, y = _make_diff_data(n=120)
    est = CompositeOrRawStacker(base_estimator=LinearRegression(), transform_name="diff", base_column="base", n_splits=5)
    est.fit(X, y)
    assert est.oof_matrix_.shape == (120, 2)
    assert np.isfinite(est.oof_matrix_).all(), "every row must have an out-of-fold prediction for both paths"

    # Leakage proof: the OOF composite column must differ from the FULL-data composite in-sample fit
    # (a leaky 'OOF' that reused the full model would be near-identical). Expect a measurable gap.
    full_insample = np.asarray(est.composite_.predict(X), dtype=float)
    gap = np.mean(np.abs(full_insample - est.oof_matrix_[:, 0]))
    assert gap > 1e-6, "OOF predictions should not equal the full-data in-sample fit (would indicate leakage)"


def test_both_degenerate_fallback_equal_split():
    """NNLS degenerate cases (no finite rows, all-zero solution) fall back to equal split [0.5, 0.5]."""
    # No finite rows -> equal split.
    oof = np.full((10, 2), np.nan)
    y = np.arange(10, dtype=float)
    w = _fit_nnls_2col(oof, y)
    assert np.allclose(w, [0.5, 0.5])

    # Both columns negatively related to y so NNLS pins both weights to 0 -> equal split.
    y2 = np.arange(50, dtype=float)
    neg = -y2
    oof2 = np.column_stack([neg, neg])
    w2 = _fit_nnls_2col(oof2, y2)
    assert np.allclose(w2, [0.5, 0.5])


def test_clone_roundtrip():
    """sklearn.clone must reproduce the estimator from its params without carrying fitted state."""
    est = CompositeOrRawStacker(base_estimator=LinearRegression(), transform_name="diff", base_column="base", n_splits=3)
    cloned = clone(est)
    assert cloned.transform_name == "diff"
    assert cloned.base_column == "base"
    assert cloned.n_splits == 3
    assert not hasattr(cloned, "weights_"), "clone must not carry fitted state"
    # Clone is independently fittable.
    X, y = _make_diff_data(n=80)
    cloned.fit(X, y)
    assert hasattr(cloned, "weights_")


def test_predict_before_fit_raises():
    est = CompositeOrRawStacker(base_estimator=LinearRegression(), base_column="base")
    X, _ = _make_diff_data(n=10)
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        est.predict(X)
