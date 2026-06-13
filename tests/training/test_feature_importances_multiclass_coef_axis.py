"""Regression: multiclass / multi-target ``coef_`` must aggregate across the
class axis, not collapse to the last class row.

Pre-fix ``_feature_importances`` did ``coef[-1, :]`` for any 2-D ``coef_``,
silently discarding every class but the last -- so a feature that drove an
early class looked unimportant. Binary models (``coef_`` shape ``(1, n)``)
keep their single signed row.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._feature_importances import get_model_feature_importances


def test_multiclass_coef_aggregates_across_classes():
    LogisticRegression = pytest.importorskip("sklearn.linear_model").LogisticRegression
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4))
    # f0 separates class 1; f1 separates class 2. f2/f3 are noise.
    y = np.zeros(n, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 1] > 0.8] = 2
    model = LogisticRegression(max_iter=600).fit(X, y)
    assert model.coef_.shape == (3, 4)

    fi = get_model_feature_importances(model, ["f0", "f1", "f2", "f3"])
    fi = np.asarray(fi, dtype=np.float64)

    # Aggregated FI is non-negative (|coef| collapse) and ranks BOTH driving
    # features above the two noise features. Pre-fix (coef[-1,:]) returned the
    # signed last-class row where f0 was negative and f1 dominated alone, so
    # f0 was NOT above both noise features by magnitude.
    assert np.all(fi >= 0.0), "aggregated multiclass FI must be non-negative"
    assert fi[0] > fi[2] and fi[0] > fi[3], "f0 (drives class 1) must beat noise"
    assert fi[1] > fi[2] and fi[1] > fi[3], "f1 (drives class 2) must beat noise"
    # Bit-check against the canonical mean(|coef|) aggregate.
    expected = np.abs(model.coef_).mean(axis=0)
    assert np.allclose(fi, expected)


def test_binary_coef_keeps_single_signed_row():
    LogisticRegression = pytest.importorskip("sklearn.linear_model").LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = LogisticRegression(max_iter=400).fit(X, y)
    assert model.coef_.shape == (1, 3)
    fi = get_model_feature_importances(model, ["a", "b", "c"])
    # Binary path is preserved: the single signed coef row, unchanged.
    assert np.allclose(np.asarray(fi, dtype=np.float64), model.coef_[0])


def test_multitarget_ridge_coef_aggregates():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 3))
    Y = np.column_stack([3.0 * X[:, 0], -2.0 * X[:, 1]])  # target0~f0, target1~f1
    model = Ridge().fit(X, Y)
    assert model.coef_.shape == (2, 3)
    fi = np.asarray(get_model_feature_importances(model, ["f0", "f1", "f2"]), dtype=np.float64)
    assert np.all(fi >= 0.0)
    # f0 and f1 both carry strong signal on one target each -> both beat f2.
    assert fi[0] > fi[2] and fi[1] > fi[2]
    assert np.allclose(fi, np.abs(model.coef_).mean(axis=0))
