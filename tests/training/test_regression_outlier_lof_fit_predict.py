"""Regression: LocalOutlierFactor (novelty=False) must be scored via fit_predict, not predict.

A plain ``LocalOutlierFactor`` has no usable ``predict`` (that needs ``novelty=True``); calling
it raises ``AttributeError`` and the outlier-detection step got silently skipped even on clean
numeric data, with the logged hint wrongly blaming NaN inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.core._setup_helpers_outliers import (
    _detector_requires_fit_predict,
    _fit_predict_outlier_detector,
)

sklearn = pytest.importorskip("sklearn")
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def test_lof_default_routes_to_fit_predict_on_clean_numeric_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    lof = LocalOutlierFactor(n_neighbors=10)  # novelty=False default
    assert _detector_requires_fit_predict(lof) is True

    # Pre-fix: ``lof.fit(X); lof.predict(X)`` raises AttributeError; OD silently skipped.
    with pytest.raises(AttributeError):
        lof.fit(X)
        lof.predict(X)

    inlier = _fit_predict_outlier_detector(LocalOutlierFactor(n_neighbors=10), X)
    assert set(np.unique(inlier)).issubset({-1, 1})
    assert (inlier == 1).any()


def test_isolation_forest_keeps_fit_predict_path():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    iso = IsolationForest(n_estimators=10, random_state=0)
    assert _detector_requires_fit_predict(iso) is False
    inlier = _fit_predict_outlier_detector(iso, X)
    assert set(np.unique(inlier)).issubset({-1, 1})
