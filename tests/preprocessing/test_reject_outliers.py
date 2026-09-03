"""Tests for ``mlframe.preprocessing.outliers.reject_outliers`` -- an imblearn-``FunctionSampler``-shaped
resampler that drops outlier rows before fitting a downstream estimator.

Previously had zero test coverage: the default path (no ``model=`` passed) constructs an
``imblearn.pipeline.Pipeline([SimpleImputer, IsolationForest])`` internally, fits it on ``X``, and keeps
only the rows the fitted detector doesn't flag as an outlier (``predict(X) == 1``).
"""

from __future__ import annotations

import numpy as np

from mlframe.preprocessing.outliers import reject_outliers


def _make_data_with_outliers(n_inliers: int = 200, n_outliers: int = 10, seed: int = 0):
    """Builds a 2D blob of inliers plus a handful of far-away outlier points; returns ``(X, y)``."""
    rng = np.random.default_rng(seed)
    inliers = rng.normal(0.0, 1.0, size=(n_inliers, 2))
    outliers = rng.normal(50.0, 1.0, size=(n_outliers, 2))
    X = np.vstack([inliers, outliers])
    y = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])
    return X, y


def test_reject_outliers_drops_rows_and_keeps_x_y_aligned():
    """Reject outliers drops rows and keeps x y aligned."""
    X, y = _make_data_with_outliers()
    X_out, y_out = reject_outliers(X, y, verbose=False)
    assert X_out.shape[0] == y_out.shape[0]
    assert X_out.shape[0] < X.shape[0], "the far-away outlier cluster should be flagged and dropped"
    assert X_out.shape[1] == X.shape[1]


def test_reject_outliers_preferentially_drops_the_injected_outlier_cluster():
    """Most of the surviving rows should come from the inlier cluster, not the injected far-away outliers."""
    X, y = _make_data_with_outliers(n_inliers=200, n_outliers=10)
    _X_out, y_out = reject_outliers(X, y, verbose=False)
    # y==1 marks the injected far-away outlier cluster; it should be a small minority of what survives
    # (IsolationForest isn't guaranteed to catch every single one, but the bulk must be gone).
    assert (y_out == 1).sum() <= 2, f"expected the outlier cluster to be mostly dropped, kept {(y_out == 1).sum()}"


def test_reject_outliers_no_outliers_present_keeps_almost_everything():
    """On data with no genuine outlier structure, IsolationForest's default contamination still flags a
    small fraction (its own baseline rate) -- but the vast majority of rows must survive."""
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(300, 3))
    y = np.zeros(300)
    X_out, _y_out = reject_outliers(X, y, verbose=False)
    assert X_out.shape[0] > 0.7 * X.shape[0]


def test_reject_outliers_accepts_a_custom_model():
    """A caller-supplied ``model`` (any object with fit/predict returning +/-1) is used instead of the
    default imblearn pipeline -- exercises the ``model is not None`` branch."""
    from sklearn.ensemble import IsolationForest

    X, y = _make_data_with_outliers()
    custom_model = IsolationForest(random_state=0)
    X_out, y_out = reject_outliers(X, y, model=custom_model, verbose=False)
    assert X_out.shape[0] == y_out.shape[0]
    assert X_out.shape[0] < X.shape[0]


def test_reject_outliers_verbose_logs_without_raising(caplog):
    """``verbose=True`` (the default) must log a summary line and not raise."""
    import logging

    X, y = _make_data_with_outliers(n_inliers=50, n_outliers=3)
    with caplog.at_level(logging.INFO):
        X_out, y_out = reject_outliers(X, y, verbose=True)
    assert X_out.shape[0] == y_out.shape[0]
    assert any("Outlier rejection" in r.message for r in caplog.records)
