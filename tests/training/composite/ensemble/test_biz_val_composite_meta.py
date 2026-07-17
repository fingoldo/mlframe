"""biz_value: CompositeOrRawStacker recovers the composite when the transform wins, and falls back to raw
(beating the standalone composite on OOS RMSE) when the base is misspecified.

Catches regressions where the OOF/NNLS blender stops adapting to which path actually helps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.meta import CompositeOrRawStacker


def _split(X, y, n_tr):
    """Split."""
    return X.iloc[:n_tr], X.iloc[n_tr:], y[:n_tr], y[n_tr:]


def test_biz_val_meta_recovers_composite_when_transform_wins():
    """DGP: y = base + 2*f + noise with a large-variance base. The 'diff' transform (T = y - base) hands the model the
    clean low-variance residual 2*f, whereas the RAW model must spend its capacity approximating the dominant base term.
    A shallow tree (which cannot represent the linear base natively -- unlike OLS, which would tie) is clearly better on
    the composite path, so the NNLS blender weights composite high (>= 0.7). (LinearRegression would make both paths
    near-perfect -> a meaningless near-tie; the tree is what makes the transform's value measurable.)"""
    rng = np.random.default_rng(1)
    n = 600
    base = rng.normal(50.0, 10.0, n)  # large-variance base dominates y; a depth-limited tree fits it only coarsely
    f = rng.normal(0.0, 1.0, n)
    y = base + 2.0 * f + rng.normal(0.0, 0.5, n)
    X = pd.DataFrame({"base": base, "f": f})

    est = CompositeOrRawStacker(
        base_estimator=DecisionTreeRegressor(max_depth=3, random_state=0),
        transform_name="diff",
        base_column="base",
        n_splits=5,
    )
    est.fit(X, y)
    w_c, _w_r = est.weights_
    assert w_c >= 0.7, f"composite should dominate the blend on a transform-friendly target; got w_composite={w_c:.3f}"


def test_biz_val_meta_falls_back_to_raw_and_beats_composite_on_misspecified_base():
    """DGP: y depends only on f; 'base' is unrelated and crosses zero. The 'ratio' composite (T = y/base,
    inverse y = T*base) then multiplies predictions by a near-zero / sign-flipping base -> blows up OOS.
    The blender must lean on the raw model and beat the standalone composite on OOS RMSE."""
    rng = np.random.default_rng(2)
    n = 800
    base = rng.normal(0.0, 1.0, n)  # crosses zero, unrelated to y -> ratio inverse is unstable
    f = rng.normal(0.0, 1.0, n)
    y = 3.0 * f + 20.0 + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f": f})

    n_tr = 600
    X_tr, X_te, y_tr, y_te = _split(X, y, n_tr)

    # Standalone composite (the path that gets hurt by the misspecified base).
    comp = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="ratio", base_column="base")
    comp.fit(X_tr, y_tr)
    rmse_comp = root_mean_squared_error(y_te, comp.predict(X_te))

    # The adaptive stacker.
    est = CompositeOrRawStacker(base_estimator=LinearRegression(), transform_name="ratio", base_column="base", n_splits=5)
    est.fit(X_tr, y_tr)
    rmse_blend = root_mean_squared_error(y_te, est.predict(X_te))

    _w_c, w_r = est.weights_
    assert w_r >= 0.6, f"raw model should dominate the blend on a misspecified base; got w_raw={w_r:.3f}"
    assert rmse_blend < rmse_comp * 0.95, (
        f"blend (fallback to raw) must beat standalone composite on OOS RMSE; blend={rmse_blend:.4f} composite={rmse_comp:.4f}"
    )
