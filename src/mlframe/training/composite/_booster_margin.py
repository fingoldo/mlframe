"""Shared raw-margin extraction for the composite booster wrappers.

Both ``CompositeClassificationEstimator`` (residual log-odds) and
``CompositeGLMEstimator`` (log-link residual) need the SAME family-dispatched
raw-score ladder: LightGBM ``raw_score=True`` -> XGBoost ``output_margin=True``
-> CatBoost ``prediction_type="RawFormulaVal"``. The only per-wrapper differences
are which booster family (classifier vs regressor) is accepted and whether a
multiclass ``(n, K)`` margin is preserved (classification) or flattened to
``(n,)`` (single-output regression). This module carries the one implementation;
each wrapper calls it with its own family set + wrapper name.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def inner_raw_margin(
    model: Any,
    X: Any,
    *,
    lgbm_attr: str,
    xgb_attr: str,
    catboost_attr: str,
    wrapper_name: str,
    keep_2d: bool,
) -> np.ndarray:
    """Raw (pre-link) margin from a fitted booster, family-dispatched.

    ``lgbm_attr`` / ``xgb_attr`` / ``catboost_attr`` name the accepted estimator
    class on each library (``"LGBMClassifier"`` vs ``"LGBMRegressor"`` etc.).
    ``wrapper_name`` prefixes the ``NotImplementedError`` raised when the inner
    exposes no raw-margin path. When ``keep_2d`` is True a multiclass ``(n, K)``
    margin is returned as-is (classification); otherwise the output is flattened
    to ``(n,)`` (single-output regression).
    """
    out = None
    try:
        import lightgbm as lgb
        if isinstance(model, getattr(lgb, lgbm_attr)):
            out = model.predict(X, raw_score=True)
    except Exception:
        pass
    if out is None:
        try:
            import xgboost as xgb
            if isinstance(model, getattr(xgb, xgb_attr)):
                out = model.predict(X, output_margin=True)
        except Exception:
            pass
    if out is None:
        try:
            import catboost as cb
            if isinstance(model, getattr(cb, catboost_attr)):
                out = model.predict(X, prediction_type="RawFormulaVal")
        except Exception:
            pass
    if out is None:
        raise NotImplementedError(
            f"{wrapper_name}: inner {type(model).__name__!r} has no raw-margin path "
            "(LightGBM raw_score / XGBoost output_margin / CatBoost RawFormulaVal). "
            "The base-margin residual contract is undefined without one -- use a "
            "gradient-boosting inner (LightGBM / XGBoost / CatBoost) instead."
        )
    arr = np.asarray(out, dtype=np.float64)
    if not keep_2d:
        return arr.reshape(-1)
    return arr.reshape(-1) if arr.ndim == 1 or arr.shape[1] == 1 else arr
