"""CatBoost booster backend + native fast SHAP for the SHAP-proxied feature selector.

CatBoost is the third major GBT family alongside xgboost / lightgbm. Two production wins for the
selector:

1. **Native fast SHAP via** ``CatBoost.get_feature_importance(type='ShapValues', data=Pool(X, ...))``.
   The catboost binary exposes path-dependent TreeSHAP from its C++ kernels directly; the shap
   library's TreeExplainer also wraps this path but adds Python marshalling overhead and a hard
   dependency on ``shap``. Calling the native API skips both. The custom numba TreeSHAP in
   ``_shap_proxy_treeshap`` covers xgboost + lightgbm only (their flat-tree dumps map onto shared
   kernels) -- catboost's oblivious-tree representation does NOT map and re-implementing it as numba
   is strictly worse than just calling the C++ kernel via ``Pool``.

2. **Native categorical handling via** ``cat_features``. Production catboost users pin
   ``cat_features`` on every fit and expect the FS step to honor it; previously they had to switch
   to xgboost which forces one-hot. We pass ``cat_features`` straight to the
   CatBoostClassifier/Regressor constructor (catboost accepts feature NAMES) so it travels through
   sklearn ``clone`` and ``set_params`` cleanly across the four honest-retrain stages.

Single-output only (binary classification / regression), matching the xgboost / lightgbm gate in
``_shap_proxy_explain``. Multiclass catboost ShapValues returns a 3-D tensor we'd have to collapse
the same way the xgboost branch already does; leaving multiclass out of scope for v1.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_CATBOOST_AVAILABLE_CACHE: Optional[bool] = None


def catboost_available() -> bool:
    """True when catboost imports cleanly. Cached for the process lifetime (the import state does not
    change at runtime); call :func:`reset_catboost_available_cache` from tests that monkeypatch the
    catboost import to simulate the not-installed configuration."""
    global _CATBOOST_AVAILABLE_CACHE
    if _CATBOOST_AVAILABLE_CACHE is not None:
        return _CATBOOST_AVAILABLE_CACHE
    try:
        import catboost  # noqa: F401
    except Exception:
        _CATBOOST_AVAILABLE_CACHE = False
        return False
    _CATBOOST_AVAILABLE_CACHE = True
    return True


def reset_catboost_available_cache() -> None:
    """Clear the cached :func:`catboost_available` probe result."""
    global _CATBOOST_AVAILABLE_CACHE
    _CATBOOST_AVAILABLE_CACHE = None


def is_catboost_estimator(est) -> bool:
    """True for a CatBoostClassifier / CatBoostRegressor / CatBoost instance (fitted or unfitted).

    Class-name probe (not isinstance) so we don't pay the catboost import cost when the user is on
    the xgboost / lightgbm path -- the SHAP-proxied FS is shared infra and every fit takes the type
    check on the hot path of ``_shap_phi_and_base``."""
    name = type(est).__name__
    return name in ("CatBoostClassifier", "CatBoostRegressor", "CatBoost")


def make_catboost_estimator(
    classification: bool,
    *,
    random_state: int = 0,
    n_estimators: int = 300,
    n_jobs: int = -1,
    cat_features: Optional[Sequence] = None,
    **kwargs,
):
    """Fast catboost booster whose native ``get_feature_importance(type='ShapValues')`` path is exact
    and well-behaved. Mirrors the xgboost / lightgbm default-estimator factories.

    ``cat_features`` is forwarded to the CatBoost constructor as-is (catboost accepts a sequence of
    integer column positions OR feature names). Passing names is preferable because the SHAP-proxied
    FS slices columns via ``.iloc[:, cols]`` between stages -- name-based ``cat_features`` survives
    the slice unchanged, integer positions need remapping per stage.

    ``n_jobs=-1`` -> ``thread_count=-1`` (catboost's own all-cores knob). ``verbose=False`` /
    ``allow_writing_files=False`` keep the process clean (no per-fit stdout flood / no on-disk
    snapshots in /tmp that survive the fit).
    """
    from catboost import CatBoostClassifier, CatBoostRegressor

    params: dict = dict(
        iterations=int(n_estimators),
        depth=4,
        learning_rate=0.1,
        thread_count=int(n_jobs),
        random_seed=int(random_state),
        verbose=False,
        allow_writing_files=False,
    )
    if cat_features is not None:
        params["cat_features"] = list(cat_features)
    if classification:
        params.setdefault("loss_function", "Logloss")
    params.update(kwargs)
    return CatBoostClassifier(**params) if classification else CatBoostRegressor(**params)


def _build_pool(X, cat_features: Optional[Sequence] = None):
    """Wrap ``X`` in a catboost :class:`Pool`. ``cat_features`` is forwarded as-is.

    Catboost requires a Pool (not a raw DataFrame / ndarray) for SHAP extraction when categorical
    features are in play -- without one it can't tell which columns are categorical."""
    from catboost import Pool

    return Pool(data=X, cat_features=list(cat_features) if cat_features is not None else None)


def catboost_shap(model, X, *, cat_features: Optional[Sequence] = None):
    """Extract a single-output ``(phi, base)`` pair from a fitted catboost model via its native SHAP.

    Returns ``(phi, base)`` where ``phi`` is shape ``(n_samples, n_features)`` and ``base`` is a
    scalar (the model's intercept in margin space) -- the exact contract callers of
    ``_shap_phi_and_base`` rely on.

    Catboost's ShapValues format: a single fit returns ``(n_samples, n_features + 1)`` with the
    last column being the per-row bias (a constant equal to the model's expected value, by
    additivity). For multiclass it returns ``(n_samples, n_classes, n_features + 1)``. We support:

      * regression: shape ``(n, f+1)``, slice off the bias column;
      * binary classification: catboost ALSO returns ``(n, f+1)`` (not 3-D), the SHAP values are in
        margin / log-odds space exactly like xgboost's binary output -- so we treat it the same way;
      * multiclass: explicitly NOT supported (raise) to mirror the xgboost / lightgbm gates in
        ``_shap_proxy_explain``.

    The base value catboost reports per row is identical across rows for one model (TreeSHAP
    invariant: the bias is the global expected value), so we collapse to a scalar via ``[0]`` rather
    than averaging -- the latter would silently mask any per-row drift.
    """
    pool = _build_pool(X, cat_features=cat_features)
    raw = model.get_feature_importance(type="ShapValues", data=pool)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:
        # (n, n_classes, n_features+1) -- multiclass. Out of scope for v1 (selector consumes a
        # single-output margin per row); explicit raise mirrors _shap_phi_and_base.
        raise ValueError(
            f"ShapProxiedFS catboost backend supports binary classification / single-target "
            f"regression only; CatBoost returned a {arr.shape} multiclass tensor."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected catboost ShapValues ndim={arr.ndim}; expected 2-D (n_samples, n_features+1)."
        )
    phi = arr[:, :-1]
    base = float(arr[0, -1])
    return phi, base


# Estimator-class names we recognise for the SHAP fast-path routing in _shap_proxy_explain.
_CATBOOST_NAMES = ("CatBoostClassifier", "CatBoostRegressor", "CatBoost")
