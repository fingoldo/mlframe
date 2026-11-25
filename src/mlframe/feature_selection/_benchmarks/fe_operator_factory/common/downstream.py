"""Downstream error of a feature set: MAE and RMSE of a ridge model and of HistGradientBoostingRegressor, as relative improvement over the raw-columns baseline.

Decisions use MAE and RMSE only (``METRICS``). R^2 is computed as a REFERENCE column (``REFERENCE_METRICS``), reported next to them as an absolute difference to the raw baseline and never used by a
verdict (see the README, section "Metrics"). Positive relative improvement means lower error than the baseline.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

METRICS = ("mae", "rmse")
REFERENCE_METRICS = ("r2",)


def make_models(hgb_iter: int = 150) -> dict:
    """Fresh ``{'ridge': ..., 'hgb': ...}`` estimators; HGB runs a fixed number of iterations without early stopping so fold results are comparable."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hgb": HistGradientBoostingRegressor(max_iter=hgb_iter, early_stopping=False, random_state=0),
    }


def fit_errors(model, Xtr, ytr, Xte, yte) -> dict:
    """Fit ``model`` on the train part and return ``{'mae', 'rmse', 'r2'}`` on the test part (``r2`` is reference only)."""
    model.fit(Xtr, ytr)
    yte = np.asarray(yte, float)
    resid = yte - model.predict(Xte)
    sst = float(((yte - yte.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid**2).mean())),
        "r2": float(1.0 - (resid**2).sum() / sst) if sst > 0 else float("nan"),
    }


def rel_improvement(err: float, base_err: float) -> float:
    """Relative error reduction versus the baseline error: ``(base - err) / base`` (positive = better than the baseline)."""
    if base_err > 0:
        return float((base_err - err) / base_err)
    # NaN is kept as the contract, but the reason is logged: a perfect baseline makes every relative improvement undefined (11-F18).
    logger.warning("rel_improvement: baseline error %r is not positive; relative improvement is undefined (NaN)", base_err)
    return float("nan")


def errors_by_feature_set(feature_sets: dict, ytr, yte, models: dict | None = None) -> dict:
    """``{'<model>|<set>': {'mae':..., 'rmse':..., 'r2':...}}`` for ``feature_sets = {name: (Xtr, Xte)}`` (every model refit per set)."""
    out = {}
    for name, (Xtr, Xte) in feature_sets.items():
        for mname, model in (models if models else make_models()).items():
            out[f"{mname}|{name}"] = fit_errors(model, Xtr, ytr, Xte, yte)
    return out


def relative_table(errs: dict, base: str = "raw") -> dict:
    """Turn ``errors_by_feature_set`` output into ``{'<model>|<set>': {'mae': rel, 'rmse': rel, 'r2': delta}}`` relative to the ``<model>|<base>`` row.

    ``mae`` / ``rmse`` are relative error reductions (decision metrics); ``r2`` is the plain difference to the baseline R^2 (reference only)."""
    out = {}
    for key, e in errs.items():
        model = key.split("|")[0]
        b = errs[f"{model}|{base}"]
        row = {m: rel_improvement(e[m], b[m]) for m in METRICS}
        row.update({m: float(e[m] - b[m]) for m in REFERENCE_METRICS})
        out[key] = row
    return out
