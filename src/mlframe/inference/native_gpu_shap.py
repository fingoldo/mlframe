"""``native_xgboost_gpu_shap_contribs``: XGBoost native GPU SHAP contributions, direct call to the booster.

REJECTED AS A SPEEDUP over ``shap.Explainer`` (kept as a documented negative result + a functional standalone
utility, per "rejected != deleted" -- not wired into ``inference.explainability.compute_shap_on_cv``). The
original hypothesis was that ``shap.Explainer`` forces XGBoost SHAP onto CPU even for a GPU-fit model, so
bypassing it via ``Booster.predict(pred_contribs=True)`` directly would win. A first, invalid A/B (comparing a
CPU-FIT model's booster call against a GPU-FIT model's booster call, not a fair "same model, two code paths"
comparison) measured ~8.2x and looked like a real win. A proper warm + best-of-5 A/B on the SAME GPU-fit model
(``shap.Explainer(gpu_model)`` vs this module's direct ``booster.predict`` call) measured only ~1.04x --
noise-level, no actionable speedup. Root cause: ``shap.Explainer`` already recognizes an XGBoost booster and
delegates internally to the model's own (GPU-accelerated, when GPU-fit) ``predict_contribs``-equivalent path
rather than reimplementing TreeSHAP itself -- there was nothing to bypass. (This is UNRELATED to
``shap.explainers.GPUTreeExplainer``, a separate class noted elsewhere in this codebase as broken on this
hardware, see ``feature_selection/shap_proxied_fs/_shap_proxy_treeshap.py`` -- ``shap.Explainer``'s automatic
dispatch is a different code path and was not affected.)

LightGBM was also considered and separately rejected: empirically measured (best-of, n=20000x20, 100 trees),
LightGBM's ``Booster.predict(pred_contrib=True)`` is NOT GPU-accelerated regardless of training device -- a
GPU-trained booster's contrib computation measured 1.68s vs a CPU-trained booster's 0.59s (SLOWER, not faster).

Kept: values are bit-identical to ``shap.Explainer`` (both implement exact TreeSHAP for XGBoost), so this
remains a valid, functional alternative for a caller who wants to avoid importing ``shap`` at all, or who wants
explicit control over the booster call -- just not a performance win.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def native_gpu_shap_available(model_stub: Any) -> bool:
    """Whether ``model_stub`` is an XGBoost model fit with a CUDA-capable device."""
    if "XGB" not in type(model_stub).__name__:
        return False
    try:
        model_stub.get_booster()
    except AttributeError:
        return False
    device = str(getattr(model_stub, "device", None) or "").lower()
    return "cuda" in device or "gpu" in device


def native_xgboost_gpu_shap_contribs(model_stub: Any, X: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute ``(shap_values, base_values)`` via XGBoost's native GPU contribution path.

    Returns ``None`` when ``model_stub`` isn't a CUDA-fit XGBoost model (caller should fall back to
    ``shap.Explainer``). XGBoost's contribs array has one trailing column (binary/regression) or trailing
    per-class column (multiclass) holding the base/expected value; it is split off to match
    ``shap.Explainer``'s ``(values, base_values)`` contract.
    """
    if not native_gpu_shap_available(model_stub):
        return None
    import xgboost as xgb

    booster = model_stub.get_booster()
    dmatrix = xgb.DMatrix(X)
    contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
    if contribs.ndim == 2:  # binary/regression: (n, n_features + 1)
        return contribs[:, :-1], contribs[:, -1]
    return contribs[:, :, :-1], contribs[:, :, -1]  # multiclass: (n, n_classes, n_features + 1)


__all__ = ["native_gpu_shap_available", "native_xgboost_gpu_shap_contribs"]
