"""Cross-package API of ``mlframe.feature_selection.shap_proxied_fs``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.shap_proxied_fs`` imports them from here. The names resolve on first use (PEP 562), so
importing this module does not load the TreeSHAP kernels, the CatBoost helpers or ``shap`` until a name is read.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PKG = "mlframe.feature_selection.shap_proxied_fs"
_NAMES = {
    "cached_nonsymmetric_ensemble": ("_shap_proxy_treeshap_catboost", "cached_nonsymmetric_ensemble"),
    "extract_ensemble": ("_shap_proxy_treeshap", "extract_ensemble"),
    "interaction_tensor_numba": ("_shap_proxy_interactions", "_interaction_tensor_numba"),
    "is_catboost_estimator": ("_shap_proxy_catboost", "is_catboost_estimator"),
    "is_supported_lightgbm": ("_shap_proxy_treeshap", "is_supported_lightgbm"),
    "is_supported_xgboost": ("_shap_proxy_treeshap", "is_supported_xgboost"),
    "maybe_patch_shap_xgb_base_score": ("_shap_proxy_explain", "_maybe_patch_shap_xgb_base_score"),
    "pick_backend": ("_shap_proxy_explain", "_pick_backend"),
    "shap_phi_and_base": ("_shap_proxy_explain", "_shap_phi_and_base"),
}

__all__ = sorted(_NAMES)


def __getattr__(name: str) -> Any:
    """Resolve a public name to its implementation on first use."""
    if name in _NAMES:
        module, attr = _NAMES[name]
        return getattr(import_module(f"{_PKG}.{module}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
