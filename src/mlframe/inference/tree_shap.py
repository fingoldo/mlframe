"""mlframe's TreeSHAP: a drop-in for ``shap.TreeExplainer`` that runs our own implementation wherever it applies.

Our implementation (``mlframe.feature_selection.shap_proxied_fs``: a table form of the path-dependent TreeSHAP on numba, a CUDA kernel for large problems, CatBoost's native kernel for CatBoost) answers for
every fitted XGBoost / LightGBM regressor or binary classifier and every CatBoost regressor / binary classifier. It matches the public ``shap.TreeExplainer(..., feature_perturbation="tree_path_dependent")``
values to machine precision on the CPU and to about 1e-5 on the float32 GPU path, ranks the features the same way, and is several times faster (the measured table is in the README).

The public ``shap`` library is still used (a) to draw plots on values computed here (``to_explanation``) and (b) as the fallback for what we do not implement: multiclass models, SHAP interaction values of
XGBoost (served by XGBoost's own native ``pred_interactions``), sklearn tree ensembles and any other family. A fallback is logged once per model class so it is visible, never silent.

Usage mirrors ``shap``: ``TreeExplainer(model).shap_values(X)``, ``.expected_value``, ``.shap_interaction_values(X)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK_LOGGED: set = set()


def _unwrap(model: Any) -> Any:
    """The tree model inside a ``TransformedTargetRegressor`` / ``Pipeline`` wrapper."""
    name = type(model).__name__
    if name == "TransformedTargetRegressor":
        return model.regressor_ if hasattr(model, "regressor_") else model.regressor
    if name == "Pipeline":
        return model.steps[-1][1]
    return model


def _log_fallback(model: Any, why: str) -> None:
    """Log once per model class that the public shap library answers instead of our implementation."""
    key = (type(model).__name__, why)
    if key not in _FALLBACK_LOGGED:
        _FALLBACK_LOGGED.add(key)
        logger.warning("mlframe TreeSHAP: %s is answered by the public shap library (%s)", type(model).__name__, why)


@dataclass
class Explained:
    """What ``TreeExplainer(model)(X)`` returns, the part of ``shap.Explanation`` the project reads: ``values`` ``(n, features)``, ``base_values`` ``(n,)``, the data and the feature names."""

    values: np.ndarray
    base_values: np.ndarray
    data: Any
    feature_names: list


class TreeExplainer:
    """``shap.TreeExplainer`` with our TreeSHAP underneath; ``backend`` is ``"auto"`` or one of ``"treeshap_numba"`` / ``"treeshap_gpu"`` / ``"shap"`` (force the public library).

    The model is remembered unwrapped; extraction and compilation happen on the first call. ``shap_kwargs`` reach the public explainer when it is used (for example ``data=``).

    Attributes: ``backend_`` (what answered the last call: ``"treeshap_numba"``, ``"treeshap_gpu"``, ``"catboost"`` or ``"shap"``), ``expected_value`` (the base value, scalar for single-output models)."""

    def __init__(self, model: Any, *, backend: str = "auto", **shap_kwargs: Any) -> None:
        self.model = model
        self._est = _unwrap(model)
        self._backend_request = backend
        self._shap_kwargs = shap_kwargs
        self._public: Optional[Any] = None
        self.backend_: Optional[str] = None
        self._expected_value: Any = None

    @property
    def expected_value(self) -> Any:
        """The base value E[f(X)] in margin space. Known before any call for XGBoost / LightGBM (it is a property of the extracted ensemble); set by the first call otherwise."""
        if self._expected_value is None and self._backend_request != "shap" and self._supported() and _is_flat_tree_model(self._est):
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble

            ens = extract_ensemble(self._est)
            if ens is not None:
                self._expected_value = float(ens.base_offset)
        return self._expected_value

    @expected_value.setter
    def expected_value(self, value: Any) -> None:
        """Store the base value reported by the last call."""
        self._expected_value = value

    def __call__(self, X: Any, check_additivity: bool = False) -> Explained:
        """``explainer(X)`` like ``shap.Explainer``: the values with the base value repeated per row."""
        frame = _as_frame(X)
        values = np.asarray(self.shap_values(frame, check_additivity=check_additivity))
        base = np.asarray(self.expected_value, dtype=np.float64).ravel()  # Intentional float64: base values are margins
        base_values = np.full(len(frame), float(base[0])) if base.size == 1 else np.broadcast_to(base, (len(frame), base.size)).copy()
        return Explained(values=values, base_values=base_values, data=frame.to_numpy(), feature_names=[str(c) for c in frame.columns])

    def _supported(self) -> bool:
        """Whether our implementation covers this model (single-output XGBoost / LightGBM / CatBoost)."""
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_catboost import is_catboost_estimator
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import is_supported_lightgbm, is_supported_xgboost

        est = self._est
        if is_catboost_estimator(est):
            return self._catboost_single_output(est)
        return bool(is_supported_xgboost(est) or is_supported_lightgbm(est))

    @staticmethod
    def _catboost_single_output(est: Any) -> bool:
        """CatBoost regressors and binary classifiers (one margin output)."""
        try:
            classes = getattr(est, "classes_", None)
            return classes is None or len(classes) <= 2
        except TypeError:  # ``classes_`` without a length: not the single-margin case
            return False

    def _catboost_backend(self) -> str:
        """``"treeshap_numba"`` for non-symmetric CatBoost trees (explained by our kernel from the JSON export), ``"catboost"`` otherwise."""
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_catboost import cached_nonsymmetric_ensemble

        return "treeshap_numba" if not self._est.get_params().get("cat_features") and cached_nonsymmetric_ensemble(self._est) is not None else "catboost"

    def _public_explainer(self) -> Any:
        """The public ``shap.TreeExplainer`` (built once, with the xgboost base_score workaround the project already applies)."""
        if self._public is None:
            import shap

            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _maybe_patch_shap_xgb_base_score

            with _maybe_patch_shap_xgb_base_score():
                self._public = shap.TreeExplainer(self._est, feature_perturbation="tree_path_dependent", **self._shap_kwargs)
        return self._public

    def shap_values(self, X: Any, check_additivity: bool = False, **_ignored: Any) -> Any:
        """SHAP values of ``X``: an ``(n, features)`` array for regressors and binary classifiers (margin / log-odds), whatever the public library returns for the families it answers (a list or a 3-D array for multiclass)."""
        frame = _as_frame(X)
        if self._backend_request != "shap" and self._supported():
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _pick_backend, _shap_phi_and_base

            phi, base = _shap_phi_and_base(self._est, frame, self._backend_request)
            self.backend_ = _pick_backend(self._est, frame, self._backend_request) if _is_flat_tree_model(self._est) else self._catboost_backend()
            self.expected_value = base
            return phi
        if self._backend_request != "shap":
            _log_fallback(self._est, "model family or output not implemented by mlframe TreeSHAP")
        self.backend_ = "shap"
        explainer = self._public_explainer()
        values = explainer.shap_values(frame, check_additivity=check_additivity)
        self.expected_value = explainer.expected_value
        return values

    def explain(self, X: Any, check_additivity: bool = False) -> Any:
        """A ready-to-plot ``shap.Explanation`` of ``X``: our values wrapped for the public plotting functions, or the public library's own Explanation for the families we do not implement."""
        frame = _as_frame(X)
        if self._backend_request != "shap" and self._supported():
            values = self.shap_values(frame)
            base = np.asarray(self.expected_value, dtype=np.float64).ravel()  # Intentional float64: base values are margins
            return to_explanation(values, frame, base_values=np.full(len(frame), float(base[0])))
        if self._backend_request != "shap":
            _log_fallback(self._est, "model family or output not implemented by mlframe TreeSHAP")
        self.backend_ = "shap"
        return self._public_explainer()(frame, check_additivity=check_additivity)

    def shap_interaction_values(self, X: Any) -> Any:
        """SHAP interaction values, ``(n, features, features)``: our pair-table kernel for LightGBM and XGBoost (XGBoost's native ``pred_interactions`` takes 20 s where the kernel takes 55 ms, 1500 rows x 50 features x 200 depth-4 trees, same values to 2e-6), CatBoost's native kernel, the public library otherwise."""
        frame = _as_frame(X)
        est = self._est
        if self._backend_request != "shap" and self._supported():
            name = type(est).__name__
            if name.startswith("LGBM") or name.startswith("XGB") or name == "Booster":
                from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import _interaction_tensor_numba

                out = _interaction_tensor_numba(est, frame, classification=name.endswith("Classifier"))
                if out is not None:
                    self.backend_ = "treeshap_numba"
                    Phi, base = out
                    self.expected_value = float(np.asarray(base).ravel()[0])
                    return Phi
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_catboost import is_catboost_estimator

            if is_catboost_estimator(est):
                from catboost import EFstrType, Pool

                cat_features = est.get_params().get("cat_features")
                contribs = np.asarray(est.get_feature_importance(Pool(frame, cat_features=cat_features), type=EFstrType.ShapInteractionValues), dtype=np.float64)  # Intentional float64: the interaction tensor contract
                self.backend_ = "catboost"
                self.expected_value = float(contribs[0, -1, -1])
                return contribs[:, :-1, :-1]
            if name.startswith("XGB"):
                import xgboost as xgb

                contribs = est.get_booster().predict(xgb.DMatrix(frame), pred_interactions=True)
                self.backend_ = "xgboost_native"
                self.expected_value = float(contribs[0, -1, -1])
                return np.asarray(contribs[:, :-1, :-1], dtype=np.float64)  # Intentional float64: the interaction tensor contract
        if self._backend_request != "shap":
            _log_fallback(est, "interaction values of this model family are not implemented by mlframe TreeSHAP")
        self.backend_ = "shap"
        explainer = self._public_explainer()
        values = explainer.shap_interaction_values(frame)
        self.expected_value = explainer.expected_value
        return values


def is_supported(model: Any) -> bool:
    """Whether our own implementation (not the public library) computes the SHAP values of ``model``: a fitted XGBoost / LightGBM / CatBoost regressor or binary classifier, possibly inside a Pipeline or a TransformedTargetRegressor."""
    return TreeExplainer(model)._supported()


def _is_flat_tree_model(est: Any) -> bool:
    """XGBoost or LightGBM (answered by our table / CUDA kernels), as opposed to CatBoost's native kernel."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_catboost import is_catboost_estimator

    return not is_catboost_estimator(est)


def _as_frame(X: Any) -> Any:
    """``X`` as a pandas frame (polars and arrays are converted; a frame passes through)."""
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "to_pandas"):
        return X.to_pandas()
    return pd.DataFrame(np.asarray(X))


def to_explanation(values: Any, X: Any, base_values: Any = None, feature_names: Any = None) -> Any:
    """A ``shap.Explanation`` built from values computed here, for the public plotting functions (``shap.plots.beeswarm`` and the rest); the public library draws, it does not compute."""
    import shap

    frame = _as_frame(X)
    names = list(feature_names) if feature_names is not None else [str(c) for c in frame.columns]
    return shap.Explanation(values=np.asarray(values), base_values=base_values, data=frame.to_numpy(), feature_names=names)
