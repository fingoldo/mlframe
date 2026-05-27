"""SHAP attribution engine for the SHAP-proxied feature selector.

Produces the (n_samples, n_features) matrix of per-row SHAP values ``phi`` plus a per-row
baseline so that the *coalition value* of any feature subset ``S`` can be approximated as
``base[i] + sum_{j in S} phi[i, j]`` without retraining a model on ``S``.

Two modes:
  - ``out_of_fold=True`` (default, honest): K-fold CV; for each fold a fresh model is trained on
    the K-1 train folds and SHAP-explained on the held-out fold. Each fold carries its OWN base
    value (``explainer.expected_value``); we store ``base`` per row so the coalition value stays
    additive within the fold the row came from. Concatenating raw phi against a single global base
    would be wrong when fold base values differ -- we avoid that by keeping ``base`` per-row.
  - ``out_of_fold=False`` (fast): one model on the whole search set, explained in-sample.

Multi-model averaging (``n_models > 1``): phi and base are averaged across models trained with
distinct seeds -- a cheap robustness knob (the research found it roughly neutral but safe).

Defaults to ``feature_perturbation="tree_path_dependent"``: fastest exact tree path, and the user's
research found it the only mode where CatBoost does NOT bloat ``expected_value``. We still assert
SHAP additivity (``base + phi.sum(1) ~= model margin``) and warn on bloat.

Binary-classification and single-target regression only (asserts single output); multiclass is out
of scope for v1 because the coalition value is a single scalar margin per row.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)

# Relative deviation of expected_value from the empirical mean margin above which we warn about a
# bloated base value (the CatBoost + interventional quirk). tree_path_dependent stays well under this.
_BASE_BLOAT_REL_TOL = 0.25
# Max allowed relative additivity violation: |base + phi.sum(1) - margin| / (|margin| scale).
_ADDITIVITY_REL_TOL = 1e-2

# Min column count at which the custom numba TreeSHAP is expected to beat the ``shap`` library. On
# narrow data the shap C-extension is already fast and the numba JIT warmup is not worth it; the win
# grows with width (OOF-SHAP on ~2000 features was the profiled hotspot). Routed by ``_pick_backend``;
# overridable via kernel_tuning_cache so the crossover is tuned per-HW rather than hardcoded.
_TREESHAP_NUMBA_MIN_FEATURES = 64


def _treeshap_numba_min_features() -> int:
    """Crossover width for routing to the custom numba TreeSHAP, from kernel_tuning_cache if present."""
    try:
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("numba_min_features"):
                return int(entry["numba_min_features"])
    except Exception:
        pass
    return _TREESHAP_NUMBA_MIN_FEATURES


def _unwrap_estimator(model):
    """Return the SHAP-explainable base estimator, mirroring BorutaShap._boruta_shap_fit_explain."""
    est_name = type(model).__name__
    if est_name == "TransformedTargetRegressor":
        return model.regressor_ if hasattr(model, "regressor_") else model.regressor
    if est_name == "Pipeline":
        from mlframe.utils.misc import get_pipeline_last_element

        return get_pipeline_last_element(model)
    return model


def _pick_backend(explainer_base, X: pd.DataFrame, backend: str) -> str:
    """Resolve the SHAP backend. ``"auto"`` picks the FASTEST correct path by model type + width:
    the custom numba/cupy TreeSHAP for supported xgboost models on wide data, else the ``shap`` lib.
    Explicit ``"shap"`` / ``"treeshap_numba"`` / ``"treeshap_gpu"`` force a path (latter two require a
    supported model and raise otherwise)."""
    if backend in ("treeshap_numba", "treeshap_gpu"):
        return backend
    if backend == "shap":
        return "shap"
    # auto
    from mlframe.feature_selection._shap_proxy_treeshap import is_supported_lightgbm, is_supported_xgboost

    if not (is_supported_xgboost(explainer_base) or is_supported_lightgbm(explainer_base)):
        return "shap"
    if X.shape[1] < _treeshap_numba_min_features():
        return "shap"  # narrow: shap C-extension already fast, skip JIT warmup
    try:
        from mlframe.feature_selection._shap_proxy_treeshap_gpu import gpu_treeshap_available

        if gpu_treeshap_available() and X.shape[0] * X.shape[1] >= 1_000_000:
            return "treeshap_gpu"
    except Exception:
        pass
    return "treeshap_numba"


def _treeshap_phi_and_base(explainer_base, X: pd.DataFrame, use_gpu: bool):
    """Custom path-dependent TreeSHAP backend (numba fallback / optional cupy). Returns ``(phi, base)``
    in margin space, or ``None`` if the model is unsupported (caller falls back to the shap library)."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    ensemble = extract_ensemble(explainer_base)
    if ensemble is None:
        return None
    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if use_gpu:
        try:
            from mlframe.feature_selection._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

            return treeshap_phi_base_gpu(ensemble, Xv)
        except Exception as exc:  # device/cupy hiccup -> numba fallback (never lose the result)
            logger.warning("ShapProxiedFS: GPU TreeSHAP failed (%s); falling back to numba.", exc)
    return treeshap_phi_base_numba(ensemble, Xv)


def _shap_phi_and_base(explainer_base, X: pd.DataFrame, backend: str = "auto"):
    """Extract a single-output (n, f) phi matrix and scalar base from a fitted tree model.

    Returns ``(phi, base)`` in margin / log-odds space (classification) or target space (regression).
    ``backend`` routes to the custom numba/cupy TreeSHAP (fast, wide data) or the ``shap`` library
    (always-correct fallback); see ``_pick_backend``.
    """
    resolved = _pick_backend(explainer_base, X, backend)
    if resolved in ("treeshap_numba", "treeshap_gpu"):
        out = _treeshap_phi_and_base(explainer_base, X, use_gpu=(resolved == "treeshap_gpu"))
        if out is not None:
            phi, base = out
            return np.asarray(phi, dtype=np.float64), float(base)
        # Unsupported despite routing -> fall through to shap.

    import shap

    explainer = shap.TreeExplainer(explainer_base, feature_perturbation="tree_path_dependent")
    phi = explainer.shap_values(X, check_additivity=False)
    base = explainer.expected_value

    # Binary classifiers: SHAP may return a list [class0, class1] or a 3-D array (n, f, classes).
    if isinstance(phi, list):
        if len(phi) == 2:  # binary: positive class
            phi = phi[1]
            base = base[1] if np.ndim(base) > 0 else base
        elif len(phi) == 1:
            phi = phi[0]
            base = base[0] if np.ndim(base) > 0 else base
        else:
            raise ValueError(
                f"ShapProxiedFS supports binary classification / single-target regression only; "
                f"SHAP returned {len(phi)} outputs (multiclass is out of scope)."
            )
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 3:
        # (n, f, n_classes) -- take the positive (last) class for binary.
        if phi.shape[2] != 2:
            raise ValueError(
                f"ShapProxiedFS supports binary / single-target only; got {phi.shape[2]} output classes."
            )
        phi = phi[:, :, 1]
        base = np.asarray(base, dtype=np.float64).ravel()[-1]
    if phi.ndim != 2:
        raise ValueError(f"Unexpected SHAP value ndim={phi.ndim}; expected 2-D (n_samples, n_features).")

    base = float(np.asarray(base, dtype=np.float64).ravel()[0]) if np.ndim(base) > 0 else float(base)
    return phi, base


def _assert_additivity_and_base(phi: np.ndarray, base: float, fold_tag: str = "") -> None:
    """Margin-reconstruction sanity check. ``base + phi.sum(1)`` is the model margin by additivity;
    we cannot recompute the true margin cheaply for every booster, so we check the weaker invariant
    that the base value sits within the spread of the actual margins (bloat detector). The scale uses
    the margin std (not the mean, which is ~0 for balanced binary targets and would false-positive)."""
    margins = base + phi.sum(axis=1)
    mean_margin = float(margins.mean())
    scale = abs(mean_margin) + float(margins.std()) + 1e-9
    if abs(base - mean_margin) / scale > _BASE_BLOAT_REL_TOL:
        logger.warning(
            "ShapProxiedFS%s: base value (%.4g) deviates >%.0f%% from mean margin (%.4g); "
            "coalition-value scale may be distorted (CatBoost/interventional bloat?). "
            "Prefer feature_perturbation='tree_path_dependent'.",
            fold_tag, base, _BASE_BLOAT_REL_TOL * 100, mean_margin,
        )


_JITTER_DEPTHS = (3, 4, 5, 6)  # cycled across models when config_jitter is on


def _fit_one(model_template, X, y, classification: bool, seed: Optional[int], jitter_depth: Optional[int] = None):
    est = clone(model_template)
    if seed is not None and hasattr(est, "random_state"):
        try:
            est.set_params(random_state=seed)
        except (ValueError, TypeError):
            pass
    # Config jitter (#8): vary tree depth across models so averaging is a Monte-Carlo over the
    # path-order arbitrariness that splits credit between correlated features (not just seed jitter).
    if jitter_depth is not None and hasattr(est, "max_depth"):
        try:
            est.set_params(max_depth=int(jitter_depth))
        except (ValueError, TypeError):
            pass
    est.fit(X, y)
    return est


def make_default_estimator(classification: bool, random_state: int = 0, n_estimators: int = 300):
    """Fast tree booster whose SHAP ``tree_path_dependent`` path is exact and well-behaved."""
    from xgboost import XGBClassifier, XGBRegressor

    params = dict(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
    )
    return XGBClassifier(**params, eval_metric="logloss") if classification else XGBRegressor(**params)


def compute_shap_matrix(
    model_template,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    classification: bool,
    out_of_fold: bool = True,
    n_splits: int = 5,
    n_models: int = 1,
    config_jitter: bool = False,
    return_variance: bool = False,
    rng: Optional[np.random.Generator] = None,
    tqdm_desc: Optional[str] = None,
    shap_backend: str = "auto",
):
    """Compute the per-row SHAP value matrix + per-row base value.

    Returns ``(phi, base, y_aligned)``, or ``(phi, base, y_aligned, phi_var)`` when
    ``return_variance`` -- where ``phi_var`` (n, f) is the model-to-model attribution variance within
    each row's fold (lever #7: subsets built from unstable attributions can be penalised). With
    ``n_models == 1`` the variance is zero. ``config_jitter`` cycles tree depth across the models.

    ``shap_backend`` ("auto" default) routes attribution between the custom fast numba/cupy TreeSHAP
    (wide xgboost data) and the ``shap`` library (always-correct fallback); see ``_pick_backend``.

    phi : (n, f) float64 -- per-row SHAP in margin (clf) / target (reg) space.
    base : (n,) float64 -- per-row baseline (the fold's expected_value), constant within a fold.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    n, f = X.shape

    def _models_phi(X_tr, y_tr, X_ex):
        """Mean phi, mean base, and (model-to-model) phi variance over n_models fits on X_ex."""
        s = np.zeros((X_ex.shape[0], f), dtype=np.float64)
        sq = np.zeros((X_ex.shape[0], f), dtype=np.float64)
        b = 0.0
        for m in range(n_models):
            seed = int(rng.integers(0, 2**31 - 1))
            depth = _JITTER_DEPTHS[m % len(_JITTER_DEPTHS)] if config_jitter else None
            est = _fit_one(model_template, X_tr, y_tr, classification, seed, jitter_depth=depth)
            pf, bf = _shap_phi_and_base(_unwrap_estimator(est), X_ex, backend=shap_backend)
            s += pf
            sq += pf * pf
            b += bf
        mean = s / n_models
        var = np.clip(sq / n_models - mean * mean, 0.0, None) if return_variance else None
        return mean, b / n_models, var

    if not out_of_fold:
        phi_acc, base_val, var = _models_phi(X, y, X)
        _assert_additivity_and_base(phi_acc, base_val)
        base_arr = np.full(n, base_val, dtype=np.float64)
        if return_variance:
            return phi_acc, base_arr, y.astype(np.float64), var
        return phi_acc, base_arr, y.astype(np.float64)

    # Out-of-fold: honest per-row attributions.
    if classification:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
        split_iter = splitter.split(X)

    phi = np.zeros((n, f), dtype=np.float64)
    base = np.zeros(n, dtype=np.float64)
    phi_var = np.zeros((n, f), dtype=np.float64) if return_variance else None

    folds = list(split_iter)
    if tqdm_desc:
        from pyutilz.system import tqdmu

        folds = tqdmu(folds, desc=tqdm_desc)

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        phi_fold, base_fold, var_fold = _models_phi(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx])
        _assert_additivity_and_base(phi_fold, base_fold, fold_tag=f" fold {fold_id}")
        phi[va_idx] = phi_fold
        base[va_idx] = base_fold
        if return_variance:
            phi_var[va_idx] = var_fold

    if return_variance:
        return phi, base, y.astype(np.float64), phi_var
    return phi, base, y.astype(np.float64)
