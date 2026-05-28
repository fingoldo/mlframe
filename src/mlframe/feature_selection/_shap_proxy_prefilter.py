"""Native-importance pre-filter for the SHAP-proxied feature selector.

On wide data (10k+ columns) the DOMINANT ``ShapProxiedFS.fit`` cost is this pre-filter: rank ALL
columns and keep ``prefilter_top`` of them BEFORE the expensive OOF-SHAP runs (SHAP cost scales with
the surviving column count, and clustering only compresses CORRELATED features, so independent noise
stays as singletons). Profiling at width=10k attributed ~66% of the fit to a single XGBoost fit on all
columns just to read ``feature_importances_``.

Four ranking methods trade speed against interaction-awareness; the dispatcher picks a quality-safe
default for moderate widths and a fast one only for very wide data (see ``resolve_prefilter_method``):

  - ``"model"``  : the original full booster fit on all columns -> ``feature_importances_`` / ``|coef_|``.
                   Interaction-aware and most faithful, but O(n*f*trees) and the wide-data hotspot.
  - ``"univariate"`` : vectorised O(n*f) per-feature ANOVA F-score (sklearn ``f_classif`` /
                   ``f_regression``). Far faster (single BLAS-ish pass), but sees only marginal signal
                   so it misses pure-interaction features (XOR-style partners with ~0 main effect).
  - ``"fast_model"`` : a reduced-budget booster (fewer/shallower trees + column subsampling) for a
                   coarse but still interaction-aware ranking. Middle of the speed/quality tradeoff.
  - ``"gpu_model"``  : the full booster fit routed to XGBoost ``device="cuda"`` when a CUDA device is
                   present. Same ranking as ``"model"`` (interaction-aware, faithful) but the all-columns
                   fit runs on the GPU; the win scales with row count. Gated via kernel_tuning_cache so
                   the CPU<->GPU crossover row count is tuned per-HW rather than hardcoded; falls back to
                   the CPU ``"model"`` path when no device / xgboost-GPU build is available.

All methods return ``working_cols``: a SORTED int array of the kept ORIGINAL column indices (length
<= ``prefilter_top``). The selector restricts its working frame to these and maps final picks back to
original indices via ``working_cols``, so the sklearn ``support_`` stays in original-column space.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PREFILTER_METHODS = ("model", "univariate", "fast_model", "gpu_model")

# Width at/above which the smart "auto" default abandons the faithful full-booster "model" prefilter
# for the cheap interaction-aware "fast_model". Below this, the all-columns model fit is affordable and
# we keep the quality-safe behaviour; above it the fit dominates the wall-clock.
#
# Tuned 2026-05-28 from 6000 -> 4000 based on the iter5+iter7 stage-breakdown bench: at 5000 features
# the "model" prefilter dominates the wall-clock (85s / 47% of a 180s fit on 4000 rows; profiled), and
# the iter5 prefilter bench measured fast_model at 5k is 5.3x faster on the prefilter stage AND keeps
# 8/8 planted informatives, matching the "model" recall. End-to-end this drops the 5k fit ~85s -> ~30s
# on the prefilter, ~3x on the dominant stage, with no measured recovery loss on the regime-data
# benchmark. 10k continues to use fast_model unchanged (auto already routed wide data above 6000).
# Overridable via kernel_tuning_cache ("shap_proxy_prefilter" -> "auto_fast_width") so the crossover is
# tuned per-HW. Values below 2000 effectively force fast_model at all widths the user runs (the iter5
# bench at 2k still showed fast_model at recall=8/8, so even that lower bound is recovery-safe).
_AUTO_FAST_WIDTH = 4000
# Row count at/above which "gpu_model" actually beats CPU "model" for the all-columns fit (GPU transfer
# + kernel-launch overhead dominates on small n; the win scales with rows). Overridable via the cache.
_GPU_MODEL_MIN_ROWS = 20000


def _prefilter_tuning() -> dict:
    """Per-HW prefilter thresholds from kernel_tuning_cache (auto_fast_width / gpu_model_min_rows), or
    ``{}`` when pyutilz / the entry is unavailable (callers fall back to the module defaults)."""
    try:
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_prefilter")
            if isinstance(entry, dict):
                return entry
    except Exception:
        pass
    return {}


def _auto_fast_width() -> int:
    return int(_prefilter_tuning().get("auto_fast_width", _AUTO_FAST_WIDTH))


def _gpu_model_min_rows() -> int:
    return int(_prefilter_tuning().get("gpu_model_min_rows", _GPU_MODEL_MIN_ROWS))


def resolve_prefilter_method(method: str, *, n_features: int, n_rows: int) -> str:
    """Map the user-facing ``prefilter_method`` to a concrete method.

    ``"auto"`` (the recommended default) keeps the quality-safe full-booster ``"model"`` ranking for
    moderate widths, and only for VERY wide data switches to the cheap interaction-aware ``"fast_model"``
    (the full fit dominates the wall-clock there). If a CUDA device is present AND the row count is large
    enough for the GPU to win, ``"auto"`` routes the wide-data case to ``"gpu_model"`` instead of
    ``"fast_model"`` -- same faithful ranking, just on the GPU. Explicit methods pass through unchanged.
    """
    if method != "auto":
        if method not in PREFILTER_METHODS:
            raise ValueError(
                f"prefilter_method={method!r} unknown; expected one of {('auto',) + PREFILTER_METHODS}.")
        return method
    if n_features < _auto_fast_width():
        return "model"
    # Very wide: prefer GPU full-fidelity ranking when it actually pays (enough rows + a device); else
    # the cheap interaction-aware fast booster.
    if n_rows >= _gpu_model_min_rows() and gpu_model_available():
        return "gpu_model"
    return "fast_model"


def gpu_model_available() -> bool:
    """True when XGBoost can fit on ``device="cuda"`` (a CUDA device is enumerable). Probed via cupy's
    runtime (already a soft dep here); falls back to False on any error so callers route to CPU."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _importances_from_fitted(est, n_features: int) -> Optional[np.ndarray]:
    """Pull a length-``n_features`` non-negative importance vector from a fitted estimator, or None."""
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_, dtype=np.float64)
    if hasattr(est, "coef_"):
        return np.abs(np.asarray(est.coef_, dtype=np.float64)).reshape(-1, n_features).sum(axis=0)
    return None


def _topk(importance: np.ndarray, k: int) -> np.ndarray:
    """Sorted int array of the indices of the ``k`` largest importances (ties broken by index)."""
    k = min(k, importance.shape[0])
    return np.sort(np.argsort(-importance, kind="stable")[:k])


def _apply_booster_cap(est, cap):
    """Cap the booster's tree count on a cloned ranking template.

    The prefilter consumes ONLY the ranking of ``feature_importances_``, never an honest loss number
    a user sees, so reducing the tree count is a pure-speed lever: importance attribution stabilises
    well below the default 300 trees (empirical rule of thumb ~100). Reuses the iter9 helper from
    ``_shap_proxy_revalidate`` to set the first recognised booster-size param
    (``n_estimators`` / ``iterations`` / ``num_boost_round``) without duplicating logic. Silently a
    no-op for templates without an ``n_estimators``-like param (e.g. a linear model in tests).
    Returns the param name actually set (or ``None``)."""
    if cap is None:
        return None
    from mlframe.feature_selection._shap_proxy_revalidate import _try_cap_n_estimators

    return _try_cap_n_estimators(est, cap)


def _rank_model(model_template, X, y, *, n_features: int, n_estimators_cap=None):
    """Full booster fit on all columns -> native importances. The original (faithful) ranking.

    When ``n_estimators_cap`` is set the cloned ranking booster's tree count is reduced to the cap
    BEFORE fitting; importance attribution stabilises well below the default tree count, so this
    cuts the fit cost roughly in proportion to the cap ratio while preserving the rank order the
    prefilter consumes (Jaccard >= 0.95 on the top-K vs uncapped, validated by the unit test)."""
    from sklearn.base import clone

    from mlframe.feature_selection._shap_proxy_explain import _unwrap_estimator

    pf = clone(model_template)
    _apply_booster_cap(pf, n_estimators_cap)
    pf.fit(X, y)
    return _importances_from_fitted(_unwrap_estimator(pf), n_features)


def _rank_fast_model(model_template, X, y, *, n_features: int, n_estimators_cap=None):
    """Reduced-budget booster (fewer/shallower trees + column subsampling) for a coarse but still
    interaction-aware ranking. Clones the user's template so non-tree estimators degrade gracefully:
    we only set the params the template actually exposes (xgboost/lightgbm/sklearn GBMs all do). The
    clone carries the template's random_state, so the subsampled fit stays deterministic per call.

    When ``n_estimators_cap`` is supplied, the per-cap-also-applied n_estimators is the MIN of
    fast_model's already-reduced budget (template/4) and the cap, so the cap can never accidentally
    INCREASE fast_model's tree budget vs its own legacy value."""
    from sklearn.base import clone

    from mlframe.feature_selection._shap_proxy_explain import _unwrap_estimator

    pf = clone(model_template)
    # Coarse-but-cheap: ~1/4 the trees, shallow, subsample columns+rows. Only set params the estimator
    # exposes (get_params keys) so a non-GBM template is left as-is rather than crashing on set_params.
    valid = set(pf.get_params(deep=False).keys()) if hasattr(pf, "get_params") else set()
    fast = {}
    n_est = getattr(pf, "n_estimators", None)
    if "n_estimators" in valid and isinstance(n_est, (int, np.integer)):
        budget = max(50, int(n_est) // 4)
        if n_estimators_cap is not None:
            # min() so the cap never increases fast_model's already-reduced budget.
            budget = min(budget, int(n_estimators_cap))
        fast["n_estimators"] = budget
    if "max_depth" in valid:
        fast["max_depth"] = 3
    if "colsample_bytree" in valid:
        fast["colsample_bytree"] = 0.5
    if "subsample" in valid:
        fast["subsample"] = 0.7
    if fast:
        try:
            pf.set_params(**fast)
        except (ValueError, TypeError):
            pass
    pf.fit(X, y)
    return _importances_from_fitted(_unwrap_estimator(pf), n_features)


def _rank_gpu_model(model_template, X, y, *, n_features: int, n_estimators_cap=None):
    """Full booster ranking with the all-columns fit routed to XGBoost ``device="cuda"``. Same ranking
    as ``model`` (interaction-aware, faithful) but on the GPU. Falls back to the CPU ``model`` path on
    any device/build error so a wrong route never loses the result.

    Honours ``n_estimators_cap`` the same way as ``_rank_model`` -- ranking only consumes the importance
    order, so a capped booster preserves the prefilter's product at lower GPU + transfer cost."""
    from sklearn.base import clone

    from mlframe.feature_selection._shap_proxy_explain import _unwrap_estimator

    pf = clone(model_template)
    _apply_booster_cap(pf, n_estimators_cap)
    routed = False
    if hasattr(pf, "get_params") and "device" in pf.get_params(deep=False):  # xgboost >= 2.0 sklearn API
        try:
            pf.set_params(device="cuda")
            routed = True
        except (ValueError, TypeError):
            routed = False
    if not routed:
        logger.warning(
            "ShapProxiedFS: prefilter_method='gpu_model' requested but the model template does not "
            "expose an xgboost-style device= param; running the prefilter fit on CPU.")
    try:
        pf.fit(X, y)
    except Exception as exc:  # device/build hiccup -> CPU fallback (never lose the prefilter)
        if routed:
            logger.warning("ShapProxiedFS: GPU prefilter fit failed (%s); retrying on CPU.", exc)
            pf = clone(model_template)
            _apply_booster_cap(pf, n_estimators_cap)
            pf.fit(X, y)
        else:
            raise
    return _importances_from_fitted(_unwrap_estimator(pf), n_features)


def _rank_univariate(X, y, *, classification: bool):
    """Vectorised O(n*f) per-feature ANOVA F-score (sklearn ``f_classif`` / ``f_regression``). Cheap
    single pass; marginal-signal only (misses pure-interaction features). NaN F-scores (constant
    columns) rank lowest."""
    from sklearn.feature_selection import f_classif, f_regression

    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    scorer = f_classif if classification else f_regression
    with np.errstate(divide="ignore", invalid="ignore"):
        scores, _ = scorer(Xv, y)
    scores = np.asarray(scores, dtype=np.float64)
    scores[~np.isfinite(scores)] = -np.inf  # constant / degenerate columns sink to the bottom
    return scores


def prefilter_columns(
    model_template,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    method: str,
    prefilter_top: int,
    classification: bool,
    n_features: int,
    n_estimators_cap: Optional[int] = 100,
) -> tuple[np.ndarray, dict]:
    """Rank all columns by ``method`` and return ``(working_cols, info)``.

    ``working_cols`` is a sorted int array of kept ORIGINAL column indices (len <= ``prefilter_top``);
    the caller restricts its working frame to these and maps final picks back via ``working_cols``.
    Returns the FULL ``arange(n_features)`` (no-op) when the ranking is unavailable (e.g. a model with
    neither ``feature_importances_`` nor ``coef_``), preserving the original behaviour. ``info`` is the
    report dict: ``dict(method=<resolved>, kept=K, of=n_features, n_estimators_cap=<cap or None>)``.

    ``n_estimators_cap`` (default 100) reduces the cloned ranking booster's tree count for the
    ``"model"``, ``"fast_model"`` and ``"gpu_model"`` methods. The prefilter consumes ONLY the
    ranking of native importances (never an absolute loss number reported to a user), and importance
    attribution stabilises well below the default 300 trees -- so this is the same "cap-the-ranker"
    pattern that iter9 applied to within-cluster refine and the trust guard. ``"fast_model"`` already
    sets a reduced budget (template / 4) and clamps to ``min(current, cap)`` so the cap can never
    accidentally INCREASE its tree count. ``"univariate"`` is a no-op (no booster). Set
    ``n_estimators_cap=None`` to disable the cap (legacy uncapped behaviour).
    """
    resolved = resolve_prefilter_method(method, n_features=n_features, n_rows=int(X.shape[0]))
    if resolved == "univariate":
        importance = _rank_univariate(X, y, classification=classification)
        applied_cap = None  # univariate has no booster -> cap is meaningfully N/A in the report
    elif resolved == "fast_model":
        importance = _rank_fast_model(model_template, X, y, n_features=n_features,
                                      n_estimators_cap=n_estimators_cap)
        applied_cap = n_estimators_cap
    elif resolved == "gpu_model":
        importance = _rank_gpu_model(model_template, X, y, n_features=n_features,
                                     n_estimators_cap=n_estimators_cap)
        applied_cap = n_estimators_cap
    else:  # "model"
        importance = _rank_model(model_template, X, y, n_features=n_features,
                                 n_estimators_cap=n_estimators_cap)
        applied_cap = n_estimators_cap

    if importance is None or importance.shape[0] != n_features:
        # Ranking unavailable -> keep all columns (identity), exactly the pre-existing fall-through.
        return np.arange(n_features), dict(method=resolved, kept=int(n_features), of=int(n_features),
                                           skipped="no_importance", n_estimators_cap=applied_cap)
    working_cols = _topk(importance, prefilter_top)
    return working_cols, dict(method=resolved, kept=int(len(working_cols)), of=int(n_features),
                              n_estimators_cap=applied_cap)


__all__ = [
    "PREFILTER_METHODS",
    "resolve_prefilter_method",
    "gpu_model_available",
    "prefilter_columns",
]
