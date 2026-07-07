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
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


# Cache-key namespace for the per-fold FITTED booster pickle inside ``compute_shap_matrix``.
# The OOF-SHAP stage's dominant cost is the n_splits * n_models booster fits (TreeExplainer
# attribution against an already-fitted booster is cheap). iter79's ``shap_phi_`` final-phi cache
# is all-or-nothing -- ANY param change (including downstream knobs that happen to share rng with
# the OOF stage in some callers) invalidates the entire phi matrix. The per-fold cache hits even
# when something orthogonal to (X_tr_fold, y_tr_fold, template, fold seed) changes; it nests
# inside the iter79 cache so a clean re-run of identical params still short-circuits at the outer
# layer without touching the per-fold layer.
_OOF_FOLD_FIT_CACHE_PREFIX = "oof_fold_fit_"


def _build_oof_fold_fit_disk_key(model_template, X_tr_fold, y_tr_fold, classification, seed, jitter_depth, n_estimators_cap):
    """Stable cache key for a per-fold fitted booster inside ``compute_shap_matrix`` (iter83).

    Cache determinants: the fold's actual training slice ((X_tr_fold, y_tr_fold) summary -- these
    arrays naturally encode the fold's row selection AND the global rng-derived KFold splitter
    seed, so no separate fold_idx / train_idx_hash term is needed), the booster template params,
    the per-fold seed (different ``n_models`` slot in the same fold has a different seed and
    must NOT collide), the jitter depth (config_jitter cycles depth across models), the
    n_estimators cap, and the classification flag.

    The eval slice (held-out fold) is intentionally NOT in the key: a cached booster scores any
    X_ex against itself, but ``compute_shap_matrix`` only uses each fold's booster to explain its
    OWN held-out rows so the X_ex passed to ``_shap_phi_and_base`` is implicitly co-determined
    with the fit data.

    Returns ``None`` if hashing fails so the caller falls through to the compute path -- the
    cache is best-effort, never a correctness gate.
    """
    try:
        from mlframe.utils.disk_cache import compose_key, hash_array_summary, hash_object

        try:
            params = model_template.get_params(deep=False)
        except Exception:
            params = {"_repr": repr(model_template)}
        x_tr_key = hash_array_summary(X_tr_fold.values if hasattr(X_tr_fold, "values") else np.asarray(X_tr_fold))
        y_tr_key = hash_array_summary(np.asarray(y_tr_fold))
        state_key = hash_object({
            "seed": int(seed) if seed is not None else None,
            "jitter_depth": int(jitter_depth) if jitter_depth is not None else None,
            "n_estimators_cap": n_estimators_cap,
            "classification": bool(classification),
            "params": params,
        })
        return _OOF_FOLD_FIT_CACHE_PREFIX + compose_key(x_tr_key, y_tr_key, state_key)
    except Exception as exc:
        logger.debug("compute_shap_matrix: per-fold fit disk-cache key build failed (%s); skipping", exc)
        return None


_SHAP_XGB_PATCHED = False


def _safe_float(x=0.0):
    """Bracket-aware float coercer for XGBoost 2.x / 3.x base_score serialised as a JSON array string (``"[0.5]"`` / ``"[5.06E-1, ...]"``); coerces to the first scalar, scalars pass through unchanged.

    Used by the shap<0.52 ``base_score`` workaround (``_maybe_patch_shap_xgb_base_score``). Module-level so it is unit-testable directly without installing the patch.
    """
    import builtins
    import re

    if isinstance(x, str):
        m = re.match(r"\s*\[\s*([^,\]]+)", x)
        if m:
            return builtins.float(m.group(1))
    return builtins.float(x)


def _maybe_patch_shap_xgb_base_score():
    """Workaround for shap<0.52 + xgboost>=2.0 base_score incompatibility (NO-OP on shap>=0.52).

    XGBoost 2.x / 3.x serialise the booster's ``base_score`` as a JSON array string ``"[0.5]"`` instead of the scalar ``"0.5"`` older XGBoost wrote. On shap < 0.52,
    ``shap.explainers._tree.XGBTreeModelLoader.__init__`` coerces it with ``float(...)`` and crashes (``could not convert string to float: '[5.06E-1]'``); we install the
    bracket-aware ``_safe_float`` onto the shap tree module's ``float`` name there. shap >= 0.52 (PR #3530) parses the array natively AND uses ``float`` as a numpy DTYPE
    (``np.asarray(base_score, dtype=float)``) -- replacing it would break that -- so the patch is a strict NO-OP on >=0.52 (it must not touch ``_shap_tree.float``).

    Idempotent (gated on ``_SHAP_XGB_PATCHED``); a no-op if shap is unavailable.
    """
    global _SHAP_XGB_PATCHED
    if _SHAP_XGB_PATCHED:
        return
    try:
        import shap
        from shap.explainers import _tree as _shap_tree
    except Exception:
        _SHAP_XGB_PATCHED = True
        return

    try:
        _shap_ver = tuple(int(p) for p in str(shap.__version__).split(".")[:2])
    except Exception:
        _shap_ver = (0, 0)
    # shap >= 0.52 handles the array base_score natively and uses ``float`` as a numpy dtype; touching it is harmful + unnecessary -> no-op.
    if _shap_ver >= (0, 52):
        _SHAP_XGB_PATCHED = True
        return

    _shap_tree.float = _safe_float
    _SHAP_XGB_PATCHED = True


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
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("numba_min_features"):
                return int(entry["numba_min_features"])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _shap_proxy_explain.py:165: %s", e)
        pass
    return _TREESHAP_NUMBA_MIN_FEATURES


def _unwrap_estimator(model):
    """Return the SHAP-explainable base estimator, mirroring BorutaShap's boruta_shap/_fit_explain."""
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
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import is_supported_lightgbm, is_supported_xgboost

    if not (is_supported_xgboost(explainer_base) or is_supported_lightgbm(explainer_base)):
        return "shap"
    if X.shape[1] < _treeshap_numba_min_features():
        return "shap"  # narrow: shap C-extension already fast, skip JIT warmup
    try:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import gpu_treeshap_available

        if gpu_treeshap_available() and X.shape[0] * X.shape[1] >= 1_000_000:
            return "treeshap_gpu"
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    return "treeshap_numba"


def _treeshap_phi_and_base(explainer_base, X: pd.DataFrame, use_gpu: bool):
    """Custom path-dependent TreeSHAP backend (numba fallback / optional cupy). Returns ``(phi, base)``
    in margin space, or ``None`` if the model is unsupported (caller falls back to the shap library)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    ensemble = extract_ensemble(explainer_base)
    if ensemble is None:
        return None
    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if use_gpu:
        try:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

            return treeshap_phi_base_gpu(ensemble, Xv)
        except Exception as exc:  # device/cupy hiccup -> numba fallback (never lose the result)
            logger.warning("ShapProxiedFS: GPU TreeSHAP failed (%s); falling back to numba.", exc)
    return treeshap_phi_base_numba(ensemble, Xv)


def _shap_phi_and_base(explainer_base, X: pd.DataFrame, backend: str = "auto"):
    """Extract a single-output (n, f) phi matrix and scalar base from a fitted tree model.

    Returns ``(phi, base)`` in margin / log-odds space (classification) or target space (regression).
    ``backend`` routes to the custom numba/cupy TreeSHAP (fast, wide data) or the ``shap`` library
    (always-correct fallback); see ``_pick_backend``. CatBoost models bypass the shap library
    entirely and use the native ``get_feature_importance(type='ShapValues')`` C++ kernel, which
    is exact for the oblivious-tree representation that the numba TreeSHAP path does not model.
    """
    # CatBoost fast path: catboost's oblivious trees do not map onto the numba TreeSHAP kernel
    # (which is calibrated for xgboost/lightgbm's flat-tree dumps) and the shap library's
    # TreeExplainer also just calls catboost's native API but pays extra Python marshalling. Skip
    # both by calling the C++ kernel directly. Class-name probe avoids importing catboost on the
    # xgboost/lightgbm hot path.
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_catboost import (
        catboost_shap, is_catboost_estimator,
    )

    if is_catboost_estimator(explainer_base):
        # Recover cat_features from the booster itself. ``_shap_proxy_cat_features`` is a runtime
        # attribute the template factory stamps on the SOURCE estimator; sklearn ``clone()`` strips
        # it from the per-fold clones used inside ``compute_shap_matrix``. The catboost constructor
        # param ``cat_features`` IS preserved by clone, so query that first; fall back to the
        # stamped attr (used when the caller hand-built a CatBoost without going through the
        # factory).
        cat_features = None
        try:
            cat_features = explainer_base.get_params().get("cat_features")
        except (AttributeError, TypeError):
            cat_features = None
        if cat_features is None:
            cat_features = getattr(explainer_base, "_shap_proxy_cat_features", None)
        return catboost_shap(explainer_base, X, cat_features=cat_features)

    resolved = _pick_backend(explainer_base, X, backend)
    if resolved in ("treeshap_numba", "treeshap_gpu"):
        out = _treeshap_phi_and_base(explainer_base, X, use_gpu=(resolved == "treeshap_gpu"))
        if out is not None:
            phi, base = out
            return np.asarray(phi, dtype=np.float64), float(base)
        # Unsupported despite routing -> fall through to shap.

    import shap

    _maybe_patch_shap_xgb_base_score()
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
            raise ValueError(f"ShapProxiedFS supports binary / single-target only; got {phi.shape[2]} output classes.")
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


def compute_phi_rank_stability(per_fold_phi_mean, top_k: int = 80) -> float:
    """Median pairwise Spearman correlation of per-fold mean |phi| feature rankings.

    ``per_fold_phi_mean`` is shape (n_folds, n_features): each row is the mean |phi| over that fold's
    validation rows. We rank features by mean |phi| within each fold (descending), then take the
    median Spearman across all unordered fold pairs. Restricting to the top-K features before ranking
    (default K=80 = 2 * max_prescreen of 40) drops the noise tail whose ranks are pure permutation
    even in a stable regime and would deflate the correlation.

    Returns a scalar in ``[-1, 1]`` (``1.0`` = identical rankings, ``0`` = unrelated, ``-1`` = inverted).
    With fewer than 2 folds the metric is undefined and ``1.0`` is returned (no folds to disagree).
    """
    arr = np.asarray(per_fold_phi_mean, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"per_fold_phi_mean must be 2-D (n_folds, n_features); got ndim={arr.ndim}")
    n_folds, n_features = arr.shape
    if n_folds < 2 or n_features == 0:
        return 1.0
    k = max(1, min(int(top_k), n_features))
    # Take the union of each fold's top-K features so the ranking is over a consistent feature set.
    # Within that set, lower ranks (1, 2, ...) = larger mean |phi|. Use the negative magnitude so
    # argsort gives descending order; downstream Spearman cares about RANK, not raw value.
    union = set()
    for f in range(n_folds):
        union.update(np.argsort(-arr[f])[:k].tolist())
    cols = np.array(sorted(union), dtype=np.int64)
    if cols.size < 2:
        return 1.0
    sub = arr[:, cols]
    # Rank within each fold (higher mean |phi| -> lower numeric rank). Use scipy if available for the
    # standard tie-handling, else fall back to argsort-of-argsort which is fine for non-tied magnitudes.
    try:
        from scipy.stats import rankdata, spearmanr  # type: ignore
        ranks = np.vstack([rankdata(-sub[f], method="average") for f in range(n_folds)])
        # spearmanr on the matrix returns the full pairwise correlation; extract upper-triangle.
        corr, _ = spearmanr(ranks, axis=1)
        if np.isscalar(corr):
            return float(corr)
        corr = np.asarray(corr, dtype=np.float64)
        iu = np.triu_indices(n_folds, k=1)
        vals = corr[iu]
    except Exception:
        ranks = np.argsort(np.argsort(-sub, axis=1), axis=1).astype(np.float64)
        vals = []
        for i in range(n_folds):
            ri = ranks[i] - ranks[i].mean()
            for j in range(i + 1, n_folds):
                rj = ranks[j] - ranks[j].mean()
                denom = float(np.sqrt((ri * ri).sum() * (rj * rj).sum()))
                vals.append(float((ri * rj).sum() / denom) if denom > 0 else 1.0)
        vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return 1.0
    # Median is more robust than mean against one outlier fold (e.g. a tiny class-imbalance KFold split).
    return float(np.median(vals))


# bench-attempt-rejected (2026-05-28, iter20): converting ``X`` to numpy at the entry of ``_fit_one`` +
# ``_models_phi`` + ``_honest_loss`` (revalidate.py) to skip XGBoost's ``_assign_dmatrix_features`` +
# ``from_cstr_to_pystr`` + ``_validate_features`` marshalling. Motivated by iter19's cProfile attributing
# ~17s of OOF-SHAP wall to that path at width=10000. Measured isolated cold+warm OOF-SHAP on a 4000x400
# post-cohort frame: baseline cold=11.37s warm=9.11s; numpy-input cold=10.90s warm=9.11s -- 4% cold, 0%
# warm. End-to-end scaling bench (medians over 4 warm runs at 6k+10k) showed OOF-SHAP within +/-3% and
# total fit within noise (15-20% Windows run-to-run variance). The cProfile 17s attribution was a
# JIT/cold-cache artifact, NOT a steady-state marshalling cost. Same lesson as iter6 (4-11% gain on
# regime synthetic, didn't ship). Do not re-attempt without a NEW data regime (object/categorical-mixed
# DataFrame where .values triggers a real copy / dtype conversion) -- on float64 single-block frames
# the .values pass-through is essentially free for both pandas AND XGBoost.


def _fit_one(model_template, X, y, classification: bool, seed: Optional[int], jitter_depth: Optional[int] = None,
             inner_n_jobs: Optional[int] = None, n_estimators_cap: Optional[int] = None):
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
    # Cap per-fit threads when several folds train concurrently (avoid outer x inner oversubscription).
    if inner_n_jobs is not None and hasattr(est, "n_jobs"):
        try:
            est.set_params(n_jobs=int(inner_n_jobs))
        except (ValueError, TypeError):
            pass
    # Cap the booster's tree count (iter19): the SHAP attribution path stabilises on per-feature
    # credit well before the full template trains -- the proxy consumes the ATTRIBUTION ranking and
    # the coalition value, both of which are determined by the fitted model's structure, not by how
    # many late trees the booster gets to add. Capping at ~100 trees mirrors the same "cap-the-ranker"
    # lever already applied to prefilter / trust-guard / within-cluster-refine (iter9/iter10/iter12).
    # We never apply via min(current, cap): the cap is a clamp, never an increase. Param dispatched
    # via the shared ``_try_cap_n_estimators`` helper so xgboost / lightgbm / catboost all work.
    if n_estimators_cap is not None:
        try:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _try_cap_n_estimators

            current = None
            for _name in ("n_estimators", "iterations", "num_boost_round", "num_iterations"):
                try:
                    current = est.get_params().get(_name)
                    if current is not None:
                        break
                except (AttributeError, TypeError):
                    continue
            effective_cap = int(n_estimators_cap) if current is None else min(int(current), int(n_estimators_cap))
            _try_cap_n_estimators(est, effective_cap)
        except (ImportError, ValueError, TypeError):
            pass
    est.fit(X, y)
    return est


_VALID_BOOSTER_KINDS = ("xgboost", "catboost")


def make_default_estimator(classification: bool, random_state: int = 0, n_estimators: int = 300, *, booster_kind: Optional[str] = None, cat_features=None):
    """Fast tree booster whose SHAP path is exact and well-behaved.

    ``booster_kind`` selects the GBT family: ``"xgboost"`` (default, fast hist tree with shap-library
    or numba TreeSHAP) or ``"catboost"`` (oblivious trees with the native C++ ShapValues kernel and
    first-class categorical feature support). ``None`` resolves to ``"xgboost"`` so the legacy
    default is byte-identical when callers do not opt in.

    ``cat_features`` is forwarded to the catboost constructor when ``booster_kind="catboost"``;
    the catboost template also stamps ``_shap_proxy_cat_features`` on the estimator so that
    sklearn ``clone()``-based downstream stages (OOF-SHAP folds, honest revalidation, refine,
    trust guard, importance ablation) can recover the categorical hint inside ``_shap_phi_and_base``.
    Ignored for xgboost (xgboost requires one-hot encoding upstream).
    """
    if booster_kind is None:
        booster_kind = "xgboost"
    kind = str(booster_kind).lower()
    if kind not in _VALID_BOOSTER_KINDS:
        raise ValueError(f"booster_kind must be one of {_VALID_BOOSTER_KINDS}; got {booster_kind!r}")
    if kind == "catboost":
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_catboost import make_catboost_estimator

        est = make_catboost_estimator(
            classification=classification, random_state=int(random_state),
            n_estimators=int(n_estimators), cat_features=cat_features,
        )
        # Stamp the categorical hint on the estimator instance so it survives sklearn ``clone`` of
        # the template across folds / honest retrains -- the `_shap_proxy_cat_features` attr is read
        # back by ``_shap_phi_and_base`` to build the catboost ``Pool`` with matching categoricals.
        est._shap_proxy_cat_features = list(cat_features) if cat_features is not None else None
        return est

    # bench-attempt-rejected (iter51): hypothesised that ``xgboost.core._init`` (QuantileDMatrix._init,
    # not a module-level init) was reducible via DMatrix sharing or Booster pooling across the ~8-60
    # per-fit calls inside one selector run. Three options surveyed:
    #   (A) DMatrix pre-construction: each fit takes a DIFFERENT column subset of X (OOF-SHAP folds,
    #       revalidation candidates, refine probes) -- no two fits share the same matrix.
    #   (B) Booster pool: xgboost's Booster.reset (3.0+) frees data caches but does NOT permit
    #       re-training a fresh booster on different data. No public API for state-reuse.
    #   (C) Module-level _init: there is none -- core.py:1553 is QuantileDMatrix._init, an instance
    #       constructor; the work it does (callback registration + C-bridge DMatrix allocation +
    #       quantile sketch) is intrinsic per-fit cost.
    # Probe (width=2000, n_rows=3000): _init ncalls=8, tottime=0.332s, percall=42ms; total e2e 25.7s.
    # At C4 (width=20000, n_rows=10000) the percall scales with data; aggregate tottime ~6s, e2e ~77s,
    # so eliminating ALL _init would yield <8% e2e speedup and the only paths to reduce ncalls require
    # either changing the proxy's per-fit column-subset contract or forking xgboost. Honest-negative.
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
    n_jobs: int = 1,
    n_estimators_cap: Optional[int] = None,
    inner_n_jobs_cap: bool = False,
    return_per_fold_phi_mean: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
):
    """Compute the per-row SHAP value matrix + per-row base value.

    Returns ``(phi, base, y_aligned)``, or ``(phi, base, y_aligned, phi_var)`` when
    ``return_variance`` -- where ``phi_var`` (n, f) is the model-to-model attribution variance within
    each row's fold (lever #7: subsets built from unstable attributions can be penalised). With
    ``n_models == 1`` the variance is zero. ``config_jitter`` cycles tree depth across the models.

    ``shap_backend`` ("auto" default) routes attribution between the custom fast numba/cupy TreeSHAP
    (wide xgboost data) and the ``shap`` library (always-correct fallback); see ``_pick_backend``.

    ``n_jobs`` runs the independent out-of-fold folds concurrently (threading backend: xgb/lgbm release
    the GIL during fit, and SHAP attribution is C/numba). Seeds are pre-drawn in fold order, so the
    result is byte-identical to the serial path regardless of which fold finishes first. ``1`` (default)
    keeps the serial path; the selector passes its own ``n_jobs`` so wide-data fits parallelize the
    OOF-SHAP stage. Each fold's own fit threads are capped so outer-folds x inner-threads don't
    oversubscribe the cores.

    ``n_estimators_cap`` (iter19) clamps the per-fold booster's tree count via ``min(current, cap)``;
    the SHAP attribution + coalition ranking are determined by the fitted model's structure (per-row
    marginal credit aggregated over trees), not by how many late-stage refinement trees the booster
    grows. Capping at ~100 mirrors the same "cap-the-ranker" lever applied to prefilter / trust-guard
    / refine (iter9-iter12); xgboost training dominates the OOF stage per cProfile (iter19), so
    fewer trees translate near-linearly to wall-clock. ``None`` disables the cap (legacy behaviour).

    phi : (n, f) float64 -- per-row SHAP in margin (clf) / target (reg) space.
    base : (n,) float64 -- per-row baseline (the fold's expected_value), constant within a fold.

    ``return_per_fold_phi_mean`` (iter59): when True, the return tuple gains a trailing
    ``per_fold_phi_mean`` of shape (n_splits, n_features), where row k is ``|phi|.mean(axis=0)`` over
    fold k's validation rows. Used downstream by ``compute_phi_rank_stability`` to gauge how stable
    the top-K |phi| ranking is across folds; the SHAP-proxied selector consumes this as an
    adaptive-prescreen-width signal. Cheap to retain: 8 * n_splits * n_features bytes (~400KB at the
    default 5-fold / 10000-column regime). With ``out_of_fold=False`` we still return a single-row
    array (the in-sample mean) so the consumer's contract stays uniform.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    n, f = X.shape

    # Content-addressable disk cache: keyed by (X summary, y summary, booster params, fold/model
    # state). The OOF-SHAP stage dominates wall-clock at C3/C4; in hyperparam sweeps and ablations
    # the same (X, y, template) tuple recurs verbatim, so a hit returns phi + base + tail without
    # paying the per-fold xgboost fit + TreeSHAP attribution cost. ``cache_dir=None`` skips entirely
    # (default, zero behaviour change). The RNG bit-stream is snapshotted at entry so the cache key
    # reflects the seed every fold WILL see, not whatever state ``rng`` reaches after the function
    # consumes draws -- without that snapshot a hit on a partially-consumed rng would key under
    # different bits than the original miss.
    _cache = None
    _cache_key = None
    if cache_dir is not None:
        try:
            from mlframe.utils.disk_cache import DiskCache, compose_key, hash_array_summary, hash_object

            _cache = DiskCache(cache_dir)
            try:
                _params = model_template.get_params(deep=False)
            except Exception:
                _params = {"_repr": repr(model_template)}
            try:
                _rng_state = rng.bit_generator.state
            except AttributeError:
                _rng_state = repr(rng)
            _x_key = hash_array_summary(X.values if hasattr(X, "values") else np.asarray(X))
            _y_key = hash_array_summary(np.asarray(y))
            _state_key = hash_object({
                "params": _params,
                "rng": _rng_state,
                "n_splits": int(n_splits),
                "n_models": int(n_models),
                "out_of_fold": bool(out_of_fold),
                "config_jitter": bool(config_jitter),
                "return_variance": bool(return_variance),
                "return_per_fold_phi_mean": bool(return_per_fold_phi_mean),
                "shap_backend": str(shap_backend),
                "n_estimators_cap": n_estimators_cap,
                "inner_n_jobs_cap": bool(inner_n_jobs_cap),
                "classification": bool(classification),
                "columns": list(map(str, getattr(X, "columns", []))),
            })
            _cache_key = "shap_phi_" + compose_key(_x_key, _y_key, _state_key)
            _hit = _cache.get(_cache_key)
            if _hit is not None:
                logger.debug("compute_shap_matrix: cache hit key=%s", _cache_key)
                # Replay the exact ``rng.integers`` draws the miss path would consume so the shared
                # rng's post-state matches byte-for-byte. Without this, downstream stages that share
                # the same rng (selector heuristics, refine, trust-guard) would observe different
                # draws between cache-hit and cache-miss runs -- breaking subset bit-identity.
                # Sequence (mirrors the non-cached code path below):
                #   * out_of_fold=False: n_models seeds.
                #   * out_of_fold=True: 1 splitter seed + n_splits * n_models fold seeds.
                if not out_of_fold:
                    for _ in range(n_models):
                        rng.integers(0, 2**31 - 1)
                else:
                    rng.integers(0, 2**31 - 1)
                    for _ in range(n_splits * n_models):
                        rng.integers(0, 2**31 - 1)
                return _hit
        except Exception as exc:
            # Cache failures are non-fatal: we lose the speedup but the compute path stays correct.
            logger.debug("compute_shap_matrix: cache disabled (%s)", exc)
            _cache = None
            _cache_key = None

    def _maybe_store(result):
        if _cache is not None and _cache_key is not None:
            try:
                _cache.put(_cache_key, result)
            except Exception as exc:
                logger.debug("compute_shap_matrix: cache put failed (%s)", exc)
        return result

    def _models_phi(X_tr, y_tr, X_ex, seeds, inner_n_jobs=None):
        """Mean phi, mean base, and (model-to-model) phi variance over n_models fits on X_ex.

        ``seeds`` is the pre-drawn list of model seeds (one per model). Passing them in -- instead of
        drawing from ``rng`` inside the loop -- decouples the per-fold work from RNG draw ORDER, so the
        out-of-fold loop can run folds concurrently and still produce the byte-identical result a serial
        run would (the seed each fold sees is fixed regardless of which fold finishes first).

        Per-fold disk cache (iter83): when ``_cache`` (the iter79 DiskCache opened above) is live,
        each model fit checks the ``oof_fold_fit_`` namespace before calling ``_fit_one``. A hit
        loads the pickled booster; the SHAP attribution then runs against the cached estimator and
        is unaffected. The outer iter79 ``shap_phi_`` cache short-circuits the WHOLE computation
        for an identical-param re-run; this inner cache handles the case where the iter79 key
        misses but the per-fold determinants happen to match (e.g. caller mutated something that
        feeds into the outer key without affecting any fold's fit data + seed)."""
        s = np.zeros((X_ex.shape[0], f), dtype=np.float64)
        sq = np.zeros((X_ex.shape[0], f), dtype=np.float64)
        b = 0.0
        for m in range(n_models):
            depth = _JITTER_DEPTHS[m % len(_JITTER_DEPTHS)] if config_jitter else None
            est = None
            fold_fit_key = None
            if _cache is not None:
                fold_fit_key = _build_oof_fold_fit_disk_key(
                    model_template, X_tr, y_tr, classification, seeds[m], depth, n_estimators_cap,
                )
                if fold_fit_key is not None:
                    try:
                        cached_est = _cache.get(fold_fit_key)
                    except Exception as exc:
                        logger.debug("compute_shap_matrix: per-fold disk cache get failed (%s); skipping", exc)
                        cached_est = None
                    if cached_est is not None:
                        est = cached_est
            if est is None:
                est = _fit_one(model_template, X_tr, y_tr, classification, seeds[m], jitter_depth=depth,
                               inner_n_jobs=inner_n_jobs, n_estimators_cap=n_estimators_cap)
                if _cache is not None and fold_fit_key is not None:
                    try:
                        _cache.put(fold_fit_key, est)
                    except Exception as exc:
                        logger.debug("compute_shap_matrix: per-fold disk cache put failed (%s); skipping", exc)
            pf, bf = _shap_phi_and_base(_unwrap_estimator(est), X_ex, backend=shap_backend)
            s += pf
            sq += pf * pf
            b += bf
        mean = s / n_models
        var = np.clip(sq / n_models - mean * mean, 0.0, None) if return_variance else None
        return mean, b / n_models, var

    if not out_of_fold:
        seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_models)]
        phi_acc, base_val, var = _models_phi(X, y, X, seeds)
        _assert_additivity_and_base(phi_acc, base_val)
        base_arr = np.full(n, base_val, dtype=np.float64)
        per_fold_mean = np.abs(phi_acc).mean(axis=0, keepdims=True) if return_per_fold_phi_mean else None
        out_tail: list = []
        if return_variance:
            out_tail.append(var)
        if return_per_fold_phi_mean:
            out_tail.append(per_fold_mean)
        if out_tail:
            return _maybe_store((phi_acc, base_arr, y.astype(np.float64), *out_tail))
        return _maybe_store((phi_acc, base_arr, y.astype(np.float64)))

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
    per_fold_mean = np.zeros((n_splits, f), dtype=np.float64) if return_per_fold_phi_mean else None

    folds = list(split_iter)
    # Pre-draw every fold's model seeds in fold order BEFORE any fit, so concurrent folds yield the
    # byte-identical result the serial loop would (the RNG is consumed in fold order regardless of
    # which fold's thread runs first).
    fold_seeds = [[int(rng.integers(0, 2**31 - 1)) for _ in range(n_models)] for _ in folds]

    import os

    n_cores = os.cpu_count() or 1
    outer = 1
    if n_jobs not in (None, 0, 1) and len(folds) > 1:
        outer = n_cores if n_jobs == -1 else int(n_jobs)
        outer = max(1, min(outer, len(folds), n_cores))
    # iter54: default lets xgboost manage all cores via its own thread pool (inner=-1). iter53 A/B at
    # width 4000+10000 measured the iter4 oversubscription cap (n_cores // outer when outer > 1) as
    # 8-9% e2e SLOWER -- per-stage: reval +8%, refine +11%, trust +12% wall-clock loss; prefilter +2%
    # small win. xgboost's internal scheduler handles outer*inner > n_cores more efficiently than the
    # joblib-side cap on 8-core modern boxes. ``inner_n_jobs_cap=True`` restores legacy behaviour for
    # callers who measure regression on their HW.
    if outer > 1:
        inner = max(1, n_cores // outer) if inner_n_jobs_cap else -1
    else:
        inner = None

    def _one_fold(fold_id, tr_idx, va_idx):
        pf, bf, vf = _models_phi(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], fold_seeds[fold_id], inner_n_jobs=inner)
        _assert_additivity_and_base(pf, bf, fold_tag=f" fold {fold_id}")
        return fold_id, va_idx, pf, bf, vf

    if outer > 1:
        from joblib import Parallel, delayed

        fold_results = Parallel(n_jobs=outer, prefer="threads")(delayed(_one_fold)(fid, tr, va) for fid, (tr, va) in enumerate(folds))
    else:
        iter_folds = folds
        if tqdm_desc:
            from pyutilz.system import tqdmu

            iter_folds = tqdmu(folds, desc=tqdm_desc)
        fold_results = [_one_fold(fid, tr, va) for fid, (tr, va) in enumerate(iter_folds)]

    # ``fold_results`` may arrive out of order when ``outer > 1`` (joblib threads scatter); each tuple
    # carries its fold id explicitly so per_fold_mean[fid] keeps the deterministic split-order mapping.
    for fold_id, va_idx, pf, bf, vf in fold_results:
        phi[va_idx] = pf
        base[va_idx] = bf
        if return_variance:
            phi_var[va_idx] = vf
        if return_per_fold_phi_mean:
            per_fold_mean[fold_id] = np.abs(pf).mean(axis=0)

    out_tail = []
    if return_variance:
        out_tail.append(phi_var)
    if return_per_fold_phi_mean:
        out_tail.append(per_fold_mean)
    if out_tail:
        return _maybe_store((phi, base, y.astype(np.float64), *out_tail))
    return _maybe_store((phi, base, y.astype(np.float64)))
