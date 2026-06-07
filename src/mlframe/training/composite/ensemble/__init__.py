"""Cross-target ensemble + OOF utilities: CompositeCrossTargetEnsemble (stack/weighted/mean strategies with validation gate), compute_oof_holdout_predictions, derive_seeds (sha256-stable subseed derivation), detect_gpu_in_use, env_signature. Split out of composite.py to keep ensemble concerns separate from discovery; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import hashlib
import logging
import math
import warnings
from collections import OrderedDict
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
# ENS-Low-7: hoist sklearn.linear_model imports out of the predict()
# hot path so an inference round-trip does not pay the import cost on
# the first call (previously imported inside CompositeCrossTargetEnsemble
# fitting helpers; predict already touched the cached Ridge class but the
# cold-import cost still showed up in profiling).
from sklearn.linear_model import Ridge, ElasticNetCV, RidgeCV

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """ENS-P2-6: prefer explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from ..estimator import CompositeTargetEstimator
from ...composite_post_shim import PrePipelinePredictShim
from ..transforms import get_transform


def _unwrap_shim(model: Any) -> tuple[Any, Any]:
    """Return ``(inner, pre_pipeline)`` for a possibly shim-wrapped component.

    For ``PrePipelinePredictShim`` returns its fitted ``pre_pipeline`` and the inner model (which may itself be a ``CompositeTargetEstimator``). For any other estimator the pre_pipeline is ``None`` and the model is returned unchanged. Used by the OOF refit path to detect shim wrappers and apply ``pre_pipeline.transform`` to the stack/holdout slices before refitting -- without this, the OOF path would call ``sklearn.clone(shim)`` which (pre-fix) raised ``Cannot clone object ... not a scikit-learn estimator`` for every shim-wrapped component.
    """
    if isinstance(model, PrePipelinePredictShim):
        return model.model, model.pre_pipeline
    return model, None


def _transform_via(pp: Any, X: Any) -> Any:
    """Apply a fitted pre_pipeline to ``X`` or return ``X`` unchanged when ``pp is None``.

    Falls back to ``X`` on ``transform`` exceptions so the inner ``fit``/``predict`` raises the more descriptive boundary-mismatch error (mirrors ``PrePipelinePredictShim._transform``).
    """
    if pp is None:
        return X
    try:
        return pp.transform(X)
    except Exception:
        return X

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CompositeCrossTargetEnsemble
# ----------------------------------------------------------------------


def derive_seeds(random_state: int, components: Sequence[str]) -> dict[str, int]:
    """Derive deterministic per-component seeds from a master seed.

    Uses sha256 truncation to keep the values stable across Python /
    numpy versions (no dependence on hash() salt randomisation). The
    returned dict maps each component name to a 32-bit unsigned int.

    Why this exists. Discovery has several internal sources of
    randomness (MI sampling, tiny-model CV split, OOF holdout split,
    bootstrap CI). Threading the same ``random_state`` through every
    one of them creates correlation: if the master seed produces an
    "easy" MI sample it tends to also produce an "easy" CV split.
    Sub-seeds break the correlation while keeping reproducibility:
    same master seed -> same sub-seeds -> same downstream randomness.
    """
    import struct
    out: dict[str, int] = {}
    for c in components:
        h = hashlib.sha256(f"{random_state}::{c}".encode()).digest()
        out[c] = struct.unpack("<I", h[:4])[0]
    return out


def detect_gpu_in_use(mlframe_models: Sequence[str]) -> list[str]:
    """Return list of model families that may be using GPU.

    Best-effort detection: imports each library only if it appears in
    ``mlframe_models`` and probes for GPU availability via the
    library's standard health-check API. Returns the subset that has
    GPU detected. Returns empty list when no GPU library is in use.

    Used by the suite to emit a one-shot warning when composite mode
    is combined with GPU training: GPU non-determinism is amplified
    by K composite-model fits and can surface as ensemble weight
    drift across runs even when ``random_state`` is fixed.
    """
    detected: list[str] = []
    families = {str(m).lower() for m in mlframe_models}
    if any(f in families for f in ("lgb", "lightgbm")):
        try:
            import lightgbm as lgb  # noqa: F401
            # LightGBM doesn't have a portable "is GPU available"
            # check; we infer from the user's stated intent only.
            # Conservative: skip the warning if we can't tell.
        except ImportError:
            pass
    if any(f in families for f in ("xgb", "xgboost")):
        try:
            import xgboost as xgb
            try:
                # XGBoost build info is the canonical "GPU available?"
                # signal post-2.x.
                bi = xgb.build_info()
                if isinstance(bi, dict) and bi.get("USE_CUDA", False):
                    detected.append("xgboost")
            except Exception:
                pass
        except ImportError:
            pass
    if any(f in families for f in ("cb", "catboost")):
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                detected.append("catboost")
        except Exception:
            pass
    return detected


def env_signature() -> dict[str, str | None]:
    """Snapshot of library versions relevant to composite-target
    discovery + serialisation. Stored on metadata so a pickle saved
    today can be reload-validated tomorrow against version drift.

    Returns ``None`` for any library not installed.
    """
    sig: dict[str, str | None] = {}
    for libname in ("numpy", "pandas", "polars", "sklearn", "lightgbm",
                    "xgboost", "catboost", "scipy", "dill"):
        try:
            mod = __import__(libname)
            sig[libname] = getattr(mod, "__version__", None)
        except Exception:
            sig[libname] = None
    return sig


def _is_monotone_nondecreasing(arr: np.ndarray) -> bool:
    """True iff arr is finite and weakly monotone non-decreasing.

    Used to auto-detect timestamp / time-index columns so the OOF helper
    can produce a time-respecting split (past-only train, future holdout)
    instead of a random shuffle that leaks the future into the past.
    """
    try:
        a = np.asarray(arr).ravel()
    except Exception:
        return False
    if a.size < 2:
        return False
    # Cast to float so timestamps / ints alike work; non-numeric -> False.
    try:
        af = a.astype(np.float64, copy=False)
    except Exception:
        return False
    if not np.all(np.isfinite(af)):
        return False
    return bool(np.all(np.diff(af) >= 0))


def _maybe_pass_sample_weight(
    fit_callable, X, y,
    sw: np.ndarray | None,
    eval_set: tuple | None = None,
):
    """Call ``fit_callable.fit(X, y[, sample_weight, eval_set])`` honouring whichever
    kwargs the inner estimator's fit signature exposes.

    Avoids hard-coding which inner estimators support sample_weight (CatBoost / LGB /
    sklearn all do; some custom shims may not).

    ``eval_set`` plumbed to fix the OOF-refit silent-drop pathology
    (observed in prod: LGBM clones with ``early_stopping_rounds`` callback
    attached but no eval data raised ``"For early stopping, at least one dataset
    and eval metric is required for evaluation"`` and were dropped from the
    cross-target ensemble -- ensemble RMSE worse than dummy).

    Falls back to the plain call on TypeError so missing-kwarg shims keep working.
    """
    import inspect as _inspect
    try:
        _sig = _inspect.signature(fit_callable.fit)
        _params = _sig.parameters
        _accepts_var_kw = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in _params.values()
        )
        _kwargs: dict = {}
        if sw is not None and (
            "sample_weight" in _params or _accepts_var_kw
        ):
            _kwargs["sample_weight"] = sw
        if eval_set is not None and (
            "eval_set" in _params or _accepts_var_kw
        ):
            # LightGBM expects a list of (X, y) tuples; XGBoost/CatBoost
            # accept either (X, y) tuple or list-of-tuples. Normalising to
            # list-of-tuples covers all three.
            _es = eval_set if isinstance(eval_set, list) else [eval_set]
            _kwargs["eval_set"] = _es
        if _kwargs:
            return fit_callable.fit(X, y, **_kwargs)
    except (TypeError, ValueError):
        pass
    # Retry with sample_weight alone if combined call failed
    if sw is not None:
        try:
            return fit_callable.fit(X, y, sample_weight=sw)
        except (TypeError, ValueError):
            pass
    return fit_callable.fit(X, y)


def _carve_inner_eval_split(
    X, y, *, frac: float = 0.1, random_state: int | None = 0,
    group_ids: np.ndarray | None = None,
):
    """Return ``(X_fit, y_fit, X_eval, y_eval)`` for OOF refits that need
    an eval_set to satisfy early-stopping callbacks on cloned boosters.

    When ``group_ids`` is supplied, carves whole groups into the eval
    slice (no group spans both fit and eval). Required for honest OOF on
    group-aware splits: rows from the same group/user/session in both fit
    and eval make early-stopping see same-group leakage, model
    under-stops, OOF RMSE artificially degrades (observed in prod:
    val_RMSE 10.64 from direct fit vs honest-OOF 13.34 from group-blind
    carve, +25% degradation that wrongly triggered the AR1 failsafe).

    Without ``group_ids`` falls back to the deterministic last-``frac``
    tail split (mirrors val_placement='forward' for temporal splits).
    For row counts below 1000 the split is skipped (returns
    ``X, y, None, None``) - early-stopping at that scale is noise."""
    try:
        n = len(y)
    except TypeError:
        return X, y, None, None
    if n < 1000:
        return X, y, None, None
    n_eval_target = max(100, int(frac * n))
    if n_eval_target >= n - 100:
        return X, y, None, None
    if group_ids is not None:
        g = np.asarray(group_ids)
        if g.shape[0] == n:
            uniq, first_idx = np.unique(g, return_index=True)
            if uniq.size >= 4:
                order = np.argsort(first_idx)
                groups_in_order = uniq[order]
                rng = np.random.default_rng(random_state)
                shuffled = rng.permutation(groups_in_order)
                _, _, counts_orig = np.unique(g, return_index=True, return_counts=True)
                idx_for_group = {gid: i for i, gid in enumerate(uniq)}
                cumulative = 0
                eval_groups: list = []
                for gid in shuffled:
                    eval_groups.append(gid)
                    cumulative += int(counts_orig[idx_for_group[gid]])
                    if cumulative >= n_eval_target:
                        break
                if 0 < cumulative < n - 100 and len(eval_groups) < uniq.size:
                    eval_set = set(eval_groups.tolist() if hasattr(eval_groups, "tolist") else list(eval_groups))
                    eval_mask = np.isin(g, list(eval_set))
                    fit_mask = ~eval_mask
                    return _slice_rows(X, fit_mask), y[fit_mask], _slice_rows(X, eval_mask), y[eval_mask]
    cut = n - n_eval_target
    if hasattr(X, "iloc"):
        return X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:]
    if hasattr(X, "select") and hasattr(X, "slice"):
        return X.slice(0, cut), y[:cut], X.slice(cut, n_eval_target), y[cut:]
    return X[:cut], y[:cut], X[cut:], y[cut:]


def _slice_rows(X, mask: np.ndarray):
    """Index rows of X (pandas / polars / ndarray) by a boolean mask."""
    if hasattr(X, "iloc"):
        return X.iloc[mask].reset_index(drop=True) if hasattr(X.iloc[mask], "reset_index") else X.iloc[mask]
    if hasattr(X, "filter") and hasattr(X, "slice"):
        return X.filter(pl.Series(mask))
    return X[mask]


# Module-level memo cache for compute_oof_holdout_predictions. Keyed by (cache_key, kfold, random_state).
# The cache_key argument is opaque -- callers pass a hashable tuple summarising the (component, X, y, sw) identity.
#
# Intentionally UNWIRED on the suite path: the cross-target ensemble builder computes each target's OOF exactly
# once per suite call on a fresh train frame, so there is no reuse to capture, and a content key on a TB-scale
# frame is forbidden by the RAM rule. The cache exists for EXTERNAL callers that legitimately repeat an identical
# OOF call with a cheap stable key (huge hit-speedup; see _benchmarks/bench_oof_cache_reuse.py). Do not wire a
# frame-hash key into the suite to "use" it -- that buys zero hits at the cost of a forbidden hash.
#
# C-P2-2: DO NOT include ``id(train_X)`` in the cache_key. Python recycles object IDs across the
# lifecycle of a long-lived suite, so two frames with disjoint content can end up sharing the same
# id and a stale cache entry can mask the swap. Prefer a content fingerprint (the suite already
# computes one via ``pipeline_cache.fingerprint_df`` /``_hash_frame``) or a stable per-session
# token. The historical comment that recommended ``id(train_X)`` was wrong; callers passing such
# a key get cross-contamination across re-runs.
#
# 16-entry LRU; once full the least-recently-USED entry is evicted on insertion. Stays in-process;
# cleared at interpreter shutdown. ``OrderedDict.move_to_end`` on cache hits keeps the eviction
# order driven by access (true LRU), not just insertion order (which would degrade to FIFO).
_OOF_HOLDOUT_CACHE: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray, list[str]]]" = OrderedDict()
_OOF_HOLDOUT_CACHE_CAP = 16


def _oof_cache_get(key: tuple):
    if key not in _OOF_HOLDOUT_CACHE:
        return None
    _OOF_HOLDOUT_CACHE.move_to_end(key)
    return _OOF_HOLDOUT_CACHE[key]


def _oof_cache_put(key: tuple, value: tuple) -> None:
    if key in _OOF_HOLDOUT_CACHE:
        _OOF_HOLDOUT_CACHE.move_to_end(key)
        _OOF_HOLDOUT_CACHE[key] = value
        return
    if len(_OOF_HOLDOUT_CACHE) >= _OOF_HOLDOUT_CACHE_CAP:
        _OOF_HOLDOUT_CACHE.popitem(last=False)
    _OOF_HOLDOUT_CACHE[key] = value


def _compute_oof_with_external_holdout(
    *,
    # Slice-stable ES (mlframe.training.SliceStableESConfig) is NOT propagated into the inner
    # OOF refit loop: this function builds its own per-fold ``eval_set`` via
    # ``_carve_eval_set_from_train_with_groups`` and a single (X_holdout, y_holdout) pair, which
    # is incompatible with the multi-eval-set / per-shard registration path slice-ES needs.
    # Callers that want robust ES inside OOF refit should use full-K-fold CV with an outer
    # selector (see ``_cv_aggregation.aggregate_fold_scores``) instead.
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    train_X: Any,
    y_train_full: np.ndarray,
    base_train_full_per_spec: dict[str, np.ndarray],
    external_holdout_X: Any,
    external_holdout_y: np.ndarray,
    external_holdout_base_per_spec: dict[str, np.ndarray],
    sample_weight: np.ndarray | None,
    full_key: tuple | None,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit each component clone on full train, predict on caller-
    supplied external holdout (typically the suite's val split).

    Mirrors the per-component branch in :func:`compute_oof_holdout_predictions`
    but skips the internal train/holdout slicing.
    """
    y_train_full = y_train_full.astype(np.float64)
    holdout_cols: list[np.ndarray] = []
    surviving_names: list[str] = []
    for model, name, spec in zip(
        component_models, component_names, component_specs,
    ):
        try:
            inner, pp = _unwrap_shim(model)
            X_stack_t = _transform_via(pp, train_X)
            X_holdout_t = _transform_via(pp, external_holdout_X)
            if isinstance(inner, CompositeTargetEstimator):
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(
                    spec["base_column"],
                )
                if base_full is None:
                    raise ValueError(
                        f"missing base column '{spec['base_column']}' "
                        "for external-holdout OOF (train side)"
                    )
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_train_full, base_full)
                if valid.sum() < 10:
                    raise ValueError(
                        "too few valid rows after domain filter"
                    )
                t_train = transform.forward(
                    y_train_full[valid], base_full[valid],
                    spec["fitted_params"],
                )
                inner_clone = clone(inner.estimator_)
                if isinstance(X_stack_t, pd.DataFrame):
                    X_train_valid = X_stack_t.iloc[valid].reset_index(
                        drop=True,
                    )
                elif _is_polars_df(X_stack_t):
                    X_train_valid = X_stack_t.filter(pl.Series(valid))
                else:
                    X_train_valid = X_stack_t[valid]
                _sw_train_valid = (
                    None if sample_weight is None
                    else sample_weight[valid]
                )
                _group_for_valid = None
                if group_ids is not None:
                    try:
                        _g_arr = np.asarray(group_ids)
                        if _g_arr.shape[0] == valid.shape[0]:
                            _group_for_valid = _g_arr[valid]
                    except (TypeError, IndexError):
                        _group_for_valid = None
                _X_fit_c, _t_fit_c, _X_ev_c, _t_ev_c = (
                    _carve_inner_eval_split(
                        X_train_valid, t_train, random_state=0,
                        group_ids=_group_for_valid,
                    )
                )
                _eval_set_c = (
                    (_X_ev_c, _t_ev_c) if _X_ev_c is not None else None
                )
                _sw_fit_c = (
                    _sw_train_valid[:len(_t_fit_c)]
                    if _sw_train_valid is not None else None
                )
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_c, _t_fit_c, _sw_fit_c,
                    eval_set=_eval_set_c,
                )
                _extra = tuple(spec.get("extra_base_columns") or ())
                _base_columns = (
                    (spec["base_column"], *_extra) if _extra else None
                )
                wrapped = CompositeTargetEstimator.from_fitted_inner(
                    fitted_inner=inner_clone,
                    transform_name=spec["transform_name"],
                    base_column=spec["base_column"],
                    base_columns=_base_columns,
                    transform_fitted_params=spec["fitted_params"],
                    y_train=y_train_full[valid],
                )
                preds = wrapped.predict(X_holdout_t)
            else:
                inner_clone = clone(inner)
                _X_fit_r, _y_fit_r, _X_ev_r, _y_ev_r = (
                    _carve_inner_eval_split(
                        X_stack_t, y_train_full, random_state=0,
                        group_ids=group_ids,
                    )
                )
                _eval_set_r = (
                    (_X_ev_r, _y_ev_r) if _X_ev_r is not None else None
                )
                _sw_fit_r = (
                    sample_weight[:len(_y_fit_r)]
                    if sample_weight is not None else None
                )
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_r, _y_fit_r, _sw_fit_r,
                    eval_set=_eval_set_r,
                )
                preds = inner_clone.predict(X_holdout_t)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if preds.shape[0] != external_holdout_y.shape[0]:
                raise ValueError(
                    f"component '{name}' predicted "
                    f"{preds.shape[0]} rows but external holdout has "
                    f"{external_holdout_y.shape[0]}"
                )
            if not np.all(np.isfinite(preds)):
                raise ValueError("non-finite holdout predictions")
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] external-holdout OOF "
                "refit failed for component '%s': %s. Excluded from "
                "ensemble weights.", name, exc,
            )
            continue
    _surviving_n = len(surviving_names)
    _total_n = len(component_names)
    if _surviving_n < _total_n:
        _dropped = [
            n for n in component_names if n not in set(surviving_names)
        ]
        logger.info(
            "compute_oof_holdout_predictions (external-holdout): built "
            "OOF matrix with %d of %d components (dropped %d: %s).",
            _surviving_n, _total_n, _total_n - _surviving_n, _dropped,
        )
    if not holdout_cols:
        _empty = (np.zeros((0, 0)), np.zeros(0), [])
        if full_key is not None:
            _oof_cache_put(full_key, _empty)
        return _empty
    _final = (
        np.column_stack(holdout_cols),
        external_holdout_y,
        surviving_names,
    )
    if full_key is not None:
        _oof_cache_put(full_key, _final)
    return _final


def compute_oof_holdout_predictions(
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    train_X: Any,
    y_train_full: np.ndarray,
    base_train_full_per_spec: dict[str, np.ndarray],
    holdout_frac: float,
    random_state: int,
    time_ordering: np.ndarray | None = None,
    kfold: int = 1,
    sample_weight: np.ndarray | None = None,
    cache_key: tuple | None = None,
    external_holdout_X: Any | None = None,
    external_holdout_y: np.ndarray | None = None,
    external_holdout_base_per_spec: dict[str, np.ndarray] | None = None,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute honest holdout predictions for each component.

    Approach: take a ``holdout_frac`` slice of train, re-fit a clone of
    each component's inner on the remaining (1-holdout_frac) rows, and
    predict on the held-out slice. For wrapped composite-target components
    we re-apply the spec's transform on the same stack_train slice to get
    T values, train the inner clone on (X_stack_train, T_stack_train),
    then wrap using ``CompositeTargetEstimator.from_fitted_inner`` and
    predict in y-scale on stack_holdout. For raw-target components the
    inner clone is fit directly on (X_stack_train, y_stack_train).

    Split strategy:

    - **External holdout** (preferred when ``external_holdout_X`` is
      supplied): fit each component clone on the FULL train, predict
      on the caller-provided external frame (the suite's val split).
      Eliminates the train-tail-vs-test distribution mismatch that
      biases NNLS weights on group-aware splits of strong-AR targets
      (observed in prod: train-tail lag_predict RMSE 15.18 vs test
      11.58 - NNLS underweights the dominant baseline because it
      looks bad on the train-tail). Caller is responsible for
      providing the parallel base columns via
      ``external_holdout_base_per_spec`` for composite components.
    - ``time_ordering`` provided and monotone-non-decreasing OR
      ``time_ordering`` is ``None`` but rows are otherwise detected to
      be time-ordered: take the trailing ``holdout_frac`` slice as the
      holdout (past-only train, future holdout) -- the analogue of
      a single ``sklearn.model_selection.TimeSeriesSplit`` fold.
    - Otherwise: random shuffle by ``random_state`` (legacy behaviour).

    Parameters
    ----------
    kfold
        ENS-Low-1: when > 1, perform K-fold OOF prediction instead of a single
        holdout slice. Each fold contributes its hold-out predictions; the
        concatenated (n_train, K) matrix is returned in the natural row order.
        ``kfold=1`` preserves the legacy single-split behaviour. Random-shuffle
        only (time-aware K-fold remains the single-split trailing slice).

    Returns
    -------
    - ``holdout_preds_matrix``: y-scale predictions; shape
      ``(n_holdout, K)`` for kfold=1, ``(n_train, K)`` for kfold>1 random.
    - ``y_holdout``: y-scale targets aligned row-for-row.
    - ``surviving_names``: subset of ``component_names`` whose
      re-fit succeeded (any failures are dropped from the matrix
      so callers can re-align weight vectors).
    """
    from sklearn.model_selection import train_test_split

    n_train = len(y_train_full)
    if n_train < 50 or holdout_frac <= 0 or holdout_frac >= 1:
        # C-Low-1: shape consistency across the three empty-return paths. ``surviving_names=[]``
        # means zero components survived; both axes are zero. Downstream consumers that probed
        # ``.shape[1]`` against ``len(surviving_names)`` were correct; consumers that probed
        # ``.shape[1]`` against ``len(component_models)`` had been silently looking at a
        # non-empty K dimension with zero rows. Standardise to ``(0, 0)`` everywhere.
        return np.zeros((0, 0)), np.zeros(0), []

    # OOF-K-NOT-CACHED: when the caller supplied a cache_key we look up the (key, kfold, rs)
    # tuple. Cache hit is bit-identical with a previous call -- same components, same X, same y,
    # same fold strategy; semantics for caller stay unchanged.
    _full_key = None
    if cache_key is not None:
        _full_key = (cache_key, int(kfold), int(random_state))
        _hit = _oof_cache_get(_full_key)
        if _hit is not None:
            logger.debug("compute_oof_holdout_predictions: cache HIT for key=%r", _full_key)
            return _hit

    # C-P1-3: warn loudly when a caller asks for time-aware K-fold (kfold>1 AND time_ordering supplied) so
    # they know the suite is silently downgrading to a single trailing-slice holdout. Forward-walking K-fold
    # is semantically ambiguous (past-only training rules out shuffled K-fold) and intentionally unsupported;
    # callers who got back ONE OOF slice instead of K used to have no signal that their kfold request was
    # ignored, biasing every downstream "average across folds" report.
    if int(kfold) > 1 and time_ordering is not None:
        logger.warning(
            "compute_oof_holdout_predictions: kfold=%d AND time_ordering supplied; K-fold is "
            "incompatible with time-aware semantics. Downgrading to a single trailing-slice "
            "holdout. Pass time_ordering=None to enable shuffled K-fold OOF.",
            int(kfold),
        )
    if int(kfold) > 1 and time_ordering is None:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=int(kfold), shuffle=True, random_state=int(random_state))
        oof_preds_by_name: dict[str, np.ndarray] = {}
        survived_set: set[str] | None = None
        for fold_train_idx, fold_holdout_idx in kf.split(np.arange(n_train)):
            fold_frac = float(fold_holdout_idx.size) / float(n_train)
            # Sub-frame views by index. Reuse the single-split path by
            # creating a synthetic ``time_ordering`` that forces the
            # specific fold_holdout slice -- simpler: inline the fit/predict
            # for this fold.
            if _is_polars_df(train_X):
                fold_train_mask = np.zeros(n_train, dtype=bool)
                fold_train_mask[fold_train_idx] = True
                X_stack = train_X.filter(pl.Series(fold_train_mask))
                X_holdout = train_X.filter(pl.Series(~fold_train_mask))
            elif isinstance(train_X, pd.DataFrame):
                X_stack = train_X.iloc[fold_train_idx].reset_index(drop=True)
                X_holdout = train_X.iloc[fold_holdout_idx].reset_index(drop=True)
            else:
                X_stack = train_X[fold_train_idx]
                X_holdout = train_X[fold_holdout_idx]
            y_stack = y_train_full[fold_train_idx].astype(np.float64)
            y_holdout_fold = y_train_full[fold_holdout_idx].astype(np.float64)
            fold_cols: dict[str, np.ndarray] = {}
            for model, name, spec in zip(component_models, component_names, component_specs):
                try:
                    inner, pp = _unwrap_shim(model)
                    X_stack_t = _transform_via(pp, X_stack)
                    X_holdout_t = _transform_via(pp, X_holdout)
                    if isinstance(inner, CompositeTargetEstimator):
                        if spec is None:
                            raise ValueError("composite component with no spec")
                        base_full = base_train_full_per_spec.get(spec["base_column"])
                        if base_full is None:
                            raise ValueError(
                                f"missing base column '{spec['base_column']}'"
                            )
                        base_stack = base_full[fold_train_idx]
                        transform = get_transform(spec["transform_name"])
                        valid = transform.domain_check(y_stack, base_stack)
                        if valid.sum() < 10:
                            raise ValueError("too few valid rows after domain filter")
                        t_stack = transform.forward(
                            y_stack[valid], base_stack[valid], spec["fitted_params"],
                        )
                        inner_clone = clone(inner.estimator_)
                        if isinstance(X_stack_t, pd.DataFrame):
                            X_stack_valid = X_stack_t.iloc[valid].reset_index(drop=True)
                        elif _is_polars_df(X_stack_t):
                            X_stack_valid = X_stack_t.filter(pl.Series(valid))
                        else:
                            X_stack_valid = X_stack_t[valid]
                        _sw_stack_valid = None if sample_weight is None else sample_weight[fold_train_idx][valid]
                        _maybe_pass_sample_weight(inner_clone, X_stack_valid, t_stack, _sw_stack_valid)
                        # Multi-base parity with _phase_composite_post: pass
                        # the full base_columns tuple so predict reconstructs
                        # the K-column base matrix matching the K alphas.
                        _extra = tuple(spec.get("extra_base_columns") or ())
                        _base_columns = (
                            (spec["base_column"], *_extra) if _extra else None
                        )
                        wrapped = CompositeTargetEstimator.from_fitted_inner(
                            fitted_inner=inner_clone,
                            transform_name=spec["transform_name"],
                            base_column=spec["base_column"],
                            base_columns=_base_columns,
                            transform_fitted_params=spec["fitted_params"],
                            y_train=y_stack[valid],
                        )
                        preds = wrapped.predict(X_holdout_t)
                    else:
                        inner_clone = clone(inner)
                        _sw_stack = None if sample_weight is None else sample_weight[fold_train_idx]
                        _group_for_fold = None
                        if group_ids is not None:
                            try:
                                _g_arr = np.asarray(group_ids)
                                if _g_arr.shape[0] >= int(np.max(fold_train_idx)) + 1:
                                    _group_for_fold = _g_arr[fold_train_idx]
                            except (TypeError, IndexError, ValueError):
                                _group_for_fold = None
                        _X_fit, _y_fit, _X_ev, _y_ev = _carve_inner_eval_split(
                            X_stack_t, y_stack, random_state=int(random_state),
                            group_ids=_group_for_fold,
                        )
                        _eval_set = (_X_ev, _y_ev) if _X_ev is not None else None
                        _sw_fit = _sw_stack[:len(_y_fit)] if _sw_stack is not None else None
                        _maybe_pass_sample_weight(
                            inner_clone, _X_fit, _y_fit, _sw_fit,
                            eval_set=_eval_set,
                        )
                        preds = inner_clone.predict(X_holdout_t)
                    preds = np.asarray(preds).reshape(-1).astype(np.float64)
                    if not np.all(np.isfinite(preds)):
                        raise ValueError("non-finite holdout predictions")
                    fold_cols[name] = preds
                except Exception as exc:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] kfold OOF refit failed "
                        "for component '%s' (kfold=%d): %s. Excluded.",
                        name, int(kfold), exc,
                    )
                    continue
            if survived_set is None:
                survived_set = set(fold_cols.keys())
            else:
                survived_set &= set(fold_cols.keys())
            for nm, preds in fold_cols.items():
                buf = oof_preds_by_name.setdefault(nm, np.full(n_train, np.nan, dtype=np.float64))
                buf[fold_holdout_idx] = preds
        if not oof_preds_by_name or not survived_set:
            # C-Low-1: shape consistency -- match the (0, 0) tiny-data short-circuit. No components survived.
            _empty = (np.zeros((0, 0)), np.zeros(0), [])
            if _full_key is not None:
                _oof_cache_put(_full_key, _empty)
            return _empty
        surviving_names = [n for n in component_names if n in survived_set]
        cols = [oof_preds_by_name[n] for n in surviving_names]
        oof_matrix = np.column_stack(cols)
        # Drop rows with any NaN (folds that lost every component) - rare.
        finite_rows = np.all(np.isfinite(oof_matrix), axis=1)
        _result = (
            oof_matrix[finite_rows],
            y_train_full.astype(np.float64)[finite_rows],
            surviving_names,
        )
        if _full_key is not None:
            _oof_cache_put(_full_key, _result)
        return _result

    # External honest holdout (caller-supplied val frame). Skip the
    # train-tail split entirely: fit each component clone on the FULL
    # train, predict on the external frame, return the parallel y
    # column the caller supplied. Defends against AR(1) train-tail
    # distribution mismatch.
    if (external_holdout_X is not None
            and external_holdout_y is not None
            and len(external_holdout_y) > 0):
        return _compute_oof_with_external_holdout(
            component_models=component_models,
            component_names=component_names,
            component_specs=component_specs,
            train_X=train_X,
            y_train_full=y_train_full,
            base_train_full_per_spec=base_train_full_per_spec,
            external_holdout_X=external_holdout_X,
            external_holdout_y=np.asarray(external_holdout_y, dtype=np.float64),
            external_holdout_base_per_spec=(
                external_holdout_base_per_spec or {}
            ),
            sample_weight=sample_weight,
            full_key=_full_key,
            group_ids=group_ids,
        )

    # Decide whether to do a time-aware split. Only the EXPLICIT ``time_ordering`` signal (the suite threads
    # ctx.timestamps here) flips to a trailing-slice holdout. The old behaviour also probed every base column and
    # auto-switched if ANY was monotone -- a false positive on sorted-but-non-temporal bases (sorted ids, binned
    # features) that silently turned a random holdout into a trailing slice and changed the OOF leakage profile.
    # Random shuffle is the safe default when no explicit time signal is given.
    use_time_split = False
    if time_ordering is not None:
        use_time_split = _is_monotone_nondecreasing(time_ordering)
        if use_time_split:
            logger.info(
                "composite OOF: time_ordering signal is monotone non-decreasing; using trailing-slice holdout instead of random shuffle."
            )

    n_holdout = max(int(round(n_train * holdout_frac)), 1)
    if use_time_split:
        cutoff = n_train - n_holdout
        train_idx = np.arange(cutoff, dtype=np.int64)
        holdout_idx = np.arange(cutoff, n_train, dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n_train)
        holdout_idx = np.sort(perm[:n_holdout])
        train_idx = np.sort(perm[n_holdout:])

    # Subset X. Branch on type so we don't pull pandas APIs on
    # polars frames.
    if _is_polars_df(train_X):
        train_mask = np.zeros(n_train, dtype=bool)
        train_mask[train_idx] = True
        X_stack = train_X.filter(pl.Series(train_mask))
        X_holdout = train_X.filter(pl.Series(~train_mask))
    elif isinstance(train_X, pd.DataFrame):
        X_stack = train_X.iloc[train_idx].reset_index(drop=True)
        X_holdout = train_X.iloc[holdout_idx].reset_index(drop=True)
    else:
        X_stack = train_X[train_idx]
        X_holdout = train_X[holdout_idx]

    y_stack = y_train_full[train_idx].astype(np.float64)
    y_holdout = y_train_full[holdout_idx].astype(np.float64)

    # Group-aware inner eval-carve parity with the raw branch + kfold path: subset the caller-supplied group_ids
    # to the stack rows so neither the composite nor the raw inner-eval carve splits a group across fit/eval
    # (within-group leakage under-stops the booster and degrades OOF RMSE on group-aware splits).
    _group_stack = None
    if group_ids is not None:
        try:
            _g_arr = np.asarray(group_ids)
            if _g_arr.shape[0] >= int(np.max(train_idx)) + 1:
                _group_stack = _g_arr[train_idx]
        except (TypeError, IndexError, ValueError):
            _group_stack = None

    holdout_cols: list[np.ndarray] = []
    surviving_names: list[str] = []
    for model, name, spec in zip(component_models, component_names, component_specs):
        try:
            inner, pp = _unwrap_shim(model)
            X_stack_t = _transform_via(pp, X_stack)
            X_holdout_t = _transform_via(pp, X_holdout)
            if isinstance(inner, CompositeTargetEstimator):
                # Composite-target wrapper. Re-fit the inner on
                # stack_train T values, then re-wrap and predict.
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(spec["base_column"])
                if base_full is None:
                    raise ValueError(
                        f"missing base column '{spec['base_column']}' for OOF"
                    )
                base_stack = base_full[train_idx]
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_stack, base_stack)
                # Drop invalid rows from stack_train; the inner will
                # train only on rows where T is finite.
                if valid.sum() < 10:
                    raise ValueError("too few valid rows after domain filter")
                t_stack = transform.forward(
                    y_stack[valid], base_stack[valid], spec["fitted_params"],
                )
                inner_clone = clone(inner.estimator_)
                if isinstance(X_stack_t, pd.DataFrame):
                    X_stack_valid = X_stack_t.iloc[valid].reset_index(drop=True)
                elif _is_polars_df(X_stack_t):
                    X_stack_valid = X_stack_t.filter(pl.Series(valid))
                else:
                    X_stack_valid = X_stack_t[valid]
                _sw_stack_valid = None if sample_weight is None else sample_weight[train_idx][valid]
                _group_stack_valid = None
                if _group_stack is not None and _group_stack.shape[0] == valid.shape[0]:
                    _group_stack_valid = _group_stack[valid]
                _X_fit_c, _t_fit_c, _X_ev_c, _t_ev_c = _carve_inner_eval_split(
                    X_stack_valid, t_stack, random_state=0,
                    group_ids=_group_stack_valid,
                )
                _eval_set_c = (_X_ev_c, _t_ev_c) if _X_ev_c is not None else None
                _sw_fit_c = (
                    _sw_stack_valid[:len(_t_fit_c)]
                    if _sw_stack_valid is not None else None
                )
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_c, _t_fit_c, _sw_fit_c,
                    eval_set=_eval_set_c,
                )
                # Multi-base parity: same fix as the kfold OOF branch
                # above. Without base_columns, predict reconstructs only
                # the primary base column and trips the K-alphas shape check.
                _extra = tuple(spec.get("extra_base_columns") or ())
                _base_columns = (
                    (spec["base_column"], *_extra) if _extra else None
                )
                wrapped = CompositeTargetEstimator.from_fitted_inner(
                    fitted_inner=inner_clone,
                    transform_name=spec["transform_name"],
                    base_column=spec["base_column"],
                    base_columns=_base_columns,
                    transform_fitted_params=spec["fitted_params"],
                    y_train=y_stack[valid],
                )
                preds = wrapped.predict(X_holdout_t)
            else:
                # Raw-target component. Re-fit the inner on
                # (X_stack, y_stack) and predict on X_holdout.
                inner_clone = clone(inner)
                _sw_stack = None if sample_weight is None else sample_weight[train_idx]
                _X_fit_r, _y_fit_r, _X_ev_r, _y_ev_r = _carve_inner_eval_split(
                    X_stack_t, y_stack, random_state=0,
                    group_ids=_group_stack,
                )
                _eval_set_r = (_X_ev_r, _y_ev_r) if _X_ev_r is not None else None
                _sw_fit_r = (
                    _sw_stack[:len(_y_fit_r)] if _sw_stack is not None else None
                )
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_r, _y_fit_r, _sw_fit_r,
                    eval_set=_eval_set_r,
                )
                preds = inner_clone.predict(X_holdout_t)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if not np.all(np.isfinite(preds)):
                # NaN preds on holdout -- exclude from ensemble.
                raise ValueError("non-finite holdout predictions")
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] OOF refit failed for component "
                "'%s': %s. Excluded from ensemble weights.", name, exc,
            )
            continue

    # C-P2-4: summary log so operators can see "ensemble built with N of K components" at INFO
    # without grepping per-component WARN lines. ``component_names`` is the full caller-supplied
    # list; ``surviving_names`` is the subset whose refit succeeded.
    _surviving_n = len(surviving_names)
    _total_n = len(component_names)
    if _surviving_n < _total_n:
        _dropped = [n for n in component_names if n not in set(surviving_names)]
        logger.info(
            "compute_oof_holdout_predictions: built OOF matrix with %d of %d components "
            "(dropped %d: %s). Per-component drop reasons logged at WARN above.",
            _surviving_n, _total_n, _total_n - _surviving_n, _dropped,
        )
    if not holdout_cols:
        # C-Low-1: shape consistency -- match the tiny-data + kfold short-circuits.
        _empty = (np.zeros((0, 0)), np.zeros(0), [])
        if _full_key is not None:
            _oof_cache_put(_full_key, _empty)
        return _empty
    _final = (np.column_stack(holdout_cols), y_holdout, surviving_names)
    if _full_key is not None:
        _oof_cache_put(_full_key, _final)
    return _final


# Wave 101 (2026-05-21): CompositeCrossTargetEnsemble class (~710 lines)
# moved to sibling _composite_cross_target_ensemble.py to drop this file
# below the 1k-line monolith threshold. Re-exported below so existing
# callers (`from mlframe.training.composite import CompositeCrossTargetEnsemble`)
# keep working.
from ._cross_target import CompositeCrossTargetEnsemble  # noqa: F401, E402

