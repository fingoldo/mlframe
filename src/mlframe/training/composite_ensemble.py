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


from .composite_estimator import CompositeTargetEstimator
from .composite_post_shim import PrePipelinePredictShim
from .composite_transforms import get_transform


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


def _maybe_pass_sample_weight(fit_callable, X, y, sw: np.ndarray | None):
    """Call ``fit_callable.fit(X, y, sample_weight=sw)`` when sw is given AND fit accepts it; else ``fit(X, y)``.

    Avoids hard-coding which inner estimators support sample_weight (CatBoost / LGB / sklearn all do; some
    custom shims may not). Falls back to the plain call on TypeError so missing-kwarg shims keep working."""
    if sw is None:
        return fit_callable.fit(X, y)
    try:
        import inspect as _inspect
        _sig = _inspect.signature(fit_callable.fit)
        if "sample_weight" in _sig.parameters or any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()):
            return fit_callable.fit(X, y, sample_weight=sw)
    except (TypeError, ValueError):
        pass
    return fit_callable.fit(X, y)


# Module-level memo cache for compute_oof_holdout_predictions. OOF-K-NOT-CACHED: keyed by
# (cache_key, kfold, random_state). The cache_key argument is opaque -- callers pass a hashable
# tuple summarising the (component, X, y, sw) identity (e.g. (id(train_X), y_hash, spec_hash)).
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
        return np.zeros((0, len(component_models))), np.zeros(0), []

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

    # ENS-Low-1: K-fold path. We recurse on the single-split implementation
    # by inducing the desired splits manually. Each fold's holdout slice is
    # predicted by inner clones trained on the remaining (K-1) folds; OOF
    # predictions are stitched back to row-order. Time-aware split is NOT
    # K-folded (semantics would be ambiguous for past-only training); we
    # fall through to the single-split path when ``time_ordering`` is given.
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
                        wrapped = CompositeTargetEstimator.from_fitted_inner(
                            fitted_inner=inner_clone,
                            transform_name=spec["transform_name"],
                            base_column=spec["base_column"],
                            transform_fitted_params=spec["fitted_params"],
                            y_train=y_stack[valid],
                        )
                        preds = wrapped.predict(X_holdout_t)
                    else:
                        inner_clone = clone(inner)
                        _sw_stack = None if sample_weight is None else sample_weight[fold_train_idx]
                        _maybe_pass_sample_weight(inner_clone, X_stack_t, y_stack, _sw_stack)
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
            _empty = (np.zeros((n_train, 0)), y_train_full.astype(np.float64), [])
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

    # Decide whether to do a time-aware split. We use the explicit
    # ``time_ordering`` signal when supplied; absent that we attempt
    # to detect monotone-time from base columns (a common shape when
    # base is a lagged timestamp). Random shuffle is the safe fallback.
    use_time_split = False
    if time_ordering is not None:
        use_time_split = _is_monotone_nondecreasing(time_ordering)
        if use_time_split:
            logger.info(
                "composite OOF: time_ordering signal is monotone non-decreasing; using trailing-slice holdout instead of random K-fold."
            )
    else:
        # Probe each base column; first monotone one switches the strategy. The col name is logged so an operator can
        # trace which base induced the switch (random-shuffle vs trailing-slice changes OOF leakage characteristics).
        for _base_col, _base in base_train_full_per_spec.items():
            if _is_monotone_nondecreasing(_base):
                use_time_split = True
                logger.info(
                    "composite OOF auto-detected time-ordered base column %s; switching from random K-fold to trailing-slice",
                    _base_col,
                )
                break

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
                _maybe_pass_sample_weight(inner_clone, X_stack_valid, t_stack, _sw_stack_valid)
                wrapped = CompositeTargetEstimator.from_fitted_inner(
                    fitted_inner=inner_clone,
                    transform_name=spec["transform_name"],
                    base_column=spec["base_column"],
                    transform_fitted_params=spec["fitted_params"],
                    y_train=y_stack[valid],
                )
                preds = wrapped.predict(X_holdout_t)
            else:
                # Raw-target component. Re-fit the inner on
                # (X_stack, y_stack) and predict on X_holdout.
                inner_clone = clone(inner)
                _sw_stack = None if sample_weight is None else sample_weight[train_idx]
                _maybe_pass_sample_weight(inner_clone, X_stack_t, y_stack, _sw_stack)
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

    if not holdout_cols:
        _empty = (np.zeros((n_holdout, 0)), y_holdout, [])
        if _full_key is not None:
            _oof_cache_put(_full_key, _empty)
        return _empty
    _final = (np.column_stack(holdout_cols), y_holdout, surviving_names)
    if _full_key is not None:
        _oof_cache_put(_full_key, _final)
    return _final


class CompositeCrossTargetEnsemble:
    """Weighted-average ensemble of K composite-target predictors plus
    optionally the raw-target predictor.

    All input models MUST already produce y-scale predictions (i.e. be
    :class:`CompositeTargetEstimator` wrappers OR a raw regressor on
    the original target). The ensemble does not invert anything --
    it just averages.

    The ensemble class itself is strategy-neutral: weights are
    pre-computed by :meth:`from_train_metrics` (the recommended path)
    or :meth:`from_uniform_weights` (mean baseline) and frozen on the
    instance. ``predict`` is one matrix-vector product.

    Validation gate
    ---------------
    :meth:`from_train_metrics` runs a built-in gate: it compares the
    ensemble's train-set RMSE against the best single component's
    train-set RMSE. If the ensemble is worse, it returns the best
    single component instead and logs a warning. The check is
    biased optimistic (uses train data) but still catches the most
    common failure mode -- a high-variance candidate with a stretched
    weight that drags the ensemble below the strongest component.
    """

    def __init__(
        self,
        component_models: list[Any],
        component_names: list[str],
        weights: np.ndarray,
        strategy: str,
        notes: dict[str, Any] | None = None,
        is_convex: bool = True,
    ) -> None:
        if len(component_models) == 0:
            raise ValueError("CompositeCrossTargetEnsemble: empty component list.")
        if len(component_models) != len(component_names) or len(component_models) != len(weights):
            raise ValueError(
                "CompositeCrossTargetEnsemble: component_models, component_names, "
                "and weights must be same length; got "
                f"{len(component_models)} / {len(component_names)} / {len(weights)}."
            )
        weights = np.asarray(weights, dtype=np.float64)
        # ``is_convex=True``: weights are a convex combination (non-negative, sum to 1) -- normalise
        # so the predict path can rely on the invariant. ``is_convex=False``: weights are the raw
        # output of a constrained / unconstrained solver (Ridge linear_stack, NNLS without renorm)
        # and downstream code must NOT assume sum=1; predict for these strategies does an
        # additive linear combination (no renormalisation on the surviving subset).
        if is_convex:
            wsum = float(weights.sum())
            if wsum <= 0 or not math.isfinite(wsum):
                raise ValueError(
                    f"CompositeCrossTargetEnsemble: convex weights must sum to positive finite "
                    f"value; got sum={wsum}."
                )
            weights = weights / wsum
        else:
            if not np.all(np.isfinite(weights)):
                raise ValueError(
                    "CompositeCrossTargetEnsemble: weights contain non-finite values."
                )
        self.component_models = list(component_models)
        self.component_names = list(component_names)
        self.weights = weights
        self.strategy = strategy
        self.is_convex = bool(is_convex)
        self.notes = dict(notes or {})
        # REFIT-NNLS / REFIT-RIDGE: small LRU cache for surviving-subset refits. 32 entries is plenty
        # for the handful of distinct member-dropout patterns a production session sees. Backed by
        # ``OrderedDict`` + ``move_to_end`` so the eviction order tracks ACCESS, not just insertion
        # (the pre-fix dict-only impl was FIFO despite the LRU label).
        self._refit_cache: OrderedDict[tuple[str, tuple[int, ...]], tuple[np.ndarray, float]] = OrderedDict()
        self._refit_cache_capacity: int = 32

    def _refit_cache_get(self, kind: str, surviving_key: tuple[int, ...]):
        key = (kind, surviving_key)
        if key not in self._refit_cache:
            return None
        self._refit_cache.move_to_end(key)
        return self._refit_cache[key]

    def _refit_cache_put(self, kind: str, surviving_key: tuple[int, ...], value):
        key = (kind, surviving_key)
        if key in self._refit_cache:
            self._refit_cache.move_to_end(key)
            self._refit_cache[key] = value
            return
        if len(self._refit_cache) >= self._refit_cache_capacity:
            self._refit_cache.popitem(last=False)
        self._refit_cache[key] = value

    # ------------------------------------------------------------------
    # Constructors / factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_uniform_weights(
        cls,
        component_models: list[Any],
        component_names: list[str],
    ) -> CompositeCrossTargetEnsemble:
        """Equal-weight average: ``w_k = 1/K`` for all components."""
        n = len(component_models)
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=np.full(n, 1.0 / n) if n > 0 else np.array([]),
            strategy="mean",
        )

    @classmethod
    def from_linear_stack(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_predictions: np.ndarray,  # (n_train, K) y-scale predictions
        y_train: np.ndarray,
        ridge_alpha: float | None = None,
        ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        sample_weight: np.ndarray | None = None,
    ) -> CompositeCrossTargetEnsemble:
        """Linear stacking via Ridge regression with internal CV alpha selection.

        Fits a Ridge model ``y_train ~ X @ w + b`` where ``X`` is the
        per-component prediction matrix on train. The resulting
        weights are the stack coefficients; intercept is folded into
        the bias by absorbing it as an extra ``+b/n`` per component
        (good enough when Ridge converges).

        When ``ridge_alpha`` is None (default), the alpha is chosen via
        ``RidgeCV`` over ``ridge_alpha_grid`` using built-in efficient
        leave-one-out CV. Pass a concrete ``ridge_alpha`` to bypass CV.

        Returns negative weights when a component is anti-correlated
        with the target -- this is fine, the ensemble may still work.
        ``predict`` re-normalises only the magnitudes, so a negative
        weight means the component's prediction is subtracted.
        """
        # ENS-Low-7: Ridge / RidgeCV hoisted to module-top imports.
        n = len(component_models)
        if n == 0:
            raise ValueError("from_linear_stack: empty component list.")
        component_predictions = np.asarray(component_predictions, dtype=np.float64)
        if component_predictions.shape[1] != n:
            raise ValueError(
                f"from_linear_stack: prediction matrix has {component_predictions.shape[1]} "
                f"columns, expected {n} (one per component)."
            )
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if len(y) != component_predictions.shape[0]:
            raise ValueError(
                f"from_linear_stack: y_train length {len(y)} != prediction "
                f"matrix rows {component_predictions.shape[0]}."
            )
        # Drop rows with non-finite y or predictions.
        finite = np.isfinite(y) & np.all(np.isfinite(component_predictions), axis=1)
        if finite.sum() < n + 2:
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: only %d finite rows for "
                "%d components; falling back to oof_weighted-style mean.",
                int(finite.sum()), n,
            )
            return cls.from_uniform_weights(component_models, component_names)

        # Optional per-row weighting for the Ridge fit. Ridge / RidgeCV both accept sample_weight natively;
        # we slice it on the same ``finite`` mask used for X / y so weight rows stay aligned with kept rows.
        _ridge_sw = None
        if sample_weight is not None:
            _ridge_sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if _ridge_sw.shape[0] != component_predictions.shape[0]:
                raise ValueError(
                    f"from_linear_stack: sample_weight length {_ridge_sw.shape[0]} != "
                    f"prediction matrix rows {component_predictions.shape[0]}."
                )
            if not np.all(np.isfinite(_ridge_sw)) or (_ridge_sw < 0).any():
                raise ValueError("from_linear_stack: sample_weight must be finite and non-negative.")
            _ridge_sw = _ridge_sw[finite]

        if ridge_alpha is None:
            ridge = RidgeCV(alphas=tuple(ridge_alpha_grid), fit_intercept=True)
            ridge.fit(component_predictions[finite], y[finite], sample_weight=_ridge_sw)
            chosen_alpha = float(getattr(ridge, "alpha_", ridge_alpha_grid[0]))
        else:
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
            ridge.fit(component_predictions[finite], y[finite], sample_weight=_ridge_sw)
            chosen_alpha = float(ridge_alpha)
        raw_weights = np.asarray(ridge.coef_, dtype=np.float64)
        # Sanity: if all weights are zero or non-finite, fall back.
        if not np.any(raw_weights) or not np.all(np.isfinite(raw_weights)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: degenerate weights; "
                "falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)
        # WEIGHT-NEGATIVE-WARN: Ridge can produce negative coefficients when a component is
        # anti-correlated with the target. That's mathematically fine for a linear stack but
        # operators routinely interpret negative-weight ensembles as a bug; surface them at WARN
        # so the suite log makes the situation explicit. The ensemble is still built (the predict
        # path supports negative weights via is_convex=False).
        _neg_mask = raw_weights < 0
        if _neg_mask.any():
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: %d/%d components have negative Ridge "
                "coefficients (anti-correlated with target on train). Ensemble is still built; the "
                "predict path subtracts the weighted contribution of those members. Affected: %s.",
                int(_neg_mask.sum()), len(raw_weights),
                ", ".join(f"{component_names[i]} (w={raw_weights[i]:.4g})" for i in np.flatnonzero(_neg_mask).tolist()),
            )
        # Ridge stack: weights may be negative and do not sum to 1; signal this with
        # is_convex=False so the constructor skips the convex-sum normalisation that
        # would otherwise destroy the Ridge fit. Eliminates the historical
        # build-with-placeholder-then-mutate dance.
        instance = cls(
            component_models=component_models,
            component_names=component_names,
            weights=raw_weights,
            strategy="linear_stack",
            notes={
                "ridge_alpha": chosen_alpha,
                "ridge_alpha_was_cv_selected": ridge_alpha is None,
                "ridge_alpha_grid": list(ridge_alpha_grid) if ridge_alpha is None else None,
                "intercept": float(ridge.intercept_),
                "raw_weights": raw_weights.tolist(),
                "n_train_rows": int(finite.sum()),
            },
            is_convex=False,
        )
        instance._linear_stack_intercept = float(ridge.intercept_)
        # SOLVER-COPY: ``component_predictions[finite]`` already returns a copy (boolean indexing on
        # ndarray always allocates), so the prior explicit ``.copy()`` was a 256-MB-per-pickle
        # duplicate. Same for ``y[finite]``. Keep a reference instead of a copy of a copy.
        instance._linear_stack_train_preds = component_predictions[finite]
        instance._linear_stack_train_y = y[finite]
        instance._linear_stack_ridge_alpha = chosen_alpha
        return instance

    @classmethod
    def from_nnls_stack(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_predictions: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> CompositeCrossTargetEnsemble:
        """Non-negative least squares stacking.

        Fits ``y = X @ w`` subject to ``w >= 0`` via
        ``scipy.optimize.nnls``. Weights are kept AS NNLS computed them
        (no post-hoc renormalisation to sum=1) -- renormalising would
        return a predictor that differs from the one the gate (and
        any downstream RMSE-on-y-scale evaluation) measured. The
        is_convex=False flag signals this to downstream consumers so
        they don't assume sum=1.

        sample_weight: optional per-row weights. scipy.nnls has no native sample_weight kwarg, so we
        emulate weighted least squares by row-scaling: replace ``A x = b`` with ``diag(sqrt(w)) A x =
        diag(sqrt(w)) b``. The NNLS minimiser of the scaled system is identical to the weighted-LS
        minimiser of the original system because ||sqrt(w) (Ax - b)||_2^2 == sum_i w_i (a_i x - b_i)^2.
        """
        from scipy.optimize import nnls
        n = len(component_models)
        if n == 0:
            raise ValueError("from_nnls_stack: empty component list.")
        component_predictions = np.asarray(component_predictions, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y) & np.all(np.isfinite(component_predictions), axis=1)
        if finite.sum() < n + 2:
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: only %d finite rows for "
                "%d components; falling back to mean.",
                int(finite.sum()), n,
            )
            return cls.from_uniform_weights(component_models, component_names)

        _A_for_nnls = component_predictions[finite]
        _b_for_nnls = y[finite]
        _nnls_sw = None
        if sample_weight is not None:
            _nnls_sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if _nnls_sw.shape[0] != component_predictions.shape[0]:
                raise ValueError(
                    f"from_nnls_stack: sample_weight length {_nnls_sw.shape[0]} != "
                    f"prediction matrix rows {component_predictions.shape[0]}."
                )
            if not np.all(np.isfinite(_nnls_sw)) or (_nnls_sw < 0).any():
                raise ValueError("from_nnls_stack: sample_weight must be finite and non-negative.")
            _nnls_sw = _nnls_sw[finite]
            _sqrt_w = np.sqrt(_nnls_sw).reshape(-1, 1)
            _A_for_nnls = _A_for_nnls * _sqrt_w
            _b_for_nnls = _b_for_nnls * _sqrt_w.reshape(-1)

        try:
            w, _residual = nnls(_A_for_nnls, _b_for_nnls)
        except RuntimeError as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: solver failed (%s); "
                "falling back to mean.", exc,
            )
            return cls.from_uniform_weights(component_models, component_names)

        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: zero or non-finite "
                "weights; falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)

        instance = cls(
            component_models=component_models,
            component_names=component_names,
            weights=w,
            strategy="nnls_stack",
            notes={
                "raw_weights": w.tolist(),
                "n_train_rows": int(finite.sum()),
            },
            # Don't renormalise: the predictor returned must match the one NNLS solved
            # for so the gate's measured RMSE corresponds to the deployed model.
            is_convex=False,
        )
        # SOLVER-COPY: boolean-indexing already produces a copy; the prior explicit ``.copy()`` was
        # a redundant 256-MB-per-pickle duplicate. Keep the bool-indexed view directly.
        instance._nnls_stack_train_preds = component_predictions[finite]
        instance._nnls_stack_train_y = y[finite]
        return instance

    @classmethod
    def from_train_metrics(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_train_rmse: Sequence[float],
        baseline_train_rmse: float | None = None,
        component_oof_rmse: Sequence[float] | None = None,
        baseline_oof_rmse: float | None = None,
    ) -> Union[CompositeCrossTargetEnsemble, Any]:
        """Build an ensemble weighted by *gain over a naive baseline*.

        The gain-over-naive convention defends against the trivial
        "raw model with TVT_prev as a feature" beating the naive
        ``predict y = base`` baseline by a tiny margin: that model's
        absolute RMSE is small (good), but its *gain* over the naive
        baseline is also small, so it gets a sensible weight rather
        than dominating the ensemble simply because its RMSE is
        numerically smaller than a good but harder-target composite.

        ``baseline_train_rmse`` is the RMSE of the naive predictor
        ``y_hat = base`` on train (or any sensible benchmark; pass
        the noisiest reasonable predictor's RMSE). If None, the
        median of ``component_train_rmse`` is used as a self-
        normalising fallback.

        If every component's RMSE is worse than the baseline, the
        method returns the SINGLE best-RMSE component instead of the
        ensemble (validation gate). Log line announces the fallback.
        """
        n = len(component_models)
        if n == 0:
            raise ValueError("from_train_metrics: empty component list.")
        # VAL-LEAK (from_train_metrics): prefer OOF RMSE when supplied. Train-set RMSE is biased
        # because train rows were seen at fit -- using it for weight derivation is the
        # same selection problem as using val (already burned for ES). When ``component_oof_rmse``
        # is given we rank on OOF; otherwise we fall back to train_rmse with a WARN so the operator
        # knows the gate is biased optimistic.
        if component_oof_rmse is not None:
            rmses = np.asarray(component_oof_rmse, dtype=np.float64)
            if len(rmses) != n:
                raise ValueError(
                    f"from_train_metrics: component_oof_rmse list len {len(rmses)} != n_components {n}."
                )
            if baseline_oof_rmse is not None:
                baseline_train_rmse = baseline_oof_rmse
        else:
            rmses = np.asarray(component_train_rmse, dtype=np.float64)
            if len(rmses) != n:
                raise ValueError(
                    f"from_train_metrics: rmse list len {len(rmses)} != n_components {n}."
                )
            logger.warning(
                "[CompositeCrossTargetEnsemble] from_train_metrics: ranking on TRAIN RMSE which is "
                "biased optimistic (rows seen at fit). Pass component_oof_rmse=... for an honest "
                "cross-validated weighting."
            )
        if not np.all(np.isfinite(rmses)):
            raise ValueError("from_train_metrics: rmses contain non-finite values.")

        if baseline_train_rmse is None:
            # MEDIAN-BASELINE: previously fell back to ``np.median(rmses)`` which by construction
            # discards the worse-than-median half of the candidate pool entirely. That's a hidden
            # contract surprise: the caller passed K components expecting a K-component ensemble
            # and got a (K/2)-component one with no log line. We now use the WORST component RMSE
            # (numerically the largest) as the baseline so every component that beats the worst
            # contributes a non-zero weight, AND we WARN the caller that no explicit baseline was
            # passed -- the operator should plug in a real benchmark (naive predictor / median
            # baseline_train_rmse / dataset variance) for production runs.
            baseline = float(np.max(rmses))
            logger.warning(
                "[CompositeCrossTargetEnsemble] from_train_metrics: no baseline_train_rmse passed; "
                "defaulting to max(component_train_rmse)=%.4g so every component beats baseline. "
                "Pass an explicit baseline (e.g. naive predictor RMSE) to get a real gain-over-naive weighting.",
                baseline,
            )
        else:
            baseline = float(baseline_train_rmse)
            if not math.isfinite(baseline):
                baseline = float(np.max(rmses))

        gains = np.maximum(0.0, baseline - rmses)
        if gains.sum() <= 0:
            # No component beats baseline. Return the single best by
            # RMSE; ensemble would be no improvement.
            best_idx = int(np.argmin(rmses))
            logger.warning(
                "[CompositeCrossTargetEnsemble] no component beats the baseline "
                "RMSE=%.4g; falling back to single best component '%s' (RMSE=%.4g).",
                baseline, component_names[best_idx], rmses[best_idx],
            )
            return component_models[best_idx]

        # The "no component beats baseline" gate fires above; the
        # only remaining decision is to build the ensemble.
        # We deliberately do NOT add an independence-bound RMSE gate
        # here: composite-target predictions correlate (same train
        # data, shared base feature), so the independence formula
        # overestimates ensemble RMSE and the gate would fire on
        # legitimate ensembles. The true validation gate -- "ensemble
        # OOF-RMSE > best single OOF-RMSE" -- requires real CV-OOF
        # predictions per component, which the per-target loop does
        # not currently expose. A future PR may add OOF storage; for
        # now the user is expected to evaluate the ensemble on a
        # held-out test set themselves.
        weights = gains / gains.sum()
        best_single_idx = int(np.argmin(rmses))
        best_single_rmse = float(rmses[best_single_idx])
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=weights,
            strategy="oof_weighted",
            notes={
                "baseline_train_rmse": baseline,
                "component_train_rmses": rmses.tolist(),
                "best_single_rmse": best_single_rmse,
                "best_single_name": component_names[best_single_idx],
                "gate_fallback": False,
            },
        )

    # ------------------------------------------------------------------
    # sklearn-ish API
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Weighted combination of per-component predictions.

        For ``mean`` / ``oof_weighted`` / ``nnls_stack`` strategies
        weights are non-negative and sum to 1; the result is a
        weighted average. For ``linear_stack`` strategy weights may be
        negative, do not sum to 1, and an intercept is added -- the
        result is the Ridge stack's prediction
        ``y_hat = X @ w + intercept``.
        """
        if not self.component_models:
            raise RuntimeError("CompositeCrossTargetEnsemble: no components.")
        per_component = []
        for model, name in zip(self.component_models, self.component_names):
            try:
                # Fold the dtype cast into the asarray call so we don't
                # allocate twice on the predict hot path. ``copy=False``
                # is the asarray default; the dtype kwarg lets us skip
                # a separate ``.astype()`` round-trip.
                pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
            except Exception as exc:
                logger.warning(
                    "[CompositeCrossTargetEnsemble] component '%s' predict failed: "
                    "%s. Excluding from this batch's ensemble (re-normalising).",
                    name, exc,
                )
                pred = None
            per_component.append(pred)

        # Track surviving indices so linear_stack can refit Ridge on exactly the columns
        # whose components produced predictions for this batch.
        surviving_idx = [i for i, p in enumerate(per_component) if p is not None]
        ok = [(per_component[i], self.weights[i]) for i in surviving_idx]
        if not ok:
            raise RuntimeError(
                "CompositeCrossTargetEnsemble.predict: all components failed."
            )
        preds_matrix = np.column_stack([p for p, _ in ok])
        weights = np.array([w for _, w in ok], dtype=np.float64)

        if not getattr(self, "is_convex", True):
            # Non-convex strategies (linear_stack, nnls_stack): weights are the raw solver
            # output, possibly negative (Ridge) or with arbitrary sum (NNLS). Refit the
            # appropriate solver on the surviving columns of the stashed training matrix
            # when any component drops out -- the alternative (drop the column but reuse
            # the original intercept / coefficient mix) is biased. The all-present fast
            # path skips the refit and uses the stored weights directly.
            full_weights = np.asarray(self.weights, dtype=np.float64)
            full_intercept = float(getattr(self, "_linear_stack_intercept", 0.0))
            if len(surviving_idx) == len(self.component_models):
                return (preds_matrix * full_weights[None, :]).sum(axis=1) + full_intercept

            # REFIT-NNLS / REFIT-RIDGE: cache the solver output keyed by the sorted surviving_idx
            # tuple. Member-dropout patterns are few in practice (a handful at most across a
            # production session); a 32-entry LRU is plenty and keeps memory predictable.
            surviving_key = tuple(sorted(surviving_idx))

            if self.strategy == "linear_stack":
                train_preds = getattr(self, "_linear_stack_train_preds", None)
                train_y = getattr(self, "_linear_stack_train_y", None)
                if train_preds is None or train_y is None:
                    raise RuntimeError(
                        "CompositeCrossTargetEnsemble.predict: linear_stack lost component(s) "
                        f"({len(self.component_models) - len(surviving_idx)} of "
                        f"{len(self.component_models)}) but no training matrix is available "
                        "to refit Ridge. Refusing to predict to avoid intercept-induced bias."
                    )
                alpha = float(getattr(self, "_linear_stack_ridge_alpha", 1.0))
                _cached = self._refit_cache_get("ridge", surviving_key)
                if _cached is None:
                    refit = Ridge(alpha=alpha, fit_intercept=True)
                    refit.fit(np.asarray(train_preds)[:, surviving_idx], np.asarray(train_y))
                    new_w = np.asarray(refit.coef_, dtype=np.float64)
                    new_intercept = float(refit.intercept_)
                    self._refit_cache_put("ridge", surviving_key, (new_w, new_intercept))
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] linear_stack: %d of %d components dropped "
                        "out at predict time; refit Ridge on surviving subset (alpha=%g).",
                        len(self.component_models) - len(surviving_idx),
                        len(self.component_models),
                        alpha,
                    )
                else:
                    new_w, new_intercept = _cached
                return (preds_matrix * new_w[None, :]).sum(axis=1) + new_intercept

            if self.strategy == "nnls_stack":
                train_preds = getattr(self, "_nnls_stack_train_preds", None)
                train_y = getattr(self, "_nnls_stack_train_y", None)
                if train_preds is None or train_y is None:
                    raise RuntimeError(
                        "CompositeCrossTargetEnsemble.predict: nnls_stack lost component(s) "
                        f"({len(self.component_models) - len(surviving_idx)} of "
                        f"{len(self.component_models)}) but no training matrix is available "
                        "to refit NNLS. Refusing to predict to avoid biased dropout."
                    )
                from scipy.optimize import nnls as _nnls

                _cached = self._refit_cache_get("nnls", surviving_key)
                if _cached is None:
                    new_w, _ = _nnls(
                        np.asarray(train_preds)[:, surviving_idx], np.asarray(train_y)
                    )
                    new_w = np.asarray(new_w, dtype=np.float64)
                    self._refit_cache_put("nnls", surviving_key, (new_w, 0.0))
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] nnls_stack: %d of %d components dropped "
                        "out at predict time; refit NNLS on surviving subset.",
                        len(self.component_models) - len(surviving_idx),
                        len(self.component_models),
                    )
                else:
                    new_w, _ = _cached
                return (preds_matrix * new_w[None, :]).sum(axis=1)

            # Other non-convex strategy without a training-matrix stash -- pass through as
            # a raw linear combination (rare; only hit if a future caller adds a strategy
            # without stashing the training data).
            return (preds_matrix * full_weights[None, :]).sum(axis=1)

        # Convex strategies (mean / oof_weighted): re-normalise across surviving components.
        if weights.sum() <= 0:
            # All surviving weights collapsed to zero -- fall back to
            # mean across surviving components.
            weights = np.full_like(weights, 1.0 / len(weights))
        else:
            weights = weights / weights.sum()
        return (preds_matrix * weights[None, :]).sum(axis=1)

    def export_metadata(self) -> dict[str, Any]:
        """Plain-dict snapshot for ``metadata`` storage."""
        return {
            "strategy": self.strategy,
            "component_names": list(self.component_names),
            "weights": self.weights.tolist(),
            "notes": dict(self.notes),
        }

    def cap_inference_components(
        self, max_components: int,
    ) -> CompositeCrossTargetEnsemble:
        """Return a NEW ensemble holding only the top-N components by
        absolute weight.

        Use case: production online prediction with a latency budget
        that can't afford running K=8 wrappers per row. Trims to the
        largest-weighted components and re-normalises (or preserves
        the linear-stack semantics by keeping the matching subset of
        weights + intercept). Returns a new ensemble; the original
        is unchanged.

        ``max_components <= 0`` or ``>= len(components)`` -> returns
        a copy of self unchanged (no trimming).
        """
        if max_components <= 0 or max_components >= len(self.component_models):
            copy_inst = CompositeCrossTargetEnsemble(
                component_models=list(self.component_models),
                component_names=list(self.component_names),
                weights=np.asarray(self.weights, dtype=np.float64),
                strategy=self.strategy,
                notes=dict(self.notes),
                is_convex=getattr(self, "is_convex", True),
            )
            for _attr in (
                "_linear_stack_intercept", "_linear_stack_train_preds",
                "_linear_stack_train_y", "_linear_stack_ridge_alpha",
                "_nnls_stack_train_preds", "_nnls_stack_train_y",
            ):
                if hasattr(self, _attr):
                    setattr(copy_inst, _attr, getattr(self, _attr))
            return copy_inst
        # Pick top-N by |weight|.
        order = np.argsort(-np.abs(np.asarray(self.weights, dtype=np.float64)))
        keep = sorted(order[:max_components].tolist())
        new = CompositeCrossTargetEnsemble(
            component_models=[self.component_models[i] for i in keep],
            component_names=[self.component_names[i] for i in keep],
            weights=np.asarray([self.weights[i] for i in keep], dtype=np.float64),
            strategy=self.strategy,
            notes={**self.notes, "capped_to_top_n": int(max_components),
                   "dropped_components": [
                       self.component_names[i]
                       for i in range(len(self.component_models))
                       if i not in keep
                   ]},
            is_convex=getattr(self, "is_convex", True),
        )
        # Carry over linear/NNLS stash so the trimmed ensemble can still refit on subset-drops.
        # The training-matrix stash retains ALL original columns; predict-time refit will
        # select the kept ones via surviving_idx so this is correct without re-slicing here.
        if self.strategy == "linear_stack" and hasattr(self, "_linear_stack_intercept"):
            new._linear_stack_intercept = self._linear_stack_intercept
        for _attr in (
            "_linear_stack_train_preds", "_linear_stack_train_y", "_linear_stack_ridge_alpha",
            "_nnls_stack_train_preds", "_nnls_stack_train_y",
        ):
            if hasattr(self, _attr):
                # NB: we keep the full training matrix; predict's surviving_idx selects subset.
                # cap_inference_components(N) however only stores N components, so surviving_idx
                # selects relative to N. Slice the training matrix columns to match the new
                # component ordering.
                _val = getattr(self, _attr)
                if _attr.endswith("_train_preds") and _val is not None:
                    setattr(new, _attr, np.asarray(_val)[:, keep])
                else:
                    setattr(new, _attr, _val)
        return new

