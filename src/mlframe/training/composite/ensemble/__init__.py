"""Cross-target ensemble + OOF utilities: CompositeCrossTargetEnsemble (stack/weighted/mean strategies with validation gate), compute_oof_holdout_predictions, derive_seeds (delegates to mlframe.core.helpers.derive_seed), detect_gpu_in_use, env_signature. Split out of composite.py to keep ensemble concerns separate from discovery; composite.py re-exports every symbol below at its bottom for full back-compat."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union, cast,
)

import numpy as np
import pandas as pd
from sklearn.base import clone
# Hoist sklearn.linear_model out of the predict() hot path so an inference round-trip does not pay the cold-import cost on the first call.
from sklearn.linear_model import Ridge, RidgeCV

try:
    import polars as pl
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """Explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from ..estimator import CompositeTargetEstimator
from ..post_shim import PrePipelinePredictShim, subset_to_fit_columns
from ..transforms import get_transform
from ._oof_split import (
    _align_fit_sw,
    _carve_inner_eval_split,
    _slice_rows,
)


def _unwrap_shim(model: Any) -> tuple[Any, Any]:
    """Return ``(inner, pre_pipeline)`` for a possibly shim-wrapped component.

    For ``PrePipelinePredictShim`` returns its fitted ``pre_pipeline`` and the inner model (which may itself be a ``CompositeTargetEstimator``); for any other estimator pre_pipeline is ``None`` and the model is returned unchanged. The OOF refit path uses this to detect shim wrappers and apply ``pre_pipeline.transform`` to stack/holdout slices before refitting -- without it the OOF path would call ``sklearn.clone(shim)``, which raises ``Cannot clone object ... not a scikit-learn estimator`` for every shim-wrapped component.
    """
    if isinstance(model, PrePipelinePredictShim):
        return model.model, model.pre_pipeline
    return model, None


logger = logging.getLogger(__name__)


def _pp_is_fitted(pp: Any) -> bool:
    """True iff every non-trivial step of ``pp`` is fitted (lazy import of the
    pipeline-helper check to avoid an import cycle at module load).
    """
    try:
        from ...pipeline._pipeline_helpers import _is_fitted
    except Exception:  # pragma: no cover - defensive; treat as unfitted
        return False
    return bool(_is_fitted(pp))


def _transform_pair_via(
    pp: Any,
    X_train: Any,
    X_holdout: Any,
    *,
    y_train: Any = None,
) -> tuple[Any, Any]:
    """Project the OOF train + holdout slices into the SAME feature space the deployed model predicts in.

    The deployed component (:class:`PrePipelinePredictShim`) routes every predict through ``pre_pipeline.transform`` and REFUSES to feed raw X to the inner (post_shim.py). The OOF estimate must mirror that, so this helper never silently falls back to raw X (which evaluated a different space than deployed and biased the NNLS weights).

    - ``pp is None`` -> pass both slices through unchanged.
    - ``pp`` already fitted (the suite-normal case: the entry's pre_pipeline was fit-transformed on full train during the main pass) -> apply ``pp.transform`` to both slices, EXACTLY as deployment does. Bit-identical to the previous fitted path; the only change is that the transform happens ONCE per slice with no surrounding per-slice try/except-and-fallback.
    - ``pp`` NOT fitted -> clone it and fit the clone on the TRAIN slice only (train-only, leak-free: the holdout slice never touches the scaler/imputer/selector fit), then transform both slices through that fitted clone. This is the honest-OOF analogue of "pp fitted on train, applied at predict" and keeps the OOF in the deployed (transformed) space instead of the old raw-X fallback.

    A fit/transform failure is allowed to propagate to the per-component ``except`` in the OOF loops, which drops the component from the ensemble -- the same outcome the deployed shim produces when its transform fails (it raises rather than predict on unscaled features), so a component that cannot be projected into the deployed space is excluded rather than scored in the wrong space.

    The caller's ``pp`` is never mutated: the unfitted branch fits a ``clone`` so the shared deployed pipeline object keeps its state across slices/folds/components.
    """
    if pp is None:
        return X_train, X_holdout
    if _pp_is_fitted(pp):
        return pp.transform(subset_to_fit_columns(X_train, pp)), pp.transform(subset_to_fit_columns(X_holdout, pp))
    # Unfitted pre_pipeline: fit a clone on the train slice (leak-free) and
    # reuse it for both slices so the OOF lives in the deployed space.
    pp_fit = clone(pp)
    try:
        pp_fit.fit(X_train, y_train)
    except TypeError:
        # Some unsupervised pipelines expose fit(X) only.
        pp_fit.fit(X_train)
    return pp_fit.transform(X_train), pp_fit.transform(X_holdout)


def _transform_via(pp: Any, X: Any) -> Any:
    """Apply a fitted pre_pipeline to ``X`` or return ``X`` unchanged when ``pp is None``.

    Retained for external callers / single-frame projection (e.g. the external-holdout train side, which is paired with the holdout via :func:`_transform_pair_via`). On an UNFITTED ``pp`` this raises rather than silently returning raw X, mirroring the deployed shim's refusal to predict on unscaled features; OOF loops route through :func:`_transform_pair_via` so the unfitted case is handled leak-free without a per-slice raise.
    """
    if pp is None:
        return X
    return pp.transform(subset_to_fit_columns(X, pp))


def derive_seeds(random_state: int, components: Sequence[str]) -> dict[str, int]:
    """Derive deterministic per-component seeds from a master seed.

    Delegates to the canonical :func:`mlframe.core.helpers.derive_seed` per component; returns a dict
    mapping each component name to an int in ``[0, 2**31 - 1)``.

    Why: threading one ``random_state`` through several randomness sources (MI sampling, tiny-model CV split, OOF holdout split, bootstrap CI) correlates them (an "easy" MI sample coincides with an "easy" CV split); independent sub-seeds break that while staying reproducible (same master seed -> same sub-seeds).

    Currently provided for external callers only: discovery does NOT yet thread these sub-seeds into its randomness sources, so wiring it in would change every draw and is not bit-identical.
    """
    from mlframe.core.helpers import derive_seed

    return {c: derive_seed(random_state, c) for c in components}


def detect_gpu_in_use(mlframe_models: Sequence[str]) -> list[str]:
    """Return list of model families that may be using GPU.

    Best-effort detection: imports each library only if it appears in ``mlframe_models`` and probes GPU availability via the library's standard health-check API; returns the GPU-detected subset (empty list when no GPU library is in use).

    Used by the suite to emit a one-shot warning when composite mode is combined with GPU training: GPU non-determinism is amplified by K composite-model fits and can surface as ensemble weight drift across runs even when ``random_state`` is fixed.
    """
    detected: list[str] = []
    families = {str(m).lower() for m in mlframe_models}
    if any(f in families for f in ("xgb", "xgboost")):
        try:
            import xgboost as xgb
            try:
                # build_info().USE_CUDA is only a BUILD flag (stock PyPI wheels are CUDA-enabled); AND it with a real device probe so a CPU-only host does not get a false GPU warning.
                bi = xgb.build_info()
                if isinstance(bi, dict) and bi.get("USE_CUDA", False):
                    try:
                        from pyutilz.system.gpu_dispatch import is_cuda_available
                        if is_cuda_available():
                            detected.append("xgboost")
                    except ImportError:
                        detected.append("xgboost")
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in __init__.py:149: %s", e)
                pass
        except ImportError:
            pass
    if any(f in families for f in ("cb", "catboost")):
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                detected.append("catboost")
        except Exception:  # nosec B110 - optional dependency import guard
            pass
    return detected


def env_signature() -> dict[str, str | None]:
    """Snapshot of library versions relevant to composite-target discovery + serialisation, stored on metadata so a pickle saved today can be reload-validated tomorrow against version drift. Returns ``None`` for any library not installed."""
    import platform
    from importlib.metadata import version, PackageNotFoundError
    # Read versions via dist metadata, NOT __import__: catboost/lgb/xgb cold-import is ~0.5-3s each and this runs on the predict-path drift check (keeps cold-import off the predict path).
    distmap = {"sklearn": "scikit-learn"}
    sig: dict[str, str | None] = {}
    for libname in ("numpy", "pandas", "polars", "sklearn", "lightgbm", "xgboost", "catboost", "scipy", "dill"):
        try:
            sig[libname] = version(distmap.get(libname, libname))
        except PackageNotFoundError:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            sig[libname] = None
    sig["python"] = platform.python_version()
    return sig


def _is_monotone_nondecreasing(arr: np.ndarray) -> bool:
    """True iff arr is finite and weakly monotone non-decreasing.

    Used to auto-detect timestamp/time-index columns so the OOF helper can produce a time-respecting split (past-only train, future holdout) instead of a random shuffle that leaks the future into the past.
    """
    try:
        a = np.asarray(arr).ravel()
    except Exception as exc:
        logger.debug("_is_monotone_nondecreasing: np.asarray coercion failed: %s", exc)
        return False
    if a.size < 2:
        return False
    # Cast to float so timestamps / ints alike work; non-numeric -> False.
    try:
        af = a.astype(np.float64, copy=False)
    except Exception as exc:
        logger.debug("_is_monotone_nondecreasing: float cast failed, non-numeric column: %s", exc)
        return False
    if not np.all(np.isfinite(af)):
        return False
    return bool(np.all(np.diff(af) >= 0))


def _maybe_pass_sample_weight(
    fit_callable, X, y,
    sw: np.ndarray | None,
    eval_set: tuple | None = None,
):
    """Call ``fit_callable.fit(X, y[, sample_weight, eval_set])`` honouring whichever kwargs the inner estimator's fit signature exposes.

    Avoids hard-coding which inner estimators support sample_weight (CatBoost/LGB/sklearn all do; some custom shims may not). ``eval_set`` is plumbed to fix the OOF-refit silent-drop pathology (observed in prod: LGBM clones with an ``early_stopping_rounds`` callback but no eval data raised ``"For early stopping, at least one dataset and eval metric is required for evaluation"`` and were dropped from the cross-target ensemble -- ensemble RMSE worse than dummy). Falls back to the plain call on TypeError so missing-kwarg shims keep working.
    """
    import inspect as _inspect
    if sw is not None and len(sw) != len(y):
        raise ValueError(f"_maybe_pass_sample_weight: sample_weight length {len(sw)} != y length {len(y)}")
    try:
        _sig = _inspect.signature(fit_callable.fit)
        _params = _sig.parameters
        _accepts_var_kw = any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in _params.values())
        _kwargs: dict = {}
        if sw is not None and ("sample_weight" in _params or _accepts_var_kw):
            _kwargs["sample_weight"] = sw
        if eval_set is not None and ("eval_set" in _params or _accepts_var_kw):
            # LightGBM expects a list of (X, y) tuples; XGBoost/CatBoost accept either (X, y) tuple or list-of-tuples. Normalising to list-of-tuples covers all three.
            _es = eval_set if isinstance(eval_set, list) else [eval_set]
            _kwargs["eval_set"] = _es
        if _kwargs:
            return fit_callable.fit(X, y, **_kwargs)
    except (TypeError, ValueError) as exc:
        logger.debug(
            "_maybe_pass_sample_weight: combined fit with kwargs %s failed (%s); retrying without them.",
            sorted(_kwargs.keys()), exc,
        )
    # Retry with sample_weight alone if combined call failed
    if sw is not None:
        try:
            return fit_callable.fit(X, y, sample_weight=sw)
        except (TypeError, ValueError) as exc:
            logger.debug(
                "_maybe_pass_sample_weight: sample_weight-only fit failed (%s); falling back to unweighted fit.",
                exc,
            )
    return fit_callable.fit(X, y)


# Module-level memo cache for compute_oof_holdout_predictions, keyed by (cache_key, kfold, random_state). cache_key is opaque -- callers pass a hashable tuple summarising the (component, X, y, sw) identity.
# Intentionally UNWIRED on the suite path: the cross-target ensemble builder computes each target's OOF exactly once per suite call on a fresh train frame, so there is no reuse to capture, and a content key on a TB-scale frame is forbidden by the RAM rule. The cache exists for EXTERNAL callers that legitimately repeat an identical OOF call with a cheap stable key (huge hit-speedup; see _benchmarks/bench_oof_cache_reuse.py). Do not wire a frame-hash key into the suite to "use" it -- that buys zero hits at the cost of a forbidden hash.
# DO NOT include ``id(train_X)`` in the cache_key: Python recycles object IDs across a long-lived suite, so two frames with disjoint content can share the same id and a stale entry can mask the swap. Prefer a content fingerprint (the suite computes one via ``pipeline_cache.fingerprint_df`` / ``_hash_frame``) or a stable per-session token. The historical comment that recommended ``id(train_X)`` was wrong; callers passing such a key get cross-contamination across re-runs.
# 16-entry LRU; once full the least-recently-USED entry is evicted on insertion. Stays in-process; cleared at interpreter shutdown. ``OrderedDict.move_to_end`` on hits keeps eviction order driven by access (true LRU), not insertion order (which would degrade to FIFO).
_OOF_HOLDOUT_CACHE: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray, list[str]]]" = OrderedDict()
_OOF_HOLDOUT_CACHE_CAP = 16


def _oof_cache_get(key: tuple):
    """LRU-touch and return the cached OOF result for ``key``, or ``None`` on a miss."""
    if key not in _OOF_HOLDOUT_CACHE:
        return None
    _OOF_HOLDOUT_CACHE.move_to_end(key)
    return _OOF_HOLDOUT_CACHE[key]


def _oof_cache_put(key: tuple, value: tuple) -> None:
    """Store ``value`` under ``key`` in the LRU OOF cache, evicting the oldest entry once over capacity."""
    if key in _OOF_HOLDOUT_CACHE:
        _OOF_HOLDOUT_CACHE.move_to_end(key)
        _OOF_HOLDOUT_CACHE[key] = value
        return
    if len(_OOF_HOLDOUT_CACHE) >= _OOF_HOLDOUT_CACHE_CAP:
        _OOF_HOLDOUT_CACHE.popitem(last=False)
    _OOF_HOLDOUT_CACHE[key] = value


def _compute_oof_with_external_holdout(
    *,
    # Slice-stable ES (mlframe.training.SliceStableESConfig) is NOT propagated into the inner OOF refit loop: this function builds its own per-fold ``eval_set`` via ``_carve_eval_set_from_train_with_groups`` and a single (X_holdout, y_holdout) pair, incompatible with the multi-eval-set / per-shard registration path slice-ES needs. Callers wanting robust ES inside OOF refit should use full-K-fold CV with an outer selector (see ``_cv_aggregation.aggregate_fold_scores``).
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    train_X: Any,
    y_train_full: np.ndarray,
    base_train_full_per_spec: dict[str, np.ndarray],
    external_holdout_X: Any,
    external_holdout_y: np.ndarray,
    sample_weight: np.ndarray | None,
    full_key: tuple | None,
    group_ids: np.ndarray | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit each component clone on full train, predict on caller-supplied external holdout (typically the suite's val split).

    Mirrors the per-component branch in :func:`compute_oof_holdout_predictions` but skips the internal train/holdout slicing.

    Holdout-side base columns are NOT taken as an argument: the ``CompositeTargetEstimator`` wrapper re-extracts its base column from ``external_holdout_X`` itself during ``predict``, so a parallel per-spec holdout-base dict would be dead weight (the train-side ``base_train_full_per_spec`` is still needed because it drives the transform.forward that produces the T values the inner is re-fit on).
    """
    y_train_full = y_train_full.astype(np.float64)
    holdout_cols: list[np.ndarray] = []
    surviving_names: list[str] = []
    for model, name, spec in zip(
        component_models, component_names, component_specs,
    ):
        try:
            inner, pp = _unwrap_shim(model)
            X_stack_t, X_holdout_t = _transform_pair_via(
                pp, train_X, external_holdout_X, y_train=y_train_full,
            )
            if isinstance(inner, CompositeTargetEstimator):
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(
                    str(spec.get("name") or spec.get("base_column")),
                )
                if base_full is None:
                    raise ValueError(f"missing base column '{spec['base_column']}' " "for external-holdout OOF (train side)")
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_train_full, base_full)
                if valid.sum() < 10:
                    raise ValueError("too few valid rows after domain filter")
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
                _sw_train_valid = None if sample_weight is None else sample_weight[valid]
                _group_for_valid = None
                if group_ids is not None:
                    try:
                        _g_arr = np.asarray(group_ids)
                        if _g_arr.shape[0] == valid.shape[0]:
                            _group_for_valid = _g_arr[valid]
                    except (TypeError, IndexError):
                        _group_for_valid = None
                _X_fit_c, _t_fit_c, _X_ev_c, _t_ev_c, _fm_c = (
                    _carve_inner_eval_split(
                        X_train_valid, t_train, random_state=int(random_state),
                        group_ids=_group_for_valid, return_fit_mask=True,
                    )
                )
                _eval_set_c = (_X_ev_c, _t_ev_c) if _X_ev_c is not None else None
                _sw_fit_c = _align_fit_sw(_sw_train_valid, _fm_c, len(_t_fit_c))
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_c, _t_fit_c, _sw_fit_c,
                    eval_set=_eval_set_c,
                )
                _extra = tuple(spec.get("extra_base_columns") or ())
                _base_columns = (spec["base_column"], *_extra) if _extra else None
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
                _X_fit_r, _y_fit_r, _X_ev_r, _y_ev_r, _fm_r = (
                    _carve_inner_eval_split(
                        X_stack_t, y_train_full, random_state=int(random_state),
                        group_ids=group_ids, return_fit_mask=True,
                    )
                )
                _eval_set_r = (_X_ev_r, _y_ev_r) if _X_ev_r is not None else None
                _sw_fit_r = _align_fit_sw(sample_weight, _fm_r, len(_y_fit_r))
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_r, _y_fit_r, _sw_fit_r,
                    eval_set=_eval_set_r,
                )
                preds = inner_clone.predict(X_holdout_t)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if preds.shape[0] != external_holdout_y.shape[0]:
                raise ValueError(f"component '{name}' predicted " f"{preds.shape[0]} rows but external holdout has " f"{external_holdout_y.shape[0]}")
            if not np.all(np.isfinite(preds)):
                raise ValueError("non-finite holdout predictions")
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            logger.warning(
                "[CompositeCrossTargetEnsemble] external-holdout OOF " "refit failed for component '%s': %s. Excluded from " "ensemble weights.",
                name,
                exc,
            )
            continue
    _surviving_n = len(surviving_names)
    _total_n = len(component_names)
    if _surviving_n < _total_n:
        _dropped = [n for n in component_names if n not in set(surviving_names)]
        logger.info(
            "compute_oof_holdout_predictions (external-holdout): built "
            "OOF matrix with %d of %d components (dropped %d: %s).",
            _surviving_n, _total_n, _total_n - _surviving_n, _dropped,
        )
    if not holdout_cols:
        _empty: tuple = (np.zeros((0, 0)), np.zeros(0), [])
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

    Approach: take a ``holdout_frac`` slice of train, re-fit a clone of each component's inner on the remaining (1-holdout_frac) rows, and predict on the held-out slice. For wrapped composite-target components re-apply the spec's transform on the same stack_train slice to get T values, train the inner clone on (X_stack_train, T_stack_train), then wrap via ``CompositeTargetEstimator.from_fitted_inner`` and predict in y-scale on stack_holdout. For raw-target components the inner clone is fit directly on (X_stack_train, y_stack_train).

    Split strategy:

    - **External holdout** (preferred when ``external_holdout_X`` is supplied): fit each component clone on the FULL train, predict on the caller-provided external frame (the suite's val split). Eliminates the train-tail-vs-test distribution mismatch that biases NNLS weights on group-aware splits of strong-AR targets (observed in prod: train-tail lag_predict RMSE 15.18 vs test 11.58 - NNLS underweights the dominant baseline because it looks bad on the train-tail). Composite components do NOT need a parallel holdout base array: the ``CompositeTargetEstimator`` wrapper re-extracts its base column directly from ``external_holdout_X`` at predict time, so the holdout-side base is read from the frame itself (a separate per-spec base dict would be dead weight).
    - ``time_ordering`` provided and monotone-non-decreasing OR ``time_ordering`` is ``None`` but rows are otherwise detected to be time-ordered: take the trailing ``holdout_frac`` slice as the holdout (past-only train, future holdout) -- the analogue of a single ``sklearn.model_selection.TimeSeriesSplit`` fold.
    - Otherwise: random shuffle by ``random_state`` (legacy behaviour).

    Parameters
    ----------
    kfold
        When > 1, perform K-fold OOF prediction instead of a single holdout slice. Each fold contributes its hold-out predictions; the concatenated (n_train, K) matrix is returned in natural row order. ``kfold=1`` preserves the legacy single-split behaviour. Random-shuffle only (time-aware K-fold remains the single-split trailing slice).
    external_holdout_base_per_spec
        Accepted for back-compat but UNUSED. Earlier docstrings called it REQUIRED for external-holdout composite components; that was wrong. The ``CompositeTargetEstimator`` wrapper re-extracts its base column from ``external_holdout_X`` itself at predict time, so no parallel holdout-base dict is consulted. Callers may keep passing it (it is silently ignored); new callers should omit it.

    Returns
    -------
    - ``holdout_preds_matrix``: y-scale predictions; shape ``(n_holdout, K)`` for kfold=1, ``(n_train, K)`` for kfold>1 random.
    - ``y_holdout``: y-scale targets aligned row-for-row.
    - ``surviving_names``: subset of ``component_names`` whose re-fit succeeded (failures are dropped from the matrix so callers can re-align weight vectors).
    """
    n_train = len(y_train_full)
    if n_train < 50 or holdout_frac <= 0 or holdout_frac >= 1:
        # Shape consistency across the three empty-return paths: ``surviving_names=[]`` means zero components survived; both axes are zero. Consumers probing ``.shape[1]`` against ``len(surviving_names)`` were correct; those probing against ``len(component_models)`` had been silently reading a non-empty K dimension with zero rows. Standardise to ``(0, 0)`` everywhere.
        return np.zeros((0, 0)), np.zeros(0), []

    # When the caller supplies a cache_key we look up the (key, kfold, rs) tuple. A hit is bit-identical with a previous call -- same components, X, y, fold strategy; caller semantics unchanged.
    _full_key = None
    if cache_key is not None:
        # Include every argument that changes the returned matrix so two calls differing only in holdout_frac / time-mode / external-holdout identity / group_ids cannot serve each other a stale entry.
        if group_ids is not None:
            _g_fp_arr = np.asarray(group_ids)
            _group_fp = (
                _g_fp_arr.shape[0],
                hash((tuple(_g_fp_arr[:64].tolist()), tuple(_g_fp_arr[-64:].tolist()))),
            )
        else:
            _group_fp = (0, 0)
        _full_key = (
            cache_key,
            int(kfold),
            int(random_state),
            round(float(holdout_frac), 6),
            int(time_ordering is not None),
            int(external_holdout_X is not None),
            (len(external_holdout_y) if external_holdout_y is not None else -1),
            _group_fp,
        )
        _hit = _oof_cache_get(_full_key)
        if _hit is not None:
            logger.debug("compute_oof_holdout_predictions: cache HIT for key=%r", _full_key)
            return cast(Tuple[np.ndarray, np.ndarray, List[str]], _hit)

    # Forward-walking K-fold OOF. When kfold>1 AND time_ordering is
    # monotone (rows already in time order), use TimeSeriesSplit -- K expanding-
    # window folds whose holdouts are FUTURE blocks, covering K slices instead of
    # one trailing slice while staying past-only. A NON-monotone time signal
    # cannot be forward-walked without sorting the OOF frame, so it still
    # downgrades to the single trailing-slice path (with a loud warn).
    _time_monotone = time_ordering is not None and _is_monotone_nondecreasing(time_ordering)
    if int(kfold) > 1 and time_ordering is not None and not _time_monotone:
        logger.warning(
            "compute_oof_holdout_predictions: kfold=%d AND a NON-monotone "
            "time_ordering; cannot forward-walk without sorting the OOF frame. "
            "Downgrading to a single trailing-slice holdout. Sort the frame by "
            "time (so time_ordering is monotone) to get a K-fold forward walk.",
            int(kfold),
        )
    if int(kfold) > 1 and (time_ordering is None or _time_monotone):
        if external_holdout_X is not None:
            logger.warning(
                "compute_oof_holdout_predictions: both kfold>1 and external_holdout_X supplied; kfold "
                "takes precedence and the external holdout frame is IGNORED. Pass kfold=1 to use the "
                "external holdout."
            )
        from sklearn.model_selection import KFold
        # Outer OOF split must be group-aware when group_ids is supplied: plain shuffled K-fold lets same-group rows span refit-train and holdout, inflating the OOF surface the NNLS weights + dummy-floor gate consume (the inner eval-carve is group-aware but the OUTER split was not). GroupKFold keeps whole groups in one fold.
        _kf_groups = None
        if group_ids is not None:
            _g_arr = np.asarray(group_ids)
            if _g_arr.shape[0] == n_train and np.unique(_g_arr).size >= int(kfold):
                _kf_groups = _g_arr
        if _time_monotone:
            from sklearn.model_selection import TimeSeriesSplit
            kf = TimeSeriesSplit(n_splits=int(kfold))
            _kf_split = kf.split(np.arange(n_train))
        elif _kf_groups is not None:
            from sklearn.model_selection import GroupKFold
            kf = GroupKFold(n_splits=int(kfold))
            _kf_split = kf.split(np.arange(n_train), groups=_kf_groups)
        else:
            kf = KFold(n_splits=int(kfold), shuffle=True, random_state=int(random_state))
            _kf_split = kf.split(np.arange(n_train))
        oof_preds_by_name: dict[str, np.ndarray] = {}
        survived_set: set[str] | None = None
        for fold_train_idx, fold_holdout_idx in _kf_split:
            # Sub-frame views by index; fit/predict inlined per fold.
            if _is_polars_df(train_X):
                fold_train_mask = np.zeros(n_train, dtype=bool)
                fold_train_mask[fold_train_idx] = True
                # Build the holdout from an EXPLICIT fold_holdout_idx mask, not ~train_mask: under TimeSeriesSplit ~train includes the FUTURE rows beyond this fold's holdout, so ~train_mask yields a frame longer than fold_holdout_idx -> length mismatch silently drops the component every fold (the pandas/ndarray branches already index by fold_holdout_idx).
                fold_holdout_mask = np.zeros(n_train, dtype=bool)
                fold_holdout_mask[fold_holdout_idx] = True
                X_stack = train_X.filter(pl.Series(fold_train_mask))
                X_holdout = train_X.filter(pl.Series(fold_holdout_mask))
            elif isinstance(train_X, pd.DataFrame):
                X_stack = train_X.iloc[fold_train_idx].reset_index(drop=True)
                X_holdout = train_X.iloc[fold_holdout_idx].reset_index(drop=True)
            else:
                X_stack = train_X[fold_train_idx]
                X_holdout = train_X[fold_holdout_idx]
            y_stack = y_train_full[fold_train_idx].astype(np.float64)
            # Per-fold OOF scoring happens once at the end against the assembled oof_preds_by_name
            # vs y_train_full as a whole, not per-fold slices, so a per-fold holdout-y copy here is unneeded.
            fold_cols: dict[str, np.ndarray] = {}
            for model, name, spec in zip(component_models, component_names, component_specs):
                try:
                    inner, pp = _unwrap_shim(model)
                    X_stack_t, X_holdout_t = _transform_pair_via(
                        pp, X_stack, X_holdout, y_train=y_stack,
                    )
                    if isinstance(inner, CompositeTargetEstimator):
                        if spec is None:
                            raise ValueError("composite component with no spec")
                        base_full = base_train_full_per_spec.get(str(spec.get("name") or spec.get("base_column")))
                        if base_full is None:
                            raise ValueError(f"missing base column '{spec['base_column']}'")
                        base_stack = base_full[fold_train_idx]
                        transform = get_transform(spec["transform_name"])
                        valid = transform.domain_check(y_stack, base_stack)
                        if valid.sum() < 10:
                            raise ValueError("too few valid rows after domain filter")
                        # Re-fit the transform params on THIS fold's train
                        # rows so alpha/beta/MAD do not see the held-out fold
                        # (spec["fitted_params"] were fit on the FULL train, incl.
                        # this fold's holdout -> mild OOF optimism). Fall back to
                        # the global params when the transform cannot 1-D-refit
                        # (e.g. multi-base needs the K-column matrix).
                        try:
                            _fold_params = transform.fit(y_stack[valid], base_stack[valid])
                        except Exception:
                            _fold_params = spec["fitted_params"]
                        t_stack = transform.forward(
                            y_stack[valid], base_stack[valid], _fold_params,
                        )
                        inner_clone = clone(inner.estimator_)
                        if isinstance(X_stack_t, pd.DataFrame):
                            X_stack_valid = X_stack_t.iloc[valid].reset_index(drop=True)
                        elif _is_polars_df(X_stack_t):
                            X_stack_valid = X_stack_t.filter(pl.Series(valid))
                        else:
                            X_stack_valid = X_stack_t[valid]
                        _sw_stack_valid = None if sample_weight is None else sample_weight[fold_train_idx][valid]
                        # Carve an inner eval_set so early-stopping boosters do not raise (they did, and the per-component except below silently dropped every ES composite component each fold -- default oof_holdout_source IS kfold). Mirrors the single-split composite branch + the raw kfold branch.
                        _group_fold_valid = None
                        if group_ids is not None:
                            try:
                                _g_arr = np.asarray(group_ids)
                                if _g_arr.shape[0] == n_train:
                                    _gf = _g_arr[fold_train_idx]
                                    if _gf.shape[0] == valid.shape[0]:
                                        _group_fold_valid = _gf[valid]
                                else:
                                    logger.warning(
                                        "[ensemble] group_ids length %d != n_train %d; group-aware carve skipped for this split.",
                                        _g_arr.shape[0], n_train,
                                    )
                            except (TypeError, IndexError, ValueError):
                                _group_fold_valid = None
                        _Xf_c, _tf_c, _Xe_c, _te_c, _fm_kc = _carve_inner_eval_split(
                            X_stack_valid, t_stack, random_state=int(random_state),
                            group_ids=_group_fold_valid, return_fit_mask=True,
                        )
                        _eval_set_kc = (_Xe_c, _te_c) if _Xe_c is not None else None
                        _sw_fit_kc = _align_fit_sw(_sw_stack_valid, _fm_kc, len(_tf_c))
                        _maybe_pass_sample_weight(
                            inner_clone, _Xf_c, _tf_c, _sw_fit_kc,
                            eval_set=_eval_set_kc,
                        )
                        # Multi-base parity with _phase_composite_post: pass the full base_columns tuple so predict reconstructs the K-column base matrix matching the K alphas.
                        _extra = tuple(spec.get("extra_base_columns") or ())
                        _base_columns = (spec["base_column"], *_extra) if _extra else None
                        wrapped = CompositeTargetEstimator.from_fitted_inner(
                            fitted_inner=inner_clone,
                            transform_name=spec["transform_name"],
                            base_column=spec["base_column"],
                            base_columns=_base_columns,
                            transform_fitted_params=_fold_params,
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
                                if _g_arr.shape[0] == n_train:
                                    _group_for_fold = _g_arr[fold_train_idx]
                                else:
                                    logger.warning(
                                        "[ensemble] group_ids length %d != n_train %d; group-aware carve skipped for this split.",
                                        _g_arr.shape[0], n_train,
                                    )
                            except (TypeError, IndexError, ValueError):
                                _group_for_fold = None
                        _X_fit, _y_fit, _X_ev, _y_ev, _fm_kr = _carve_inner_eval_split(
                            X_stack_t, y_stack, random_state=int(random_state),
                            group_ids=_group_for_fold, return_fit_mask=True,
                        )
                        _eval_set = (_X_ev, _y_ev) if _X_ev is not None else None
                        _sw_fit = _align_fit_sw(_sw_stack, _fm_kr, len(_y_fit))
                        _maybe_pass_sample_weight(
                            inner_clone, _X_fit, _y_fit, _sw_fit,
                            eval_set=_eval_set,
                        )
                        preds = inner_clone.predict(X_holdout_t)
                    preds = np.asarray(preds).reshape(-1).astype(np.float64)
                    if not np.all(np.isfinite(preds)):
                        raise ValueError("non-finite holdout predictions")
                    fold_cols[name] = preds
                except Exception as exc:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
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
            # Shape consistency -- match the (0, 0) tiny-data short-circuit. No components survived.
            _empty: tuple = (np.zeros((0, 0)), np.zeros(0), [])
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

    # External honest holdout (caller-supplied val frame). Skip the train-tail split entirely: fit each component clone on the FULL train, predict on the external frame, return the parallel y column the caller supplied. Defends against AR(1) train-tail distribution mismatch.
    if external_holdout_X is not None and external_holdout_y is not None and len(external_holdout_y) > 0:
        # external_holdout_base_per_spec is intentionally NOT forwarded: the
        # CompositeTargetEstimator wrapper re-extracts its base column from
        # external_holdout_X at predict time, so a parallel holdout-base dict
        # is unused. The public param is retained for back-compat (callers may
        # still pass it) but has no effect.
        return _compute_oof_with_external_holdout(
            component_models=component_models,
            component_names=component_names,
            component_specs=component_specs,
            train_X=train_X,
            y_train_full=y_train_full,
            base_train_full_per_spec=base_train_full_per_spec,
            external_holdout_X=external_holdout_X,
            external_holdout_y=np.asarray(external_holdout_y, dtype=np.float64),
            sample_weight=sample_weight,
            full_key=_full_key,
            group_ids=group_ids,
            random_state=random_state,
        )

    # Decide whether to do a time-aware split. Only the EXPLICIT ``time_ordering`` signal (the suite threads ctx.timestamps here) flips to a trailing-slice holdout. The old behaviour also probed every base column and auto-switched if ANY was monotone -- a false positive on sorted-but-non-temporal bases (sorted ids, binned features) that silently turned a random holdout into a trailing slice and changed the OOF leakage profile. Random shuffle is the safe default when no explicit time signal is given.
    use_time_split = False
    _time_order = None
    if time_ordering is not None:
        if _is_monotone_nondecreasing(time_ordering):
            use_time_split = True
            logger.info("composite OOF: time_ordering signal is monotone non-decreasing; using trailing-slice holdout instead of random shuffle.")
        else:
            # Explicit but non-monotone time signal: recover the forward-walk by SORTING rows by time, instead of silently random-shuffling (which discards the very signal the caller passed and leaks future rows into the refit-train slice). argsort is stable so ties keep their original order.
            _time_order = np.argsort(np.asarray(time_ordering), kind="stable")
            use_time_split = True
            logger.warning(
                "composite OOF: time_ordering was supplied but is NOT monotone "
                "non-decreasing; sorting rows by time to build the trailing-slice "
                "holdout (previously this silently fell back to a random shuffle, "
                "discarding the time signal)."
            )

    n_holdout = max(round(n_train * holdout_frac), 1)
    # Group-aware outer holdout: carve WHOLE groups so no group spans the refit-train and holdout slices (a random row permutation lets same-group rows leak across the split, inflating the OOF surface). Only when group_ids covers all rows with enough distinct groups and we are not on the explicit time-split path.
    _group_holdout = None
    if not use_time_split and group_ids is not None:
        _g_arr = np.asarray(group_ids)
        if _g_arr.shape[0] == n_train and np.unique(_g_arr).size >= 4:
            _group_holdout = _g_arr
    if use_time_split:
        cutoff = n_train - n_holdout
        if _time_order is not None:
            # Non-monotone path: trailing slice in TIME order.
            train_idx = _time_order[:cutoff].astype(np.int64)
            holdout_idx = _time_order[cutoff:].astype(np.int64)
        else:
            train_idx = np.arange(cutoff, dtype=np.int64)
            holdout_idx = np.arange(cutoff, n_train, dtype=np.int64)
    elif _group_holdout is not None:
        uniq, first_idx = np.unique(_group_holdout, return_index=True)
        rng = np.random.default_rng(random_state)
        shuffled_groups = rng.permutation(uniq[np.argsort(first_idx)])
        counts = {gid: int((_group_holdout == gid).sum()) for gid in uniq}
        cumulative, hold_groups = 0, []
        for gid in shuffled_groups:
            hold_groups.append(gid)
            cumulative += counts[gid]
            if cumulative >= n_holdout:
                break
        hold_mask = np.isin(_group_holdout, hold_groups)
        if hold_mask.all() or not hold_mask.any():  # degenerate carve (all/none) -> fall back to row permutation
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n_train)
            holdout_idx = np.sort(perm[:n_holdout])
            train_idx = np.sort(perm[n_holdout:])
        else:
            holdout_idx = np.nonzero(hold_mask)[0]
            train_idx = np.nonzero(~hold_mask)[0]
    else:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n_train)
        holdout_idx = np.sort(perm[:n_holdout])
        train_idx = np.sort(perm[n_holdout:])

    # Subset X. Branch on type so we don't pull pandas APIs on polars frames.
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

    # Group-aware inner eval-carve parity with the raw branch + kfold path: subset the caller-supplied group_ids to the stack rows so neither the composite nor the raw inner-eval carve splits a group across fit/eval (within-group leakage under-stops the booster and degrades OOF RMSE on group-aware splits).
    _group_stack = None
    if group_ids is not None:
        try:
            _g_arr = np.asarray(group_ids)
            if _g_arr.shape[0] == n_train:
                _group_stack = _g_arr[train_idx]
            else:
                logger.warning(
                    "[ensemble] group_ids length %d != n_train %d; group-aware carve skipped for this split.",
                    _g_arr.shape[0], n_train,
                )
        except (TypeError, IndexError, ValueError):
            _group_stack = None

    holdout_cols: list[np.ndarray] = []
    surviving_names = []
    for model, name, spec in zip(component_models, component_names, component_specs):
        try:
            inner, pp = _unwrap_shim(model)
            X_stack_t, X_holdout_t = _transform_pair_via(
                pp, X_stack, X_holdout, y_train=y_stack,
            )
            if isinstance(inner, CompositeTargetEstimator):
                # Composite-target wrapper. Re-fit the inner on stack_train T values, then re-wrap and predict.
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(str(spec.get("name") or spec.get("base_column")))
                if base_full is None:
                    raise ValueError(f"missing base column '{spec['base_column']}' for OOF")
                base_stack = base_full[train_idx]
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_stack, base_stack)
                # Drop invalid rows from stack_train; the inner trains only on rows where T is finite.
                if valid.sum() < 10:
                    raise ValueError("too few valid rows after domain filter")
                # Per-fold transform refit (see the kfold branch).
                try:
                    _fold_params = transform.fit(y_stack[valid], base_stack[valid])
                except Exception:
                    _fold_params = spec["fitted_params"]
                t_stack = transform.forward(
                    y_stack[valid], base_stack[valid], _fold_params,
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
                _X_fit_c, _t_fit_c, _X_ev_c, _t_ev_c, _fm_sc = _carve_inner_eval_split(
                    X_stack_valid, t_stack, random_state=int(random_state),
                    group_ids=_group_stack_valid, return_fit_mask=True,
                )
                _eval_set_c = (_X_ev_c, _t_ev_c) if _X_ev_c is not None else None
                _sw_fit_c = _align_fit_sw(_sw_stack_valid, _fm_sc, len(_t_fit_c))
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_c, _t_fit_c, _sw_fit_c,
                    eval_set=_eval_set_c,
                )
                # Multi-base parity: same fix as the kfold OOF branch above. Without base_columns, predict reconstructs only the primary base column and trips the K-alphas shape check.
                _extra = tuple(spec.get("extra_base_columns") or ())
                _base_columns = (spec["base_column"], *_extra) if _extra else None
                wrapped = CompositeTargetEstimator.from_fitted_inner(
                    fitted_inner=inner_clone,
                    transform_name=spec["transform_name"],
                    base_column=spec["base_column"],
                    base_columns=_base_columns,
                    transform_fitted_params=_fold_params,
                    y_train=y_stack[valid],
                )
                preds = wrapped.predict(X_holdout_t)
            else:
                # Raw-target component. Re-fit the inner on (X_stack, y_stack) and predict on X_holdout.
                inner_clone = clone(inner)
                _sw_stack = None if sample_weight is None else sample_weight[train_idx]
                _X_fit_r, _y_fit_r, _X_ev_r, _y_ev_r, _fm_sr = _carve_inner_eval_split(
                    X_stack_t, y_stack, random_state=int(random_state),
                    group_ids=_group_stack, return_fit_mask=True,
                )
                _eval_set_r = (_X_ev_r, _y_ev_r) if _X_ev_r is not None else None
                _sw_fit_r = _align_fit_sw(_sw_stack, _fm_sr, len(_y_fit_r))
                _maybe_pass_sample_weight(
                    inner_clone, _X_fit_r, _y_fit_r, _sw_fit_r,
                    eval_set=_eval_set_r,
                )
                preds = inner_clone.predict(X_holdout_t)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if not np.all(np.isfinite(preds)):
                raise ValueError("non-finite holdout predictions")  # NaN preds on holdout -- exclude from ensemble
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            logger.warning(
                "[CompositeCrossTargetEnsemble] OOF refit failed for component "
                "'%s': %s. Excluded from ensemble weights.", name, exc,
            )
            continue

    # Summary log so operators can see "ensemble built with N of K components" at INFO without grepping per-component WARN lines. ``component_names`` is the full caller-supplied list; ``surviving_names`` is the subset whose refit succeeded.
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
        # Shape consistency -- match the tiny-data + kfold short-circuits.
        _empty = (np.zeros((0, 0)), np.zeros(0), [])
        if _full_key is not None:
            _oof_cache_put(_full_key, _empty)
        return _empty
    _final = (np.column_stack(holdout_cols), y_holdout, surviving_names)
    if _full_key is not None:
        _oof_cache_put(_full_key, _final)
    return _final


from ._cross_target import CompositeCrossTargetEnsemble
from ._calibration import OutputCalibrator, fit_output_calibrator
from ._stackers import (
    META_STACKER_KINDS,
    build_meta_stack_ensemble,
    fit_gbm_meta_stacker,
    fit_lasso_meta_stacker,
    fit_ridge_meta_stacker,
)
