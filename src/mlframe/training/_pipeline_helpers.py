"""Pipeline application helpers extracted from ``trainer.py``.

Pre-pipeline transforms (SimpleImputer, StandardScaler, feature
selectors) with caching, passthrough-column handling, and
identity-equivalent detection.

Key functions:
- ``_apply_pre_pipeline_transforms`` — fit/transform train+val
- ``_prepare_test_split`` — transform test_df through fitted pipeline
- ``_passthrough_cols_fit_transform`` — hide passthrough cols, run fn, reattach
- Pipeline cache: structural-identity short-circuit for identical
  pre_pipelines across models (saves ~46s per duplicate fit)
"""

from __future__ import annotations

import contextlib
import logging
import os
from timeit import default_timer as timer

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram
from .phases import phase

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from .utils import log_ram_usage, filter_existing

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Pipeline structural-identity cache
# ═══════════════════════════════════════════════════════════════════

from collections import OrderedDict  # noqa: E402
import threading as _threading  # noqa: E402

_PRE_PIPELINE_CACHE_LOCK = _threading.Lock()
_PRE_PIPELINE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
# Default LRU bound; overridable per call via TrainingBehaviorConfig.pre_pipeline_cache_max so long-running services can tune memory vs reuse rate without monkey-patching.
_PRE_PIPELINE_CACHE_MAX: int = 4


def _content_fingerprint_for_cache(arr) -> tuple:
    """Content-based fingerprint of an array / DataFrame / target.

    id()-keying is unsafe: GC recycles ids and the suite's per-target loop
    persists ``filtered_train_df`` across targets with the same id() so
    target-2 would otherwise reuse target-1's fit-transform output. The
    fingerprint samples shape + dtype + 10 evenly-spaced cells; column
    names are folded in for DataFrame inputs so a rename invalidates the
    key. Falls back to ``("uncached", id)`` so a failure forces a miss
    rather than a wrong-cache hit.
    """
    if arr is None:
        return ("none",)
    try:
        col_names = None
        if hasattr(arr, "columns"):
            try:
                col_names = tuple(str(c) for c in arr.columns)
            except Exception:
                col_names = None
        if hasattr(arr, "to_numpy"):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ("uncached", id(arr))
        elif hasattr(arr, "values"):
            np_arr = arr.values
        elif isinstance(arr, np.ndarray):
            np_arr = arr
        else:
            np_arr = np.asarray(arr)
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return ("uncached", id(arr))
        shape = tuple(int(s) for s in np_arr.shape)
        dtype_str = str(np_arr.dtype)
        flat = np_arr.ravel()
        n = int(flat.size)
        if n == 0:
            return (shape, dtype_str, b"", col_names)
        idx = [int(i * (n - 1) / 9) for i in range(10)] if n >= 10 else list(range(n))
        try:
            sampled = bytes(np.ascontiguousarray(flat[idx]).tobytes())
        except Exception:
            return ("uncached", id(arr))
        return (shape, dtype_str, sampled, col_names)
    except Exception:
        return ("uncached", id(arr))


def _prepare_test_split(
    df,
    test_df,
    test_idx,
    test_target,
    target,
    real_drop_columns,
    model,
    pre_pipeline,
    skip_pre_pipeline_transform,
    skip_preprocessing=False,
    selector_passthrough_cols=None,
):
    """Prepare test DataFrame and target for evaluation."""
    # Lazy import: _data_helpers and _pipeline_helpers depend on each other; importing at top-level creates a circular load.
    from ._data_helpers import _subset_dataframe, _extract_target_subset
    if (df is not None) or (test_df is not None):
        if test_df is None:
            test_df = _subset_dataframe(df, test_idx, real_drop_columns)

        if test_target is None:
            test_target = _extract_target_subset(target, test_idx)

        if model is not None and pre_pipeline:
            # 2026-05-13: even when the train/val path took a cache hit
            # (``skip_pre_pipeline_transform=True``), test_df MUST be
            # transformed if the pre_pipeline is fitted AND changes the
            # data (not identity-equivalent).  The 2026-05-13 prod crash
            # hit this precisely: ``cached_dfs`` carried raw test_df
            # with NaN; ``skip_pre_pipeline_transform`` was True;
            # LinearRegression received NaN -> ValueError.  Tree models
            # (CB/XGB/LGB) handle NaN natively, so they were fine.
            #
            # The guard: if the pre_pipeline is identity-equivalent
            # (kept all columns, no recipes), transform is a no-op AND
            # we dodge NotFittedError from an unfitted instance (train
            # cache returned data without fitting).  Otherwise, ALWAYS
            # apply transform when the pre_pipeline IS fitted.
            _id_equiv = getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
            if not _id_equiv:
                _do_transform = (
                    not skip_pre_pipeline_transform
                    or _is_fitted(pre_pipeline)
                )
                if _do_transform:
                    if skip_preprocessing:
                        feature_selector = _extract_feature_selector(pre_pipeline)
                        if feature_selector is not None:
                            test_df = _passthrough_cols_fit_transform(
                                feature_selector.transform,
                                test_df,
                                passthrough_cols=selector_passthrough_cols,
                            )
                    else:
                        test_df = _passthrough_cols_fit_transform(
                            pre_pipeline.transform,
                            test_df,
                            passthrough_cols=selector_passthrough_cols,
                        )
        columns = list(test_df.columns) if hasattr(test_df, "columns") else []
    else:
        columns = []
        test_df = None

    return test_df, test_target, columns


def _extract_feature_selector(pre_pipeline):
    """Extract the feature selector ('pre' step) from a sklearn Pipeline.

    Feature selectors are added as the 'pre' step in pipelines built by
    ModelPipelineStrategy.build_pipeline() in strategies.py.

    Args:
        pre_pipeline: The preprocessing pipeline (sklearn Pipeline or transformer)

    Returns:
        The feature selector if found, otherwise None
    """
    if pre_pipeline is None:
        return None
    # If it's a Pipeline with named steps, look for the 'pre' step
    if hasattr(pre_pipeline, "named_steps") and "pre" in pre_pipeline.named_steps:
        return pre_pipeline.named_steps["pre"]
    # If it's not a Pipeline, it might be the feature selector itself (e.g., MRMR, RFECV)
    if not isinstance(pre_pipeline, Pipeline):
        return pre_pipeline
    return None


def _is_fitted(estimator):
    """Check if an sklearn estimator is already fitted.

    Uses sklearn's check_is_fitted() to determine if the estimator has been
    fitted. This is useful for determining whether to call fit_transform()
    or just transform() on a pipeline/feature selector that may have been
    loaded from cache.

    Args:
        estimator: An sklearn-compatible estimator (Pipeline, RFECV, etc.)

    Returns:
        bool: True if the estimator is fitted, False otherwise
    """
    if estimator is None:
        return False
    # For a sklearn Pipeline, ``check_is_fitted`` passes as long as ANY step
    # has fitted state -- even if later steps are still unfitted. That bit
    # us on 2026-04-22 (fuzz c0031): LinearModelStrategy's pre_pipeline had
    # a fitted MRMR step (reused from a prior CB iteration) but un-fitted
    # encoder/imputer/scaler. _is_fitted returned True -> code took the
    # ".transform only" branch -> imputer.transform raised ValueError 'The
    # feature names should match those that were passed during fit'.
    # Require every non-trivial step to be fitted.
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(estimator, Pipeline):
            for _name, step in estimator.steps:
                if step is None or step == "passthrough":
                    continue
                try:
                    check_is_fitted(step)
                except NotFittedError:
                    return False
            return True
    except Exception:
        pass
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False


def _multilabel_target_to_1d_for_supervised_encoders(target):
    """Collapse a 2-D multilabel target to a 1-D signal for supervised encoders.

    Many supervised feature transformers (category_encoders TargetEncoder,
    sklearn TargetEncoder, polars-ds supervised steps) only accept 1-D y.
    For multilabel data the natural reduction is "any positive label", which
    keeps a useful signal for target-mean encoders without crashing the fit.

    Handles three input shapes:
      - 2-D ndarray (N, K): canonical multilabel
      - 1-D object ndarray of per-row arrays: stacked first, then collapsed
        (the polars ``pl.List(pl.Int8)`` -> pandas object roundtrip lands
        here; surfaced 3-way fuzz c0008 (cb_linear / multilabel target)
        where the encoder rejected the object-cell column with
        ``Encoders require their input argument must be uniformly strings
        or numbers. Got ['ndarray']``)
      - Anything else: returned unchanged.
    """
    if target is None:
        return target
    arr = np.asarray(target) if not isinstance(target, np.ndarray) else target
    if arr.dtype == object and arr.ndim == 1 and arr.shape[0] > 0:
        _first = arr[0]
        if hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))):
            try:
                arr = np.stack([np.asarray(c) for c in arr], axis=0)
            except Exception:
                return target
    if arr.ndim != 2:
        return target
    return (arr.sum(axis=1) > 0).astype(np.int8)


def _passthrough_cols_fit_transform(fn, df, *args, passthrough_cols=None, fit=False, target=None, groups=None):
    """Run a selector fit/transform on df with passthrough_cols hidden, then re-attach them.

    Feature selectors (MRMR, RFECV) can't encode text or list-of-float embedding columns;
    catboost needs them back intact for fit. Hide -> run -> re-attach preserves both.

    Numpy-output fallback (2026-04-22): if the inner ``fn`` is a default sklearn
    Pipeline (no ``set_output(transform="pandas")``), ``out`` comes back as a numpy
    array. The original code detected this via ``hasattr(out, "columns")`` and
    silently returned numpy -- dropping ``passthrough_cols`` and, worse, collapsing
    pd.Categorical dtypes in the selected columns to numpy object strings, which
    crashes LGB's Dataset construction on the ``'HOURLY'`` path. We now rebuild a
    pd.DataFrame from the reduced-input column names so passthrough_cols re-attach
    and downstream models take the native-pandas fastpath.

    ``groups`` is threaded through fit/fit_transform so GroupKFold-aware feature
    selectors (RFECV with cv=GroupKFold(), MRMR with grouped CV) receive the
    sample-grouping signal. fix audit row FS-P1-1.
    """

    # 2026-05-11 Wave 20: convert sklearn/polars "empty output" errors into
    # an empty-frame return so the downstream ``train_df.shape[1] == 0``
    # guard at trainer.py:4515 fires cleanly. Triggered by MRMR / RFECV
    # confirming 0 predictors: ``fit_transform`` returns ``(N, 0)`` which
    # crashes in either:
    #   - SimpleImputer / scaler: ``ValueError: need at least one array
    #     or dtype is required`` (numpy.find_common_type on empty list).
    #   - sklearn's pandas-container wrap of a polars ``(N, 0)`` frame:
    #     ``ValueError: need at least one array to concatenate``
    #     (polars.to_numpy -> vstack on empty list).
    # Surfaced by Wave 15 MRMR fuzz axes (c0008 with interactions_max_order=3
    # + fe_max_steps=2 confirming 0 predictors).
    def _call_fit(_fn, _arg, _target_arg):
        """Invoke fit/fit_transform with ``groups`` when supported. sklearn
        feature selectors that consume groups (RFECV(cv=GroupKFold()), MRMR)
        accept ``groups`` as a kwarg on ``fit`` / ``fit_transform``. Best-effort:
        if the underlying transformer does not accept the kwarg we retry
        without it instead of failing.

        sklearn Pipeline.fit raises ``ValueError`` (not ``TypeError``) when an
        unrecognised kwarg like ``groups`` is passed - the routing layer rejects
        non-step-namespaced parameters with a "Pipeline.fit does not accept"
        message. Treat that as the same "does not consume groups" signal as
        a plain ``TypeError`` from a bare transformer.
        """
        if groups is None:
            return _fn(_arg, _target_arg)
        try:
            return _fn(_arg, _target_arg, groups=groups)
        except TypeError:
            return _fn(_arg, _target_arg)
        except ValueError as _exc:
            _msg = str(_exc)
            if (
                "does not accept the groups parameter" in _msg
                or "got an unexpected keyword argument" in _msg
                or "unexpected keyword argument 'groups'" in _msg
            ):
                return _fn(_arg, _target_arg)
            raise

    def _run_and_empty_check(_fn, _arg, _target_arg):
        try:
            return _call_fit(_fn, _arg, _target_arg) if fit else _fn(_arg)
        except ValueError as _exc:
            _m = str(_exc)
            if "need at least one array to concatenate" in _m or "at least one array or dtype is required" in _m:
                # Empty output (0 features); return an empty DataFrame of
                # matching shape so the suite's downstream 0-feature guard
                # catches it.
                if isinstance(_arg, pl.DataFrame):
                    return _arg.select([])
                if hasattr(_arg, "iloc"):
                    return _arg.iloc[:, :0]
                return np.empty((len(_arg), 0))
            raise

    if not passthrough_cols or df is None or not hasattr(df, "columns"):
        return _run_and_empty_check(fn, df, target)
    present = [c for c in passthrough_cols if c in df.columns]
    if not present:
        return _run_and_empty_check(fn, df, target)
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        held = df.select(present)
        reduced = df.drop(present)
    else:
        held = df[present]
        reduced = df.drop(columns=present)
    out = _call_fit(fn, reduced, target) if fit else fn(reduced)
    if hasattr(out, "columns"):
        if isinstance(out, pl.DataFrame):
            out = out.with_columns([held[c] for c in present])
        else:
            held_pd = held.to_pandas() if is_polars else held
            # 2026-05-12 Wave 26: single ``pd.concat`` instead of per-column
            # ``out.loc[:, c] = ...`` assignment loop. Each column write
            # triggers a full-DataFrame copy; concat does one allocation
            # and copies all columns in one pass.
            # ``out`` may be a view/slice of an inner sklearn pipeline
            # result (e.g. SelectorMixin returning ``X.iloc[:, support_]``);
            # ``pd.concat`` semantics consume views cleanly, no
            # SettingWithCopyWarning. The original ``out.copy()`` is NOT
            # needed before concat -- concat returns a new frame regardless.
            out = pd.concat([out, held_pd], axis=1)
    elif isinstance(out, np.ndarray) and out.ndim == 2:
        # Reconstruct a DataFrame using the reduced-input column names when the
        # transformer preserved the column count. If the shape differs (e.g. a
        # feature selector dropped columns), we can't safely name them -- fall
        # back to positional names and warn via debug log.
        reduced_cols = list(reduced.columns)
        if out.shape[1] == len(reduced_cols):
            col_names = reduced_cols
        else:
            col_names = [f"f{i}" for i in range(out.shape[1])]
        out = pd.DataFrame(out, columns=col_names, index=getattr(reduced, "index", None))
        held_pd = held.to_pandas() if is_polars else held
        # Wave 26: single concat instead of per-column assignment loop.
        out = pd.concat([out, held_pd], axis=1)
    return out


# Cache + lock are defined once near the top; the duplicate definitions used to live here.


def _pipeline_signature_for_cache(pipeline) -> str:
    """Stable signature for the pipeline structure + per-step shallow params.

    Two structurally identical pipelines (same step classes, same per-step
    kwargs) get the same string and hit the cache; any divergence (e.g. a
    custom scaler with different ``with_mean``) misses. Failures inside
    ``get_params`` (custom transformers without sklearn API) fall back to a
    class-only signature -- a conservative "no cache" since the same class
    might have different state.
    """
    if pipeline is None:
        return "None"
    parts = []
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        # Single transformer (not a Pipeline) -- include its repr.
        return f"single:{type(pipeline).__name__}:{repr(pipeline)}"
    for name, step in steps:
        kls = type(step).__name__
        try:
            params = step.get_params(deep=False)
            kw = ",".join(f"{k}={params[k]!r}" for k in sorted(params))
        except Exception:
            kw = "?"
        parts.append(f"{name}:{kls}({kw})")
    return "|".join(parts)


def _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target=None, target_name=None):
    """Compose a CONTENT-based cache key.

    id()-keying was unsafe on two axes: GC-recycled ids can collide and
    the per-target loop re-uses ``filtered_train_df`` (same id) across
    different targets - the second target would otherwise see the first
    target's fit-transform output. Including the target fingerprint AND
    the target name guarantees per-target isolation.
    """
    sig = _pipeline_signature_for_cache(pipeline)
    return (
        _content_fingerprint_for_cache(train_df),
        _content_fingerprint_for_cache(val_df),
        _content_fingerprint_for_cache(train_target),
        str(target_name) if target_name is not None else "",
        sig,
    )


def _pre_pipeline_cache_get(train_df, val_df, pipeline, train_target=None, target_name=None, cache_max: int | None = None):
    """LRU-touch lookup; returns ``(train_out, val_out)`` or ``None``."""
    if train_df is None or pipeline is None:
        return None
    key = _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target, target_name)
    with _PRE_PIPELINE_CACHE_LOCK:
        if key in _PRE_PIPELINE_CACHE:
            _PRE_PIPELINE_CACHE.move_to_end(key)
            return _PRE_PIPELINE_CACHE[key]
    return None


def _pre_pipeline_cache_set(train_df, val_df, pipeline, train_out, val_out, train_target=None, target_name=None, cache_max: int | None = None):
    """Insert under LRU, evicting the oldest entry if over capacity.

    ``cache_max`` overrides the module default; pass through from the
    caller's ``TrainingBehaviorConfig.pre_pipeline_cache_max`` so
    long-running services can tune memory vs hit-rate.
    """
    if train_df is None or pipeline is None:
        return
    key = _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target, target_name)
    _cap = int(cache_max) if cache_max is not None else _PRE_PIPELINE_CACHE_MAX
    with _PRE_PIPELINE_CACHE_LOCK:
        _PRE_PIPELINE_CACHE[key] = (train_out, val_out)
        _PRE_PIPELINE_CACHE.move_to_end(key)
        while len(_PRE_PIPELINE_CACHE) > _cap:
            _PRE_PIPELINE_CACHE.popitem(last=False)


def _pre_pipeline_cache_clear() -> None:
    """Manual eviction hook -- mainly for tests + edge cases where the
    per-target loop wants to drop stale state explicitly."""
    with _PRE_PIPELINE_CACHE_LOCK:
        _PRE_PIPELINE_CACHE.clear()


def _apply_pre_pipeline_transforms(
    model,
    pre_pipeline,
    train_df,
    val_df,
    train_target,
    skip_pre_pipeline_transform,
    skip_preprocessing,
    use_cache,
    model_file_name,
    verbose,
    selector_passthrough_cols=None,
    target_name: str | None = None,
    cache_max: int | None = None,
    groups=None,
):
    """Apply pre-pipeline transformations to train and validation DataFrames.

    Args:
        model: The model being trained
        pre_pipeline: Preprocessing pipeline (may include feature selector + preprocessing steps)
        train_df: Training DataFrame
        val_df: Validation DataFrame (or None)
        train_target: Training target values
        skip_pre_pipeline_transform: If True, skip entire pipeline (for cached DFs)
        skip_preprocessing: If True, skip only preprocessing steps but run feature selectors
        use_cache: Whether to use cached pipeline
        model_file_name: Model file path for cache checking
        verbose: Verbosity level
        groups: optional grouping array for grouped-CV-aware selectors (RFECV with
            GroupKFold(), grouped MRMR). Threaded into ``fit_transform`` /
            ``fit_transform_resample`` when the underlying transformer accepts it
            (fix audit row FS-P1-1).
    """
    if model is not None and pre_pipeline:
        t0_pre = timer()
        # 2026-05-13 (duplicate-pipeline skip): capture input column names
        # BEFORE transform so we can detect identity-equivalent pipelines
        # (selected all columns, created none). Set on the pre_pipeline
        # instance so the suite loop in core/main.py can skip redundant
        # branches.
        _input_cols = (
            list(train_df.columns) if hasattr(train_df, "columns") else None
        )
        # 2026-05-12: structurally-identical-pipeline cache. When the
        # PER-TARGET loop runs Linear then MLP back-to-back, both build
        # ``SimpleImputer + StandardScaler``; without this short-circuit we
        # re-fit the identical arithmetic on the same train_df (~46s linear
        # + ~18s mlp on the 4M-row TVT log). Only fires on the fresh
        # fit-transform path (the other branches are already cheap).
        # Capture the INPUT content fingerprints BEFORE any rebind so the cache
        # populate path keys on the original (caller-owned) df contents. The
        # populate path computes the key from the SAME inputs so a cache miss
        # followed by a populate is guaranteed to land in the slot the next
        # lookup will read from.
        _cache_key_entry = _pre_pipeline_cache_key(
            train_df, val_df, pre_pipeline,
            train_target=train_target, target_name=target_name,
        )
        _cache_hit = _pre_pipeline_cache_get(
            train_df, val_df, pre_pipeline,
            train_target=train_target, target_name=target_name,
        )
        if _cache_hit is not None and not skip_pre_pipeline_transform and not skip_preprocessing and not _is_fitted(pre_pipeline):
            train_df_cached, val_df_cached = _cache_hit
            if verbose:
                logger.info("Reusing pre_pipeline fit-transform from cache " "(same train_df + structurally identical pipeline).")
            shape_str = f"{train_df_cached.shape[0]:_}x{train_df_cached.shape[1]}" if hasattr(train_df_cached, "shape") else ""
            if verbose:
                logger.info(
                    "  pre_pipeline done (cached) -- train: %s, %.1fs",
                    shape_str,
                    timer() - t0_pre,
                )
            return train_df_cached, val_df_cached
        with phase("pre_pipeline_fit_transform"):
            if skip_pre_pipeline_transform:
                if verbose:
                    logger.info("Skipping pre_pipeline fit/transform (using cached DFs)")
            elif skip_preprocessing:
                # Only run feature selector, skip preprocessing steps (scaler/imputer/encoder)
                # This is used when polars-ds pipeline already applied scaling/imputation
                feature_selector = _extract_feature_selector(pre_pipeline)
                if feature_selector is not None:
                    if _is_fitted(feature_selector):
                        if verbose:
                            logger.info("Using pre-fitted feature selector (transform only): %s", feature_selector)
                        train_df = _passthrough_cols_fit_transform(
                            feature_selector.transform,
                            train_df,
                            passthrough_cols=selector_passthrough_cols,
                        )
                    else:
                        if verbose:
                            logger.info("Fitting feature selector: %s", feature_selector)
                        train_df = _passthrough_cols_fit_transform(
                            feature_selector.fit_transform,
                            train_df,
                            passthrough_cols=selector_passthrough_cols,
                            fit=True,
                            target=train_target,
                            groups=groups,
                        )
                    if verbose:
                        log_ram_usage()
                    if val_df is not None:
                        if verbose:
                            logger.info(f"Transforming val_df via feature selector...")
                        val_df = _passthrough_cols_fit_transform(
                            feature_selector.transform,
                            val_df,
                            passthrough_cols=selector_passthrough_cols,
                        )
                        if verbose:
                            log_ram_usage()
                elif verbose:
                    logger.info("No feature selector found in pipeline, skipping all transforms")
            elif _is_fitted(pre_pipeline):
                if verbose:
                    try:
                        logger.info("Using pre-fitted pipeline (transform only): %s", pre_pipeline)
                    except (ValueError, TypeError):
                        pass
                train_df = _passthrough_cols_fit_transform(
                    pre_pipeline.transform,
                    train_df,
                    passthrough_cols=selector_passthrough_cols,
                )
                if verbose:
                    log_ram_usage()
                if val_df is not None:
                    if verbose:
                        logger.info(f"Transforming val_df via pre_pipeline...")
                    # Historical 0-row val skip removed 2026-04-27 (batch 3).
                    # The original empty-val window came from outlier
                    # detection rejecting almost every val row; that's now
                    # guarded at the source by the val-side ``min_keep``
                    # floor + class-balance pre-check in
                    # ``core._apply_outlier_detection_global``. If a 0-row
                    # val still arrives here it's an upstream bug -- letting
                    # SimpleImputer raise ``Found array with 0 sample(s)``
                    # surfaces it immediately instead of training a model
                    # we can't evaluate.
                    val_df = _passthrough_cols_fit_transform(
                        pre_pipeline.transform,
                        val_df,
                        passthrough_cols=selector_passthrough_cols,
                    )
                    if verbose:
                        log_ram_usage()
            else:
                if verbose:
                    logger.info("Fitting & transforming train_df via pre_pipeline %s...", pre_pipeline)
                # 2026-04-24 Session 6: supervised encoders (category_encoders
                # TargetEncoder, polars-ds supervised steps) reject 2-D y. Collapse
                # multilabel targets to "any positive label" for the encoder fit
                # only -- actual model still trains on the full (N, K) target.
                _enc_target = _multilabel_target_to_1d_for_supervised_encoders(train_target)
                train_df = _passthrough_cols_fit_transform(
                    pre_pipeline.fit_transform,
                    train_df,
                    passthrough_cols=selector_passthrough_cols,
                    fit=True,
                    target=_enc_target,
                    groups=groups,
                )
                if verbose:
                    log_ram_usage()
                if val_df is not None:
                    if verbose:
                        logger.info("Transforming val_df via pre_pipeline %s...", pre_pipeline)
                    # Historical 0-row val skip removed 2026-04-27 (batch 3) --
                    # see fit-transform branch comment for rationale.
                    val_df = _passthrough_cols_fit_transform(
                        pre_pipeline.transform,
                        val_df,
                        passthrough_cols=selector_passthrough_cols,
                    )
                    if verbose:
                        log_ram_usage()
            _maybe_clean_ram()
            if verbose:
                shape_str = f"{train_df.shape[0]:_}x{train_df.shape[1]}" if hasattr(train_df, "shape") else ""
                logger.info("  pre_pipeline done -- train: %s, %.1fs", shape_str, timer() - t0_pre)
            # 2026-05-12: populate the LRU cache so the next sklearn-non-
            # native model in this per-target iteration (typically MLP
            # after Linear) gets a cache hit on the same train_df + the
            # same structural pipeline. Guarded: only stash when we
            # actually went through the fit-transform branch (the others
            # didn't have anything new to cache anyway).
            if not skip_pre_pipeline_transform and not skip_preprocessing:
                # Reuse the entry-time key so a populate after a miss is guaranteed to land in the slot the next lookup will read from.
                try:
                    _cap = int(cache_max) if cache_max is not None else _PRE_PIPELINE_CACHE_MAX
                    with _PRE_PIPELINE_CACHE_LOCK:
                        _PRE_PIPELINE_CACHE[_cache_key_entry] = (train_df, val_df)
                        _PRE_PIPELINE_CACHE.move_to_end(_cache_key_entry)
                        while len(_PRE_PIPELINE_CACHE) > _cap:
                            _PRE_PIPELINE_CACHE.popitem(last=False)
                except Exception as _cache_err:
                    logger.debug(
                        "pre_pipeline cache populate skipped: %s",
                        _cache_err,
                    )

        # 2026-05-13 (duplicate-pipeline skip): detect whether this
        # pre_pipeline left the column set unchanged (identity-equivalent).
        # Marker is set on the pre_pipeline instance so the suite loop in
        # core/main.py can skip this branch when an equivalent one
        # (ordinary or another identity-selector) already ran.
        if _input_cols is not None and hasattr(train_df, "columns"):
            _output_cols = list(train_df.columns)
            try:
                pre_pipeline._mlframe_identity_equivalent = (
                    _input_cols == _output_cols
                )
            except Exception:
                pass  # non-writable pre_pipeline (e.g. tuple), safe to ignore

    return train_df, val_df
