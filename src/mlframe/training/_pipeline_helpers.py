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

import logging
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

from .utils import log_ram_usage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Pipeline structural-identity cache
# ═══════════════════════════════════════════════════════════════════

from collections import OrderedDict  # noqa: E402
import threading as _threading  # noqa: E402

_PRE_PIPELINE_CACHE_LOCK = _threading.Lock()
_PRE_PIPELINE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
# Default LRU bound; overridable per call via TrainingBehaviorConfig.pre_pipeline_cache_max so long-running services can tune memory vs reuse rate without monkey-patching.
# Audit D P2-1 (2026-05-18): pre-fix default was 4 which silently evicted on the typical suite
# (cb + lgb + xgb + mlp + linear == 5 models). Bumped to 8 so a standard suite fits without
# thrashing; callers needing tighter bounds still set TrainingBehaviorConfig.pre_pipeline_cache_max.
_PRE_PIPELINE_CACHE_MAX: int = 8


# Audit D P0-3 (2026-05-18): a content-fingerprint failure must force a MISS, not a stable
# cache hit. The previous ``("uncached", id(arr))`` return was a stable tuple key, so two
# consecutive targets passing the SAME filtered_train_df id produced an IDENTICAL cache key
# for both targets -- target-2 silently consumed target-1's fit-transform output. The sentinel
# below is a unique-per-invocation marker; default identity equality / hash make two instances
# NEVER compare equal even when wrapping the same input id.
class _UncachableSentinel:
    """Per-instance identity marker; two instances NEVER compare equal under default object
    eq/hash, so cache keys built around two sentinels cannot collide."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return "<UncachableSentinel>"


def _fresh_uncachable() -> tuple:
    """Return a never-equal-to-anything-else sentinel tuple. Used in place of the unsafe
    ``("uncached", id(arr))`` which collided cross-target."""
    return ("uncached", _UncachableSentinel())


def _content_fingerprint_for_cache(arr) -> tuple:
    """Content-based fingerprint of an array / DataFrame / target.

    id()-keying is unsafe: GC recycles ids and the suite's per-target loop
    persists ``filtered_train_df`` across targets with the same id() so
    target-2 would otherwise reuse target-1's fit-transform output. The
    fingerprint folds in (n_rows, n_cols, per-column dtypes, column names)
    as cheap drift-detection signature, then point-samples 4 whole rows
    (first, near-head, midpoint, last) without materialising the frame --
    a polars row is a tuple pulled by direct column indexing, so cost is
    O(n_cols) regardless of n_rows. The previous ``arr.to_numpy()`` path
    materialised the entire frame just to slice 10 cells, defeating the
    very cache the per-target loop relies on (fingerprint cost > cache
    benefit on 100+ GB frames). Falls back to ``("uncached", id)`` so a
    failure forces a miss rather than a wrong-cache hit.
    """
    if arr is None:
        return ("none",)
    try:
        # Polars DataFrame: row(i) is a tuple of n_cols scalars -- O(n_cols), no full materialisation.
        if pl is not None and isinstance(arr, pl.DataFrame):
            n_rows, n_cols = int(arr.height), int(arr.width)
            col_names = tuple(str(c) for c in arr.columns)
            dtypes_str = tuple(str(dt) for dt in arr.dtypes)
            # Nested dtypes (List / Array / Struct) yield Python list/dict cells inside `arr.row(i)`;
            # lists are unhashable, so the fingerprint tuple cannot be used as a dict key. Force-uncached
            # on detection -- embedding columns are the common trigger (warning fired in
            # get_pandas_view_of_polars_df) and the cache contract is content-isolated regardless.
            if any(s.startswith(("List", "Array", "Struct")) for s in dtypes_str):
                return _fresh_uncachable()
            if n_rows == 0:
                return ("pl", (n_rows, n_cols), col_names, dtypes_str, ())
            sample_idx = (0, min(8, n_rows - 1), n_rows // 2, n_rows - 1)
            try:
                rows = tuple(arr.row(i) for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pl", (n_rows, n_cols), col_names, dtypes_str, rows)

        # Polars Series: iloc-equivalent is ``arr[i]`` -- O(1), no full materialisation.
        if pl is not None and isinstance(arr, pl.Series):
            n = int(arr.len())
            dtype_str = str(arr.dtype)
            # List / Array / Struct dtypes yield Python list/dict cells; lists are unhashable so the
            # fingerprint tuple cannot be used as a dict key. Force-uncached on detection.
            if dtype_str.startswith(("List", "Array", "Struct")):
                return _fresh_uncachable()
            if n == 0:
                return ("pls", (n,), dtype_str, ())
            sample_idx = (0, min(8, n - 1), n // 2, n - 1)
            try:
                cells = tuple(arr[i] for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pls", (n,), dtype_str, cells)

        # Pandas DataFrame: .iat / .iloc[i].values -- O(n_cols), no full materialisation.
        if isinstance(arr, pd.DataFrame):
            n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
            col_names = tuple(str(c) for c in arr.columns)
            dtypes = tuple(str(dt) for dt in arr.dtypes)
            if n_rows == 0:
                return ("pd", (n_rows, n_cols), col_names, dtypes, ())
            sample_idx = (0, min(8, n_rows - 1), n_rows // 2, n_rows - 1)
            try:
                # object-dtype cells from pyarrow ListArray materialise as Python lists -- unhashable.
                # Coerce row cells to a hashable form: leave scalars alone, repr() any list/dict/ndarray.
                # Audit D L-7 (2026-05-18): ``repr()`` on a 100-element numpy array is O(n_elements)
                # in characters; on a List-of-floats embedding column each repr is hundreds of
                # chars. We sample only 4 rows so the worst-case cost is 4 × O(embedding_len)
                # chars per fingerprint call -- bounded and small relative to the cache-hit
                # savings; a blake2b pre-hash would compact it further but is not currently
                # actionable (only 4-row bound makes this <<1ms even on wide embedding frames).
                def _row_to_hashable(r):
                    out = []
                    for v in r.values.tolist():
                        if isinstance(v, (list, dict)) or hasattr(v, "tolist"):
                            out.append(repr(v))
                        else:
                            out.append(v)
                    return tuple(out)
                rows = tuple(_row_to_hashable(arr.iloc[i]) for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pd", (n_rows, n_cols), col_names, dtypes, rows)

        # Pandas Series: .iat[i] -- O(1).
        if isinstance(arr, pd.Series):
            n = int(arr.shape[0])
            dtype_str = str(arr.dtype)
            if n == 0:
                return ("pds", (n,), dtype_str, ())
            sample_idx = (0, min(8, n - 1), n // 2, n - 1)
            try:
                cells = tuple(arr.iat[i] for i in sample_idx)
            except Exception:
                return _fresh_uncachable()
            return ("pds", (n,), dtype_str, cells)

        # NumPy / array-like: a 1-D / 2-D array is already in RAM; the previous flat-index sample is fine.
        if isinstance(arr, np.ndarray):
            np_arr = arr
        elif hasattr(arr, "values") and not hasattr(arr, "to_numpy"):
            np_arr = arr.values
        else:
            np_arr = np.asarray(arr)
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return _fresh_uncachable()
        shape = tuple(int(s) for s in np_arr.shape)
        dtype_str = str(np_arr.dtype)
        flat = np_arr.ravel()
        n = int(flat.size)
        if n == 0:
            return ("np", shape, dtype_str, b"")
        idx = [int(i * (n - 1) / 9) for i in range(10)] if n >= 10 else list(range(n))
        try:
            sampled = bytes(np.ascontiguousarray(flat[idx]).tobytes())
        except Exception:
            return _fresh_uncachable()
        return ("np", shape, dtype_str, sampled)
    except Exception:
        return _fresh_uncachable()


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
            # Even when the train/val path took a cache hit
            # (``skip_pre_pipeline_transform=True``), test_df MUST be
            # transformed if the pre_pipeline is fitted AND changes the
            # data (not identity-equivalent). A prod crash hit this
            # precisely: ``cached_dfs`` carried raw test_df with NaN;
            # ``skip_pre_pipeline_transform`` was True; LinearRegression
            # received NaN -> ValueError. Tree models (CB/XGB/LGB)
            # handle NaN natively, so they were fine.
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
    # us in fuzz testing: LinearModelStrategy's pre_pipeline had
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
    _collapsed = (arr.sum(axis=1) > 0).astype(np.int8)
    if len(np.unique(_collapsed)) > 1:
        return _collapsed
    # "Any positive label" degenerates to all-1 (or all-0) when every row has at
    # least one positive across K labels - happens for sparse multilabel with
    # K>=3 and base rates ~50%. Fall back to the per-label signal that itself
    # has the most variation (closest to 50/50 split). MRMR / TargetEncoder both
    # need at least 2 unique values; degrading to per-label is the smallest
    # information loss vs raising or returning a constant column.
    for _j in np.argsort(np.abs(arr.mean(axis=0) - 0.5)):  # most-balanced first
        _col = arr[:, _j].astype(np.int8)
        if len(np.unique(_col)) > 1:
            return _col
    return _collapsed


def _passthrough_cols_fit_transform(fn, df, *args, passthrough_cols=None, fit=False, target=None, groups=None, sample_weight=None):
    """Run a selector fit/transform on df with passthrough_cols hidden, then re-attach them.

    Feature selectors (MRMR, RFECV) can't encode text or list-of-float embedding columns;
    catboost needs them back intact for fit. Hide - run - re-attach preserves both.

    Numpy-output fallback: if the inner ``fn`` is a default sklearn
    Pipeline (no ``set_output(transform="pandas")``), ``out`` comes back as a numpy
    array. The original code detected this via ``hasattr(out, "columns")`` and
    silently returned numpy - dropping ``passthrough_cols`` and, worse, collapsing
    pd.Categorical dtypes in the selected columns to numpy object strings, which
    crashes LGB's Dataset construction on the ``'HOURLY'`` path. We now rebuild a
    pd.DataFrame from the reduced-input column names so passthrough_cols re-attach
    and downstream models take the native-pandas fastpath.

    ``groups`` is threaded through fit/fit_transform so GroupKFold-aware feature
    selectors (RFECV with cv=GroupKFold(), MRMR with grouped CV) receive the
    sample-grouping signal.

    ``sample_weight`` is threaded analogously into fit/fit_transform when the
    underlying selector is stamped with ``_mlframe_use_sample_weights_in_fs_ = True``
    (set by ``_build_pre_pipelines`` when ``FeatureSelectionConfig.use_sample_weights_in_fs``
    is True). MRMR.fit and RFECV.fit both accept ``sample_weight`` as a kwarg; the
    selector receives the active suite-level weights so selected features reflect
    the weight schema (recency / fairness / class balance).
    """

    # Convert sklearn/polars "empty output" errors into
    # an empty-frame return so the downstream ``train_df.shape[1] == 0``
    # guard at trainer.py:4515 fires cleanly. Triggered by MRMR / RFECV
    # confirming 0 predictors: ``fit_transform`` returns ``(N, 0)`` which
    # crashes in either:
    #   - SimpleImputer / scaler: ``ValueError: need at least one array
    #     or dtype is required`` (numpy.find_common_type on empty list).
    #   - sklearn's pandas-container wrap of a polars ``(N, 0)`` frame:
    #     ``ValueError: need at least one array to concatenate``
    #     (polars.to_numpy -> vstack on empty list).
    # Surfaced by MRMR fuzz axes (c0008 with interactions_max_order=3
    # + fe_max_steps=2 confirming 0 predictors).
    # Resolve the underlying selector instance so we can read the weight-aware marker.
    # ``fn`` is typically a bound method (``selector.fit_transform``); fall back to None when not applicable.
    _selector = getattr(fn, "__self__", None)
    _wants_sw = bool(sample_weight is not None and getattr(_selector, "_mlframe_use_sample_weights_in_fs_", False))

    def _call_fit(_fn, _arg, _target_arg):
        """Invoke fit/fit_transform with ``groups`` / ``sample_weight`` when supported.

        sklearn feature selectors that consume groups (RFECV(cv=GroupKFold()), MRMR)
        accept ``groups`` as a kwarg on ``fit`` / ``fit_transform``. MRMR.fit and
        RFECV.fit also accept ``sample_weight``; the marker
        ``_mlframe_use_sample_weights_in_fs_`` gates whether to forward it.
        Best-effort: if the underlying transformer does not accept a kwarg we retry
        without it instead of failing.

        sklearn Pipeline.fit raises ``ValueError`` (not ``TypeError``) when an
        unrecognised kwarg like ``groups`` is passed; the routing layer rejects
        non-step-namespaced parameters with a "Pipeline.fit does not accept"
        message. Treat that as the same "does not consume groups" signal as
        a plain ``TypeError`` from a bare transformer.
        """
        _kwargs = {}
        if groups is not None:
            _kwargs["groups"] = groups
        if _wants_sw:
            _kwargs["sample_weight"] = sample_weight
        if not _kwargs:
            return _fn(_arg, _target_arg)
        try:
            return _fn(_arg, _target_arg, **_kwargs)
        except TypeError:
            # Drop sample_weight first (more selectors accept groups than sw); then drop groups.
            if "sample_weight" in _kwargs:
                _kwargs.pop("sample_weight")
                if _kwargs:
                    try:
                        return _fn(_arg, _target_arg, **_kwargs)
                    except TypeError:
                        pass
            return _fn(_arg, _target_arg)
        except ValueError as _exc:
            _msg = str(_exc)
            if (
                "does not accept the groups parameter" in _msg
                or "got an unexpected keyword argument" in _msg
                or "unexpected keyword argument 'groups'" in _msg
                or "unexpected keyword argument 'sample_weight'" in _msg
            ):
                if "sample_weight" in _kwargs:
                    _kwargs.pop("sample_weight")
                    if _kwargs:
                        try:
                            return _fn(_arg, _target_arg, **_kwargs)
                        except (TypeError, ValueError):
                            pass
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
            # Bare ``held.to_pandas()`` consolidates Arrow buffers (~30x slower on wide frames + degrades pl.Enum
            # to object dtype). Route through the project's Arrow split-blocks bridge so passthrough columns
            # preserve their CategoricalDtype / DatetimeTZDtype etc.
            if is_polars:
                from .utils import get_pandas_view_of_polars_df as _get_pandas_view
                held_pd = _get_pandas_view(held)
            else:
                held_pd = held
            # Single ``pd.concat`` instead of per-column
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
        # Single concat instead of per-column assignment loop.
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


def _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target=None, target_name=None, sample_weight=None):
    """Compose a CONTENT-based cache key.

    id()-keying was unsafe on two axes: GC-recycled ids can collide and
    the per-target loop re-uses ``filtered_train_df`` (same id) across
    different targets - the second target would otherwise see the first
    target's fit-transform output. Including the target fingerprint AND
    the target name guarantees per-target isolation.

    Audit D P1-3 (2026-05-18): the target-fingerprint AND target-name are deliberately BOTH
    folded for defence-in-depth, NOT because both are required for correctness. Either alone is
    sufficient: the content fingerprint detects any cell-level divergence and the name detects
    name-only swaps where two targets happen to share content. Keeping both means a future
    refactor that drops one does not silently re-enable cross-target contamination. Do NOT
    refactor to a single dimension assuming the other is redundant.

    ``sample_weight`` is folded only when the inner selector is marked
    weight-aware (``_mlframe_use_sample_weights_in_fs_``); otherwise FS is
    weight-invariant and the cache stays valid across weight schemas.
    Weight fingerprinting uses the cheap 10-cell sampler shared with other
    content-based keys so the cost is O(1) regardless of n_rows.
    """
    sig = _pipeline_signature_for_cache(pipeline)
    _wants_sw = False
    try:
        # Walk the pipeline to find a selector with the marker set.
        if pipeline is not None:
            if hasattr(pipeline, "_mlframe_use_sample_weights_in_fs_"):
                _wants_sw = bool(getattr(pipeline, "_mlframe_use_sample_weights_in_fs_", False))
            elif hasattr(pipeline, "steps"):
                for _, _step in pipeline.steps:
                    if getattr(_step, "_mlframe_use_sample_weights_in_fs_", False):
                        _wants_sw = True
                        break
    except Exception:
        _wants_sw = False
    _sw_fp = _content_fingerprint_for_cache(sample_weight) if (_wants_sw and sample_weight is not None) else ("no_sw",)
    return (
        _content_fingerprint_for_cache(train_df),
        _content_fingerprint_for_cache(val_df),
        _content_fingerprint_for_cache(train_target),
        str(target_name) if target_name is not None else "",
        sig,
        _sw_fp,
    )


def _pre_pipeline_cache_get(train_df, val_df, pipeline, train_target=None, target_name=None, cache_max: int | None = None, sample_weight=None):
    """LRU-touch lookup; returns ``(train_out, val_out)`` or ``None``."""
    if train_df is None or pipeline is None:
        return None
    key = _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target, target_name, sample_weight=sample_weight)
    with _PRE_PIPELINE_CACHE_LOCK:
        if key in _PRE_PIPELINE_CACHE:
            _PRE_PIPELINE_CACHE.move_to_end(key)
            return _PRE_PIPELINE_CACHE[key]
    return None


def _pre_pipeline_cache_set(train_df, val_df, pipeline, train_out, val_out, train_target=None, target_name=None, cache_max: int | None = None, sample_weight=None):
    """Insert under LRU, evicting the oldest entry if over capacity.

    ``cache_max`` overrides the module default; pass through from the
    caller's ``TrainingBehaviorConfig.pre_pipeline_cache_max`` so
    long-running services can tune memory vs hit-rate.
    """
    if train_df is None or pipeline is None:
        return
    key = _pre_pipeline_cache_key(train_df, val_df, pipeline, train_target, target_name, sample_weight=sample_weight)
    _cap = int(cache_max) if cache_max is not None else _PRE_PIPELINE_CACHE_MAX
    with _PRE_PIPELINE_CACHE_LOCK:
        # Store the pipeline as third element so future hits can transfer fit state.
        _PRE_PIPELINE_CACHE[key] = (train_out, val_out, pipeline)
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
    sample_weight=None,
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
        # Duplicate-pipeline skip: capture input column names
        # BEFORE transform so we can detect identity-equivalent pipelines
        # (selected all columns, created none). Set on the pre_pipeline
        # instance so the suite loop in core/main.py can skip redundant
        # branches.
        _input_cols = (
            list(train_df.columns) if hasattr(train_df, "columns") else None
        )
        # Structurally-identical-pipeline cache. When the
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
            sample_weight=sample_weight,
        )
        _cache_hit = _pre_pipeline_cache_get(
            train_df, val_df, pre_pipeline,
            train_target=train_target, target_name=target_name,
            sample_weight=sample_weight,
        )
        if _cache_hit is not None and not skip_pre_pipeline_transform and not skip_preprocessing and not _is_fitted(pre_pipeline):
            _cache_entry = _cache_hit
            # Older entries hold a 2-tuple; new entries hold (train, val, fitted_pipeline).
            if len(_cache_entry) == 3:
                train_df_cached, val_df_cached, fitted_cached = _cache_entry
            else:
                train_df_cached, val_df_cached = _cache_entry
                fitted_cached = None
            # Transfer fit state from cached pipeline so the caller's (cloned, unfitted) instance
            # can transform test_df at predict time. Without this, _prepare_test_split called
            # pre_pipeline.transform on an unfitted Pipeline and raised NotFittedError (surfaced
            # by test_train_mixed_linear_and_lgb after value-transforming pipelines
            # stopped being marked identity-equivalent).
            if fitted_cached is not None:
                try:
                    # Audit D P1-8 (2026-05-18): a shallow ``__dict__.update`` shares mutable
                    # state between the cached pipeline and pre_pipeline -- any per-call mutation
                    # (e.g. ``set_params`` on a wrapper, or a transformer that lazily mutates
                    # ``self.fitted_state_``) silently corrupts the cached copy because both
                    # instances point to the SAME dict entries. Shallow-copy the dict and copy
                    # the fitted attributes; downstream sklearn transforms only read fitted
                    # attributes, not mutate them, so a shallow attribute-copy is enough.
                    # ``copy.copy`` per-attribute is cheaper than ``copy.deepcopy`` (which
                    # duplicates trained model weights -- ranges in GB for tree ensembles).
                    import copy as _cp
                    for _k, _v in fitted_cached.__dict__.items():
                        try:
                            pre_pipeline.__dict__[_k] = _cp.copy(_v)
                        except Exception:
                            # Defensive: a non-copyable attribute (e.g. file handle wrapper)
                            # falls back to the original reference. Logged at debug because the
                            # standard sklearn fitted attributes are all copyable.
                            pre_pipeline.__dict__[_k] = _v
                except Exception as _state_err:
                    logger.debug("pre_pipeline cache state transfer skipped: %s", _state_err)
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
                            sample_weight=sample_weight,
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
                    # The historical 0-row val skip has been removed.
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
                # Supervised encoders (category_encoders
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
                    sample_weight=sample_weight,
                )
                # One-glance FS-retention log so operators don't need to crack open metadata pickles
                # to see how many columns the selector kept. Reports the selector's class name when
                # detectable, falls back to the pipeline repr; harmless on no-op pipelines.
                if verbose:
                    try:
                        _kept = train_df.shape[1] if hasattr(train_df, "shape") and len(train_df.shape) == 2 else None
                        _input_n = len(_input_cols) if _input_cols is not None else None
                        if _kept is not None and _input_n is not None:
                            _selector = _extract_feature_selector(pre_pipeline)
                            _selector_label = type(_selector).__name__ if _selector is not None else type(pre_pipeline).__name__
                            logger.info("  FS selector %s retained %d of %d features", _selector_label, _kept, _input_n)
                    except Exception:
                        pass
                if verbose:
                    log_ram_usage()
                # 0-feature short-circuit: when MRMR/RFECV selects no features,
                # _passthrough_cols_fit_transform catches Pipeline's "need at least one
                # array" ValueError and returns an empty (N, 0) frame -- BUT the Pipeline
                # is left half-fitted (selector fitted, imputer/scaler not). Running
                # pre_pipeline.transform on val_df then raises NotFittedError. Mirror the
                # empty-frame return on val_df so trainer.py's 0-feature guard can fire
                # cleanly. (Covered by test_mrmr_no_impact_classification
                # which uses min_relevance_gain=10.0 to force 0 features.)
                _train_is_empty = (
                    hasattr(train_df, "shape") and len(train_df.shape) == 2 and train_df.shape[1] == 0
                )
                if val_df is not None and _train_is_empty:
                    if verbose:
                        logger.info(
                            "Skipping val_df transform: train_df has 0 features after fit (selector "
                            "rejected all). Returning empty (N, 0) val_df to match.",
                        )
                    if pl is not None and isinstance(val_df, pl.DataFrame):
                        val_df = val_df.select([])
                    elif hasattr(val_df, "iloc"):
                        val_df = val_df.iloc[:, :0]
                elif val_df is not None:
                    if verbose:
                        logger.info("Transforming val_df via pre_pipeline %s...", pre_pipeline)
                    # The historical 0-row val skip has been removed --
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
            # Populate the LRU cache so the next sklearn-non-native
            # model in this per-target iteration (typically MLP
            # after Linear) gets a cache hit on the same train_df + the
            # same structural pipeline. Guarded: only stash when we
            # actually went through the fit-transform branch (the others
            # didn't have anything new to cache anyway).
            if not skip_pre_pipeline_transform and not skip_preprocessing:
                # Reuse the entry-time key so a populate after a miss is guaranteed to land in the slot the next lookup will read from.
                # Store the now-fitted pre_pipeline alongside (train_df, val_df); future cache hits
                # transfer fit state to the caller's cloned instance so test_df.transform works.
                try:
                    _cap = int(cache_max) if cache_max is not None else _PRE_PIPELINE_CACHE_MAX
                    with _PRE_PIPELINE_CACHE_LOCK:
                        _PRE_PIPELINE_CACHE[_cache_key_entry] = (train_df, val_df, pre_pipeline)
                        _PRE_PIPELINE_CACHE.move_to_end(_cache_key_entry)
                        while len(_PRE_PIPELINE_CACHE) > _cap:
                            _PRE_PIPELINE_CACHE.popitem(last=False)
                except Exception as _cache_err:
                    logger.debug(
                        "pre_pipeline cache populate skipped: %s",
                        _cache_err,
                    )

        # Identity-equivalent = column set unchanged AND no value-transforming
        # steps (ce/imp/scaler/transform). Column-list-equality alone gave a
        # false positive for LinearModelStrategy's encoder+imputer+scaler chain:
        # those steps preserve column names while remapping string cats to
        # numeric, so the suite skipped pre_pipeline.transform on test_df and
        # LogisticRegression got raw string cats at predict time (TypeError in
        # safe_sparse_dot, surfaced by fuzz_regression_sensors).
        if _input_cols is not None and hasattr(train_df, "columns"):
            _output_cols = list(train_df.columns)
            _cols_same = _input_cols == _output_cols
            _has_value_transforms = False
            _named = getattr(pre_pipeline, "named_steps", None)
            if _named:
                for _step_name in ("ce", "imp", "scaler", "transform"):
                    if _named.get(_step_name) is not None:
                        _has_value_transforms = True
                        break
            try:
                pre_pipeline._mlframe_identity_equivalent = (
                    _cols_same and not _has_value_transforms
                )
            except Exception:
                pass  # non-writable pre_pipeline (e.g. tuple), safe to ignore

    return train_df, val_df
