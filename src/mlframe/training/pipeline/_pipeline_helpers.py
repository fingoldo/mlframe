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

from ..utils import maybe_clean_ram_adaptive as _maybe_clean_ram
from ..phases import phase

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from sklearn.pipeline import Pipeline

from ..utils import log_ram_usage

logger = logging.getLogger(__name__)


# Wave 93 (2026-05-21): Pipeline structural-identity + content cache moved
# to sibling file _pipeline_cache.py to drop _pipeline_helpers.py below
# the 1k-line monolith threshold. Re-exported here so existing callers
# (`from ._pipeline_helpers import _content_fingerprint_for_cache`, etc.)
# keep working.
from ._pipeline_cache import (  # noqa: F401, E402
    _PRE_PIPELINE_CACHE,
    _PRE_PIPELINE_CACHE_LOCK,
    _PRE_PIPELINE_CACHE_MAX,
    _UncachableSentinel,
    _fresh_uncachable,
    _content_fingerprint_for_cache,
    _pipeline_signature_for_cache,
    _pre_pipeline_cache_key,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_set,
    _pre_pipeline_cache_clear,
)


def _selector_output_columns(selector):
    """The OUTPUT (selected) column names of a fitted feature selector, or None.

    A name-keyed selector (RFECV / MRMR / BorutaShap) records the columns it was fit on in ``feature_names_in_`` and the kept subset in
    ``support_`` (a boolean mask) or via ``get_support()``. Its ``transform`` selects those output columns BY NAME, so the count of selected output
    columns is the reliable width of an already-transformed frame -- independent of the selector's recorded INPUT width, which a zero-variance
    pre-filter can shrink below the raw frame width. Returns the list of selected output names, or None when the selector doesn't expose a
    name-keyed support contract (then the caller falls back to the input-width heuristic).
    """
    if selector is None:
        return None
    names_in = getattr(selector, "feature_names_in_", None)
    if names_in is None:
        return None
    support = getattr(selector, "support_", None)
    if support is None and hasattr(selector, "get_support"):
        try:
            support = selector.get_support()
        except Exception:
            support = None
    if support is None:
        return None
    try:
        support = np.asarray(support)
        names_in = list(names_in)
        if support.dtype == bool or (support.size and isinstance(support.flat[0], (bool, np.bool_))):
            return [c for c, keep in zip(names_in, support) if keep]
        return [names_in[int(i)] for i in support]
    except Exception:
        return None


def _test_df_is_raw_pipeline_input(pre_pipeline, test_df, passthrough_cols, skip_preprocessing) -> bool:
    """True when ``test_df`` still needs the fitted pipeline's transform applied (it carries the RAW input schema, or any schema wider than the
    pipeline's transformed OUTPUT), False only when it is already at the pipeline's transformed OUTPUT width (so re-transforming would double-apply).

    Primary discriminator -- the feature selector's OUTPUT width. A fitted selector (RFECV / MRMR / BorutaShap) reduces its input to the columns in
    ``feature_names_in_[support_]``, so a frame WIDER than that selected output still needs the selection applied; a frame already at (or below) the
    output width has been transformed and must not be re-selected. This is immune to the input-width ambiguity that broke the prior ``n_features_in_``
    heuristic: a selector applies a zero-variance pre-filter at fit entry, so ``n_features_in_`` (e.g. 6) can be SMALLER than the raw frame width
    (e.g. 7) -- making a raw 7-col frame compare unequal to ``n_features_in_=6`` and be misclassified as already-transformed, so its transform was
    skipped and the model (trained on the 4 selected cols) received the raw 7-col frame -> ``LightGBMError: number of features in data (7) != training
    (4)`` (fuzz c0026: use_mrmr_fs + rfecv on cb/lgb/linear). The output width has no such ambiguity: raw frames are strictly wider than the output.

    Fallback discriminator -- the fitted estimator's ``n_features_in_`` width (the legacy heuristic), used only when no name-keyed selector output
    set is available (e.g. a pure preprocessing pipeline with no selector). Defaults to True (transform) whenever neither signal is determinable,
    so the NaN-to-LinearRegression guard (the original reason the fitted-pipeline override exists) is preserved on any ambiguous frame.
    """
    if not hasattr(test_df, "shape"):
        return True
    _n_passthrough = len(passthrough_cols) if passthrough_cols else 0
    _effective_width = int(test_df.shape[1]) - _n_passthrough

    # Primary discriminator -- the feature selector's OUTPUT width. A selector reduces its input to ``len(support_[support_])`` columns, so a frame
    # WIDER than that output still needs the selection applied (raw, or carrying upstream-dropped cols); a frame already AT (or below) the output
    # width has been transformed and must not be re-selected. The output width is robust where the input-width heuristic below is not: a selector
    # applies a zero-variance pre-filter at fit entry, so its ``n_features_in_`` can be SMALLER than the raw frame width -- which made the raw frame
    # compare unequal to ``n_features_in_`` and be misclassified as already-transformed (the c0026 4-vs-7 LightGBMError). The output width has no
    # such ambiguity: raw frames are strictly wider than the selected output.
    _sel = _extract_feature_selector(pre_pipeline)
    _out_cols = _selector_output_columns(_sel)
    if _out_cols is not None:
        return _effective_width > len(_out_cols)

    # Fallback: input-width heuristic. The estimator whose input width matters is the one the transform feeds: the feature selector under
    # skip_preprocessing, else the whole pipeline.
    _est = pre_pipeline
    if skip_preprocessing and _sel is not None:
        _est = _sel
    _expected_in = getattr(_est, "n_features_in_", None)
    if _expected_in is None and hasattr(_est, "named_steps"):
        # Pipeline: the first step carries the input-width contract.
        for _step in getattr(_est, "named_steps", {}).values():
            _expected_in = getattr(_step, "n_features_in_", None)
            if _expected_in is not None:
                break
    if _expected_in is None:
        return True  # can't tell -> preserve the transform (NaN-guard posture)
    # Raw input -> widths match; already-transformed -> diverged.
    return _effective_width == int(_expected_in)


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
    from .._data_helpers import _subset_dataframe, _extract_target_subset
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
            # ``_mlframe_identity_equivalent`` is computed on the TRAIN transform
            # and stored on the pipeline object; it can go STALE when a pipeline /
            # bare feature-selector object is reused across rounds (e.g. a non-FS
            # round sets it True, then the MRMR round reuses the same tree
            # pipeline). A fitted feature selector REDUCES the column set, so its
            # transform is never a no-op: when one is present AND the test frame
            # still carries the raw input width, force the transform on. The
            # ``is_raw`` guard means an already-transformed test frame (skip-path
            # cache hit, narrower than the fitted input width) is left untouched,
            # so this never double-transforms; unfitted-identity pipelines (the
            # NotFittedError-dodge case) are excluded by the ``_is_fitted`` check.
            if _id_equiv and _is_fitted(pre_pipeline):
                _sel_chk = _extract_feature_selector(pre_pipeline)
                if (
                    _sel_chk is not None
                    and _is_fitted(_sel_chk)
                    and _test_df_is_raw_pipeline_input(
                        pre_pipeline, test_df, selector_passthrough_cols, skip_preprocessing
                    )
                ):
                    _id_equiv = False
            if not _id_equiv:
                # ``skip_pre_pipeline_transform=True`` is set on the pipeline-
                # cache hit, where the cached test_df is ALREADY the transformed
                # output (process_model cached train/val/test as one generation).
                # Re-running the fitted pipeline on that transformed frame
                # double-transforms it and raises sklearn's "Unexpected input
                # dimension X, expected Y" (Y = raw input width the pipeline was
                # fit on, X = the already-reduced output width) -- or, for a
                # column-count-preserving scaler pipeline, re-scales and trips
                # the finite check on the already-transformed values. Surfaced
                # by fuzz (5 linear/mlp combos after a tree model populated the
                # cache).
                #
                # But the override exists for a real prod incident: when skip is
                # True yet test_df is still RAW (carries the pipeline's input
                # schema, e.g. NaN cells), a fitted pipeline MUST transform it
                # or NaN reaches LinearRegression (test_pre_pipeline_applied_to_
                # test.py). Discriminate the two by feature width: a raw frame
                # matches the pipeline's fitted ``n_features_in_`` (minus the
                # hidden passthrough cols); an already-transformed frame does
                # not. Only re-transform a fitted pipeline under the skip flag
                # when the frame still looks like raw pipeline input.
                _fitted = _is_fitted(pre_pipeline)
                if not skip_pre_pipeline_transform:
                    _do_transform = True
                elif _fitted and _test_df_is_raw_pipeline_input(
                    pre_pipeline, test_df, selector_passthrough_cols, skip_preprocessing,
                ):
                    _do_transform = True
                else:
                    _do_transform = False
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
                if not _step_is_fitted(step):
                    return False
            return True
    except Exception:
        pass
    return _step_is_fitted(estimator)


def _step_is_fitted(step) -> bool:
    """``check_is_fitted`` for a single estimator, ignoring mlframe-private markers.

    The suite stamps selectors with trailing-underscore markers (``_mlframe_use_sample_weights_in_fs_`` etc.). sklearn's
    default ``check_is_fitted`` heuristic treats ANY trailing-underscore attribute as fitted state, so a freshly-cloned
    selector carrying only those markers would be misreported as fitted -> the suite skips the fit -> NotFittedError on
    transform. Restrict the fitted check to genuine sklearn-fitted attributes by excluding the ``_mlframe_*`` markers.
    """
    if step is None:
        return False
    # Custom ``__sklearn_is_fitted__`` (some wrappers define it) is authoritative; honour it before the attr heuristic.
    if hasattr(step, "__sklearn_is_fitted__"):
        try:
            check_is_fitted(step)
            return True
        except NotFittedError:
            return False
        except Exception:
            pass
    try:
        fitted_attrs = [a for a in vars(step) if a.endswith("_") and not a.startswith("__") and not a.startswith("_mlframe_")]
    except TypeError:
        fitted_attrs = None
    if fitted_attrs is not None and not fitted_attrs:
        # Only mlframe markers (or nothing) end in ``_`` -> genuinely unfitted; sklearn's default heuristic would be
        # fooled by the trailing-underscore markers into reporting fitted.
        return False
    try:
        if fitted_attrs:
            check_is_fitted(step, attributes=fitted_attrs)
        else:
            check_is_fitted(step)
        return True
    except NotFittedError:
        return False
    except Exception:
        try:
            check_is_fitted(step)
            return True
        except NotFittedError:
            return False


def _is_stale_fit_state_value_error(exc) -> bool:
    """True for the sklearn ValueErrors that signal a transformer was fitted on a
    DIFFERENT feature schema than the frame now being transformed -- the two
    variants are a width/count mismatch and a feature-name mismatch:

      * ``Unexpected input dimension N, expected M`` (_SetOutputMixin)
      * ``X has N features, but ... is expecting M features``
      * ``The feature names should match those that were passed during fit ...``
        / ``Feature names seen at fit time, yet now missing: ...`` /
        ``Feature names unseen at fit time: ...``

    Both are the same stale-fit-state failure mode as NotFittedError /
    AttributeError: a pipeline / feature-selector object reused across rounds
    carries fit state for a different input schema (e.g. fitted on the FS-reduced
    output of a prior model and now handed the raw frame). They warrant the same
    fit_transform recovery rather than crashing the run. Unrelated data
    ValueErrors (NaN, string-to-float, etc.) do NOT match, so they still
    propagate.
    """
    msg = str(exc).lower()
    return (
        "unexpected input dimension" in msg
        or "features, but" in msg
        or "is expecting" in msg
        or "feature names should match" in msg
        or "feature names seen at fit time" in msg
        or "feature names unseen at fit time" in msg
    )


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
    # When the selector is wrapped in an sklearn Pipeline (the common case: any preprocessing step wraps MRMR /
    # RFECV under the ``'pre'`` step), ``fn.__self__`` is the WRAPPING Pipeline -- which is never stamped -- so the
    # marker must be read off the inner ``'pre'`` step. A bare ``sample_weight`` kwarg also will not reach the inner
    # step through ``Pipeline.fit``; sklearn routes per-step fit_params via the ``<step>__param`` namespace, so the
    # kwarg key becomes ``pre__sample_weight`` when forwarding through a Pipeline.
    _bound = getattr(fn, "__self__", None)
    _selector = _extract_feature_selector(_bound) if _bound is not None else None
    _sw_via_pipeline = isinstance(_bound, Pipeline)
    _wants_sw = bool(sample_weight is not None and getattr(_selector, "_mlframe_use_sample_weights_in_fs_", False))
    _sw_kwarg = "pre__sample_weight" if _sw_via_pipeline else "sample_weight"

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
            _kwargs[_sw_kwarg] = sample_weight
        if not _kwargs:
            return _fn(_arg, _target_arg)
        try:
            return _fn(_arg, _target_arg, **_kwargs)
        except TypeError:
            # Drop sample_weight first (more selectors accept groups than sw); then drop groups.
            if _sw_kwarg in _kwargs:
                _kwargs.pop(_sw_kwarg)
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
                if _sw_kwarg in _kwargs:
                    _kwargs.pop(_sw_kwarg)
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
                from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
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
        # Mirror the sibling branch above (lines 373-378): bare ``held.to_pandas()`` consolidates Arrow buffers (~30x slower on wide frames + degrades pl.Enum to object dtype). Route through the project Arrow split-blocks bridge.
        if is_polars:
            from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
            held_pd = _get_pandas_view(held)
        else:
            held_pd = held
        # Single concat instead of per-column assignment loop.
        out = pd.concat([out, held_pd], axis=1)
    return out


# Wave 93 (2026-05-21): _pipeline_signature_for_cache /
# _pre_pipeline_cache_key / _get / _set / _clear moved to sibling file
# _pipeline_cache.py and re-exported from this module's top-level imports.


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
        # + ~18s mlp on a 4M-row prod log). Only fires on the fresh
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
            sample_weight=sample_weight, key=_cache_key_entry,
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
                except (AttributeError, TypeError) as _state_err:
                    # Whole-transfer failure (e.g. ``fitted_cached`` has no ``__dict__`` -- a
                    # ``__slots__``-only object). Unlike the per-attribute fallback above, this
                    # leaves ``pre_pipeline`` with NO fitted state, so the caller's predict-time
                    # ``transform`` will raise NotFittedError. Surface at WARNING -- a real
                    # incomplete state transfer must not be hidden at debug level.
                    logger.warning(
                        "pre_pipeline cache state transfer FAILED (%s: %s); pre_pipeline left "
                        "without fitted state -- predict-time transform may raise NotFittedError",
                        type(_state_err).__name__, _state_err,
                    )
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
                        # Fit-state mismatch fallback (iter-347 family): _is_fitted
                        # uses sklearn's check_is_fitted heuristic which can
                        # report True on a partially-state-transferred clone whose
                        # selector-specific attrs (BorutaShap.selected_features_,
                        # MRMR.support_) were not copied across. If transform then
                        # raises NotFittedError or AttributeError, retry via
                        # fit_transform so the suite recovers instead of dropping
                        # the model.
                        try:
                            train_df = _passthrough_cols_fit_transform(
                                feature_selector.transform,
                                train_df,
                                passthrough_cols=selector_passthrough_cols,
                            )
                        except (NotFittedError, AttributeError, ValueError) as _selector_state_exc:
                            if isinstance(_selector_state_exc, ValueError) and not _is_stale_fit_state_value_error(_selector_state_exc):
                                raise  # a genuine data ValueError, not a stale-fit-schema mismatch
                            if verbose:
                                logger.warning(
                                    "Pre-fitted feature selector %s raised %s on transform; "
                                    "falling back to fit_transform with current target+groups. "
                                    "This usually means the cache state transfer didn't replicate "
                                    "every selector-private attribute (e.g. BorutaShap.selected_features_) "
                                    "or the selector was fitted on a different input width.",
                                    type(feature_selector).__name__,
                                    type(_selector_state_exc).__name__,
                                )
                            train_df = _passthrough_cols_fit_transform(
                                feature_selector.fit_transform,
                                train_df,
                                passthrough_cols=selector_passthrough_cols,
                                fit=True,
                                target=train_target,
                                groups=groups,
                                sample_weight=sample_weight,
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
                # Fit-state mismatch fallback (iter-365 family, sibling of iter-347):
                # ``_is_fitted`` may return True on a partially-state-transferred
                # pipeline whose inner step (BorutaShap.selected_features_,
                # MRMR.support_) lacks the selector-private attribute the
                # ``transform`` reads. Catch NotFittedError / AttributeError
                # and recover via fit_transform so the suite doesn't drop the
                # model. Identical logic to the ``skip_preprocessing`` branch
                # above; both code paths reach the same failure mode.
                try:
                    train_df = _passthrough_cols_fit_transform(
                        pre_pipeline.transform,
                        train_df,
                        passthrough_cols=selector_passthrough_cols,
                    )
                except (NotFittedError, AttributeError, ValueError) as _pipeline_state_exc:
                    if isinstance(_pipeline_state_exc, ValueError) and not _is_stale_fit_state_value_error(_pipeline_state_exc):
                        raise  # a genuine data ValueError, not a stale-fit-schema mismatch
                    if verbose:
                        logger.warning(
                            "Pre-fitted pre_pipeline raised %s on transform; "
                            "falling back to fit_transform with current target+groups. "
                            "Likely cause: cache state transfer didn't replicate every "
                            "inner-step attribute (e.g. BorutaShap.selected_features_) or the "
                            "pipeline was fitted on a different input width.",
                            type(_pipeline_state_exc).__name__,
                        )
                    _enc_target_recover = _multilabel_target_to_1d_for_supervised_encoders(train_target)
                    train_df = _passthrough_cols_fit_transform(
                        pre_pipeline.fit_transform,
                        train_df,
                        passthrough_cols=selector_passthrough_cols,
                        fit=True,
                        target=_enc_target_recover,
                        groups=groups,
                        sample_weight=sample_weight,
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
                # to see how many columns the selector kept. Emitted at INFO regardless of verbose so
                # every selector's kept/dropped counts are visible in default logs (one line per FS fit).
                try:
                    _kept = train_df.shape[1] if hasattr(train_df, "shape") and len(train_df.shape) == 2 else None
                    _input_n = len(_input_cols) if _input_cols is not None else None
                    if _kept is not None and _input_n is not None:
                        _selector = _extract_feature_selector(pre_pipeline)
                        _selector_label = type(_selector).__name__ if _selector is not None else type(pre_pipeline).__name__
                        logger.info(
                            "FS selector %s retained %d of %d features (dropped %d)",
                            _selector_label, _kept, _input_n, max(_input_n - _kept, 0),
                        )
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

        # Validate the pre_pipeline output against what the model expects.
        # A mis-shaped pre_pipeline output (e.g. a custom step that drops a
        # column the fitted model needs) otherwise surfaces as an opaque
        # sklearn/booster error far from the cause ("X has N features, but ...
        # is expecting M" deep inside .predict). Raise an actionable error here
        # naming the missing/extra columns. Only validate when the model's
        # expected feature names are KNOWN (model already fitted, e.g. a reused
        # round); on the first fit they're unknown -- skip to avoid over-validation.
        _validate_pre_pipeline_output_against_model(model, pre_pipeline, train_df)

    return train_df, val_df


def _validate_pre_pipeline_output_against_model(model, pre_pipeline, train_df) -> None:
    """Raise an actionable error if the pre_pipeline output can't feed the model.

    Checks (a) the output is non-empty, and (b) -- when the model's expected
    feature names are known from a prior fit -- that the output column set
    matches, naming the missing/extra columns and the pre_pipeline step. Silent
    no-op when the expected names are unknown (first fit) so we don't reject a
    legitimate first-fit shape.
    """
    if train_df is None or not hasattr(train_df, "columns"):
        return
    output_cols = list(train_df.columns)

    # Non-empty check (always actionable: a 0-column frame can never feed a model).
    if len(output_cols) == 0:
        # 0-feature selector output is a distinct, already-handled scenario
        # (trainer.py's 0-feature guard); don't intercept it here.
        return

    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return  # model not yet fitted -> expected schema unknown, skip.
    expected_cols = list(expected)
    if not expected_cols:
        return

    output_set = set(output_cols)
    expected_set = set(expected_cols)
    if output_set == expected_set:
        return

    missing = [c for c in expected_cols if c not in output_set]
    extra = [c for c in output_cols if c not in expected_set]
    _step = type(pre_pipeline).__name__
    _last = getattr(pre_pipeline, "steps", None)
    if _last:
        _step = f"{_step}(last step '{_last[-1][0]}')"
    raise ValueError(
        f"pre_pipeline output ({len(output_cols)} cols) does not match the "
        f"{len(expected_cols)} features the fitted model expects. "
        f"Missing (expected by model, dropped by pre_pipeline): {missing}. "
        f"Extra (produced by pre_pipeline, unknown to model): {extra}. "
        f"Check the pre_pipeline step {_step}."
    )
