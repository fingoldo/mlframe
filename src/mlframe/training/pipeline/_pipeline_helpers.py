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

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from sklearn.pipeline import Pipeline

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
    _content_fingerprint_for_cache,
    _fresh_uncachable,
    _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_key,
    _pre_pipeline_cache_set,
    _UncachableSentinel,
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
    from .._data_helpers import _extract_target_subset, _subset_dataframe
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
                    and _test_df_is_raw_pipeline_input(pre_pipeline, test_df, selector_passthrough_cols, skip_preprocessing)
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


def _raise_pre_pipeline_rowcount_change(n_in: int, n_out: int) -> None:
    """Raise when a pre_pipeline changed the train row count (row-preserving-slot violation)."""
    raise ValueError(
        f"pre_pipeline changed the train row count ({n_in:_} -> {n_out:_}). The pre_pipeline slot is "
        f"row-preserving: a resampler (imblearn SMOTE / RandomOver/UnderSampler / FunctionSampler) "
        f"cannot be used here -- target and sample_weight are NOT resampled in lockstep, misaligning "
        f"X and y at model fit. Use a model-level imbalance knob (lgb/xgb scale_pos_weight / "
        f"is_unbalance, catboost auto_class_weights, sklearn class_weight='balanced')."
    )


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
    # Row-preserving-slot guard: a resampler makes ``out`` grow/shrink vs ``held``, misaligning the
    # positional re-attach below. Raise the shared actionable message (not an opaque pandas error).
    _out_n = out.shape[0] if hasattr(out, "shape") and len(getattr(out, "shape", ())) >= 1 else None
    _held_n = held.shape[0] if hasattr(held, "shape") and len(getattr(held, "shape", ())) >= 1 else None
    if _out_n is not None and _held_n is not None and _out_n != _held_n:
        _raise_pre_pipeline_rowcount_change(_held_n, _out_n)
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


# _apply_pre_pipeline_transforms / _validate_pre_pipeline_output_against_model carved to _pipeline_helpers_apply.py (1k-LOC ceiling).
from ._pipeline_helpers_apply import (  # noqa: E402, F401
    _apply_pre_pipeline_transforms,
    _validate_pre_pipeline_output_against_model,
)
