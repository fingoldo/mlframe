"""Fit-state checks and pass-through reattachment for the pre-pipeline, kept in a leaf module.

Leaf module: the pre-pipeline helpers and their apply sibling both need these, and this module imports nothing from the package.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from mlframe.utils.log_throttle import log_throttle

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

logger = logging.getLogger("mlframe.training.pipeline._pipeline_helpers")


def _extract_feature_selector(pre_pipeline: Any) -> Any:
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


def _is_fitted(estimator: Any) -> bool:
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
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed: %s", e)
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
        except NotFittedError as _exc:
            log_throttle(logger, "_pipeline_fit_state._step_is_fitted", logging.WARNING, "_step_is_fitted fell back to its default after %s", type(_exc).__name__)
            return False
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
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
    except NotFittedError as _exc:
        log_throttle(logger, "_pipeline_fit_state._step_is_fitted", logging.WARNING, "_step_is_fitted fell back to its default after %s", type(_exc).__name__)
        return False
    except Exception as exc:
        log_throttle(logger, "_pipeline_fit_state._step_is_fitted", logging.WARNING, "_step_is_fitted: attribute-scoped check_is_fitted failed (%s); retrying unscoped.", exc)
        try:
            check_is_fitted(step)
            return True
        except NotFittedError as _exc:
            log_throttle(logger, "_pipeline_fit_state._step_is_fitted", logging.WARNING, "_step_is_fitted fell back to its default after %s", type(_exc).__name__)
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
            except Exception as exc:
                logger.debug("multilabel target collapse: per-row stack failed, target left unnormalized: %s", exc)
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


def _empty_selection_output(arg):
    """An empty frame or array with the row count of ``arg``, for a transformer that dropped every column."""
    if isinstance(arg, pl.DataFrame):
        return arg.select([])
    if hasattr(arg, "iloc"):
        return arg.iloc[:, :0]
    return np.empty((len(arg), 0))


def _row_count(obj):
    """Leading dimension of ``obj``, or None when it has no shape."""
    shape: tuple[int, ...] = getattr(obj, "shape", ())
    return shape[0] if len(shape) >= 1 else None


def _check_row_preserving(out, held) -> None:
    """Raise the shared actionable error when a resampler changed the row count, which would misalign the positional re-attach."""
    out_n = _row_count(out)
    held_n = _row_count(held)
    if out_n is not None and held_n is not None and out_n != held_n:
        _raise_pre_pipeline_rowcount_change(held_n, out_n)


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
    # guard in train_and_evaluate_model (_trainer_train_and_evaluate.py) fires cleanly. Triggered by MRMR / RFECV
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
        """Run the (possibly fit-)transform and, if it fails with numpy's "need at least one array to concatenate" (a transformer that dropped every column), swallow it and return an empty frame/array of matching row count so downstream 0-feature guards see a consistent shape instead of a raw crash."""
        try:
            return _call_fit(_fn, _arg, _target_arg) if fit else _fn(_arg)
        except ValueError as _exc:
            _m = str(_exc)
            if "need at least one array to concatenate" in _m or "at least one array or dtype is required" in _m:
                # Empty output (0 features); return an empty DataFrame of
                # matching shape so the suite's downstream 0-feature guard
                # catches it.
                return _empty_selection_output(_arg)
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
    _check_row_preserving(out, held)
    out = _reattach_passthrough_to_frame(out, present, held, is_polars, reduced)
    return out


def _reattach_passthrough_to_frame(out, present, held, is_polars, reduced):
    """Reattach the passthrough columns to the transformed frame."""
    if hasattr(out, "columns"):
        if isinstance(out, pl.DataFrame):
            out = out.with_columns([held[c] for c in present])
        else:
            # Bare ``held.to_pandas()`` consolidates Arrow buffers (~30x slower on wide frames + degrades pl.Enum
            # to object dtype). Route through the project's Arrow split-blocks bridge so passthrough columns
            # preserve their CategoricalDtype / DatetimeTZDtype etc.
            if is_polars:
                from mlframe.training.utils import get_pandas_view_of_polars_df as _get_pandas_view

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
            from mlframe.training.utils import get_pandas_view_of_polars_df as _get_pandas_view

            held_pd = _get_pandas_view(held)
        else:
            held_pd = held
        # Single concat instead of per-column assignment loop.
        out = pd.concat([out, held_pd], axis=1)
    return out
