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

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# Pipeline structural-identity + content cache moved
# to sibling file _pipeline_cache.py to drop _pipeline_helpers.py below
# the 1k-line monolith threshold. Re-exported here so existing callers
# (`from ._pipeline_helpers import _content_fingerprint_for_cache`, etc.)
# keep working.
from ._pipeline_cache import (  # noqa: F401
    _PRE_PIPELINE_CACHE,
    _PRE_PIPELINE_CACHE_LOCK,
    _PRE_PIPELINE_CACHE_MAX,
    _PRE_PIPELINE_CACHE_MAX_BYTES,
    _approx_entry_bytes,
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
        except Exception as exc:
            logger.debug("selector.get_support() failed, feature names unresolvable: %s", exc)
            support = None
    if support is None:
        return None
    try:
        support = np.asarray(support)
        names_in = list(names_in)
        if support.dtype == bool or (support.size and isinstance(support.flat[0], (bool, np.bool_))):
            return [c for c, keep in zip(names_in, support) if keep]
        return [names_in[int(i)] for i in support]
    except Exception as exc:
        logger.debug("selector support -> feature-name mapping failed: %s", exc)
        return None


_TRANSFORMED_BY_ATTR = "_mlframe_pre_pipeline_id"
"""``DataFrame.attrs`` key naming the fitted pre_pipeline (by ``id``) whose transform produced the frame."""


def _mark_transformed_by(frame, pre_pipeline) -> None:
    """Record on a pandas frame which fitted pipeline transformed it, so a cache hit can tell it is not raw input.

    Polars frames carry no ``attrs`` and are left unmarked (the width discriminator still applies to them). The id is
    stable for as long as the pipeline lives, and the pipeline cache holds the pipeline for the suite's lifetime.
    """
    attrs = getattr(frame, "attrs", None)
    if isinstance(attrs, dict):
        attrs[_TRANSFORMED_BY_ATTR] = id(pre_pipeline)


def _transformed_by(frame):
    """The ``id`` of the pipeline that transformed ``frame``, or ``None`` when it carries no mark."""
    attrs = getattr(frame, "attrs", None)
    return attrs.get(_TRANSFORMED_BY_ATTR) if isinstance(attrs, dict) else None


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
        # A selector's OUTPUT width is its selected base columns PLUS any engineered-recipe columns it
        # appends (MRMR fits both a support_ subset AND feature-engineering recipes; the transformed
        # frame carries both). Counting only support_ under-reports the true output width, so an
        # already-transformed frame whose engineered columns pad it back up to (or past) the base
        # count is misjudged as "still raw" and gets double-transformed through the fitted pipeline --
        # which was fit on the raw INPUT width, not this reduced+engineered width -- raising
        # category_encoders' "Unexpected input dimension" (fuzz c0018/c0111: MRMR selected 2 base + 4
        # engineered = 6 actual output cols, but the support_-only count of 2 made 6 > 2 look raw).
        _n_engineered = len(getattr(_sel, "_engineered_recipes_", None) or [])
        return _effective_width > (len(_out_cols) + _n_engineered)

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
                elif _transformed_by(test_df) == id(pre_pipeline):
                    # The cached frame carries the mark of this very pipeline's transform. The width test below cannot
                    # tell a column-count-preserving pipeline's output from its input (a scaler over 11 columns returns 11
                    # columns), so a cached, already-scaled test frame was scaled a second time on every later target.
                    _do_transform = False
                elif _fitted and _test_df_is_raw_pipeline_input(
                    pre_pipeline,
                    test_df,
                    selector_passthrough_cols,
                    skip_preprocessing,
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
                    _mark_transformed_by(test_df, pre_pipeline)
        columns = list(test_df.columns) if hasattr(test_df, "columns") else []
    else:
        columns = []
        test_df = None

    return test_df, test_target, columns


from ._pipeline_fit_state import (  # noqa: F401 - moved to the leaf module; re-exported here
    _extract_feature_selector,
    _is_fitted,
    _is_stale_fit_state_value_error,
    _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform,
    _raise_pre_pipeline_rowcount_change,
    _reattach_passthrough_to_frame,
    _step_is_fitted,
)

# _pipeline_signature_for_cache /
# _pre_pipeline_cache_key / _get / _set / _clear moved to sibling file
# _pipeline_cache.py and re-exported from this module's top-level imports.


# _apply_pre_pipeline_transforms / _validate_pre_pipeline_output_against_model carved to _pipeline_helpers_apply.py (1k-LOC ceiling).
from ._pipeline_helpers_apply import (  # noqa: F401
    _apply_pre_pipeline_transforms,
    _validate_pre_pipeline_output_against_model,
)
