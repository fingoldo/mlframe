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

from ..phases import phase
from ..utils import maybe_clean_ram_adaptive as _maybe_clean_ram

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


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
    _content_fingerprint_for_cache,
    _fresh_uncachable,
    _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_key,
    _pre_pipeline_cache_set,
    _UncachableSentinel,
)

# Parent helpers used by the moved bodies (defined before the parent's bottom re-export, so this top-level import is cycle-safe).
from ._pipeline_helpers import (  # noqa: E402
    _extract_feature_selector, _is_fitted, _is_stale_fit_state_value_error,
    _multilabel_target_to_1d_for_supervised_encoders, _passthrough_cols_fit_transform,
    _raise_pre_pipeline_rowcount_change,
)

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
        _input_cols = list(train_df.columns) if hasattr(train_df, "columns") else None
        # Row count BEFORE transform. The pre_pipeline slot is row-PRESERVING (selects columns /
        # engineers features, never rows). A resampler here breaks that: this driver returns only
        # (train_df, val_df), so train_target + sample_weight stay at the original row count while
        # train_df grows/shrinks, silently desyncing X from y at model fit (guard at the return).
        _input_n_rows = train_df.shape[0] if hasattr(train_df, "shape") and len(getattr(train_df, "shape", ())) >= 1 else None
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
                            logger.info("Transforming val_df via feature selector...")
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
                        logger.info("Transforming val_df via pre_pipeline...")
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
                _train_is_empty = hasattr(train_df, "shape") and len(train_df.shape) == 2 and train_df.shape[1] == 0
                if val_df is not None and _train_is_empty:
                    if verbose:
                        logger.info(
                            "Skipping val_df transform: train_df has 0 features after fit (selector " "rejected all). Returning empty (N, 0) val_df to match.",
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
                pre_pipeline._mlframe_identity_equivalent = _cols_same and not _has_value_transforms
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

        # Row-count contract guard (see _input_n_rows above). Skip on unknown row count or a
        # 0-feature frame (handled above). The passthrough path raises the same error earlier.
        if _input_n_rows is not None and hasattr(train_df, "shape") and len(train_df.shape) >= 1:
            _out_n_rows = train_df.shape[0]
            _is_zero_feature = len(train_df.shape) == 2 and train_df.shape[1] == 0
            if _out_n_rows != _input_n_rows and not _is_zero_feature:
                _raise_pre_pipeline_rowcount_change(_input_n_rows, _out_n_rows)

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
