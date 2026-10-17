"""``_phase_fit_pipeline`` carved out of
``mlframe.training.core._phase_helpers_fit_split``.

Re-imported at the parent's module bottom so historical
``from mlframe.training.core._phase_helpers_fit_split import _phase_fit_pipeline``
resolves transparently.
"""

from __future__ import annotations

from typing import Any

import logging
from timeit import default_timer as timer
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

# The split-out forgot to mirror the parent's imports +
# NamedTuple defs, so every call into ``_phase_fit_pipeline`` /
# ``_phase_train_val_test_split`` raised NameError. Mirroring the parent
# module's imports here so this file is genuinely self-contained.
if TYPE_CHECKING:
    pass

from ._misc_helpers import (
    _df_shape_str,
    _elapsed_str,
)
from ..configs import PreprocessingExtensionsConfig
from ..utils import (
    log_phase,
)
from ..strategies import get_strategy, get_polars_cat_columns
from ..pipeline import (
    apply_preprocessing_extensions,
    fit_and_transform_pipeline,
)
from ._phase_fit_pipeline_blocks import (
    _phase_fit_pipeline_dt_cols,
    _phase_fit_pipeline_fte_owned_dt_sources,
    _phase_fit_pipeline_string_object_category_columns,
)
from ._phase_fit_result import FitPipelineResult
from ._phase_fit_pipeline_composite import _composite_fe_supervised_target, _run_composite_fe_stages  # noqa: F401
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.core._phase_helpers_fit_split")


def extensions_with_split_seed(
    preprocessing_extensions: "PreprocessingExtensionsConfig | dict | None", split_seed: "int | None"
) -> "PreprocessingExtensionsConfig":
    """``preprocessing_extensions`` as a config whose ``random_seed`` is the suite's split seed unless the caller pinned one.

    Accepts None / dict / config like ``_phase_fit_pipeline``; None becomes the default config, which is what that
    function builds for None anyway.
    """
    if preprocessing_extensions is None:
        preprocessing_extensions = PreprocessingExtensionsConfig()
    elif isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    if getattr(preprocessing_extensions, "random_seed", 0) is None and split_seed is not None:
        preprocessing_extensions = preprocessing_extensions.model_copy(update={"random_seed": int(split_seed)})
    return preprocessing_extensions


def _detect_native_cat_models(strategies) -> tuple[bool, bool]:
    """Return ``(has_cb, all_native_cat)`` from RESOLVED strategy instances.

    Keyed off strategy identity, not the model's string name, so an estimator passed as an INSTANCE (e.g. ``CatBoostClassifier()``)
    is classified the same as its ``"cb"`` alias. A name-only check stringifies the instance to ``"catboostclassifier()"`` and
    silently mis-classes it as non-CB / non-native, skipping the CB ordinal-encoding auto-flip footgun guard downstream.
    Native-cat strategies (no upstream encoder needed to consume raw cats): CatBoost / LightGBM(tree) / XGBoost / HGB.
    """
    from ..strategies import CatBoostStrategy, TreeModelStrategy, XGBoostStrategy, HGBStrategy

    native = (CatBoostStrategy, TreeModelStrategy, XGBoostStrategy, HGBStrategy)
    has_cb = any(isinstance(s, CatBoostStrategy) for s in strategies)
    all_native = bool(strategies) and all(isinstance(s, native) for s in strategies)
    return has_cb, all_native


def _row_wise_replay_config(preprocessing_extensions, fitted: dict) -> dict | None:
    """What predict needs to recompute the row-wise extension columns (row_summary_* / row_extreme_*), or None when
    neither step is enabled.

    The enabled/params knobs, plus the state fitted on train that ``apply_row_wise_steps`` hands back in ``fitted``:
    the pinned numeric column list and the extremality reference. The columns are NOT stateless - scored without the
    reference, a single-row serving batch is its own median and every extremality score is 0.0, and re-deriving the
    column list from the serving frame admits a column that was all-null on train. Without this stamp at all, predict
    never reproduces these columns and every model that saw them at fit time raises "feature names ... unseen at fit
    time" on its first real call - default-ON, so that hit every deployed model.
    """
    if preprocessing_extensions is None or not (
        getattr(preprocessing_extensions, "row_wise_summary_stats_enabled", False)
        or getattr(preprocessing_extensions, "row_wise_extreme_columns_enabled", False)
    ):
        return None
    k_raw = getattr(preprocessing_extensions, "row_wise_extreme_columns_k", None)
    config = {
        "summary_stats_enabled": bool(getattr(preprocessing_extensions, "row_wise_summary_stats_enabled", False)),
        "summary_stats_list": getattr(preprocessing_extensions, "row_wise_summary_stats_list", None),
        "extreme_columns_enabled": bool(getattr(preprocessing_extensions, "row_wise_extreme_columns_enabled", False)),
        "extreme_columns_k": int(k_raw) if k_raw is not None else 3,
    }
    config.update(fitted)
    return config


def _phase_fit_pipeline(
    *,
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    mlframe_models: list[str],
    pipeline_config: Any,
    preprocessing_config: Any,
    feature_types_config: Any,
    preprocessing_extensions: Any,
    metadata: dict,
    verbose: bool,
    target_by_type: Any = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    # Full-length (pre-split), aligned to the ORIGINAL df -- sliced by train_idx/val_idx/test_idx below for the entity/time-keyed FE steps (state_duration,
    # recency_aggregation, ...) that need a group/time key the suite already computes for CV/splitting but never threaded this far before. ``None`` when the
    # caller has no grouping/time-awareness configured -- those steps simply no-op (see ``_entity_time_composite_fe.py``).
    group_ids: np.ndarray | None = None,
    timestamps: Any = None,
    # Auxiliary relational table (interaction/events log distinct from train/val/test) for steps that need a SEPARATE reference table: latent_interaction_svd,
    # nearest_past_join. None (default) is a genuine no-op for both.
    auxiliary_events_df: Any = None,
) -> "FitPipelineResult":
    """Pipeline fitting and transformation.

    Decomposes datetime columns BEFORE the pre-pipeline clone (otherwise the cloned frames
    retain raw datetimes that crash numpy/sklearn/CB downstream), saves Polars originals for
    the fastpath, runs ``fit_and_transform_pipeline``, then applies any
    ``PreprocessingExtensionsConfig``. Mutates ``metadata`` in-place.
    """
    t0_phase3 = timer()
    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    was_polars_input = isinstance(train_df, pl.DataFrame)

    _phase_fit_pipeline_classification_lgb_linear_ridge(train_df, metadata)

    pipeline_config, all_models_polars_native = _resolve_pipeline_config_for_models(
        train_df=train_df,
        mlframe_models=mlframe_models,
        pipeline_config=pipeline_config,
        was_polars_input=was_polars_input,
        verbose=verbose,
    )

    # Datetime columns must be decomposed BEFORE the pre-pipeline clone, otherwise the cloned frames retain raw datetimes and reach downstream where
    # numpy/sklearn/CB raise.
    train_df, val_df, test_df = _decompose_datetime_columns(train_df, val_df, test_df, metadata, feature_types_config, verbose)

    train_df_polars_pre, val_df_polars_pre, test_df_polars_pre, cat_features_polars = _alias_polars_pre_frames(train_df, val_df, test_df, was_polars_input)

    train_df_pandas_pre_meta = _snapshot_pandas_pre_meta(feature_types_config, was_polars_input, train_df)

    # Normalize preprocessing_extensions BEFORE the categorical-composite check below (needs a real
    # PreprocessingExtensionsConfig instance, not a raw dict/None) -- hoisted from its original
    # post-fit_and_transform_pipeline position so this earlier, pre-encoding step can read it too.
    preprocessing_extensions = _phase_fit_pipeline_post_fit_transform_pipeline(preprocessing_extensions)
    train_df, val_df, test_df, preprocessing_extensions = _run_composite_fe_stages(
        train_df,
        val_df,
        test_df,
        preprocessing_extensions,
        target_by_type=target_by_type,
        metadata=metadata,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        group_ids=group_ids,
        timestamps=timestamps,
        auxiliary_events_df=auxiliary_events_df,
        verbose=verbose,
    )

    t0_fit_pipeline = timer()
    train_df, val_df, test_df, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=pipeline_config,
        ensure_float32=preprocessing_config.ensure_float32_dtypes,
        verbose=verbose,
        text_features=feature_types_config.text_features if feature_types_config else [],
        embedding_features=feature_types_config.embedding_features if feature_types_config else [],
    )
    if verbose:
        logger.info("  fit_and_transform_pipeline done in %s", _elapsed_str(t0_fit_pipeline))

    polars_pipeline_applied = was_polars_input and pipeline_config.prefer_polarsds and pipeline is not None

    # Snapshot the train_df_polars_pre column set so we can detect which new
    # columns the extensions produced and back-merge them into the polars-pre
    # frames. fix audit row FE-P1-3.
    _pre_polars_columns_snapshot = list(train_df_polars_pre.columns) if isinstance(train_df_polars_pre, pl.DataFrame) else None
    train_df, val_df, test_df, extensions_pipeline = _run_preprocessing_extensions(
        train_df,
        val_df,
        test_df,
        preprocessing_extensions,
        target_by_type=target_by_type,
        metadata=metadata,
        train_idx=train_idx,
        verbose=verbose,
    )

    if extensions_pipeline is not None:
        cat_features = []
    test_df_polars_pre, train_df_polars_pre, val_df_polars_pre = _phase_fit_pipeline_polars_native_fastpath_mrmr(
        train_df, _pre_polars_columns_snapshot, was_polars_input, train_df_polars_pre, val_df_polars_pre, test_df_polars_pre, val_df, test_df, verbose
    )

    _stamp_post_pipeline_metadata(metadata, pipeline, extensions_pipeline, cat_features, train_df)

    _phase_fit_pipeline_verbose(verbose, train_df, cat_features, was_polars_input, cat_features_polars, t0_phase3)

    return FitPipelineResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        pipeline=pipeline,
        extensions_pipeline=extensions_pipeline,
        cat_features=cat_features,
        cat_features_polars=cat_features_polars,
        was_polars_input=was_polars_input,
        all_models_polars_native=all_models_polars_native,
        polars_pipeline_applied=polars_pipeline_applied,
        train_df_polars_pre=train_df_polars_pre,
        val_df_polars_pre=val_df_polars_pre,
        test_df_polars_pre=test_df_polars_pre,
        pipeline_config=pipeline_config,
        preprocessing_extensions=preprocessing_extensions,
        train_df_pandas_pre_meta=train_df_pandas_pre_meta,
    )


def _phase_fit_pipeline_classification_lgb_linear_ridge(train_df, metadata):
    """Block of _phase_fit_pipeline starting at ``if train_df is not None and hasattr(train_df, "columns"):``."""
    # Capture the RAW input column list BEFORE the main pipeline transform or extensions stage runs. ``metadata["columns"]`` (set later, post-pipeline +
    # post-extensions) is the *output* schema the trained model was fit on; the predict-time input-validation step needs the *input* schema instead. Without
    # this snapshot, ``_validate_input_columns_against_metadata`` filters predict inputs against post-pipeline column names (truncatedsvd0..9 from a
    # PreprocessingExtensionsConfig dim_reducer, or sklearn one-hot expansions cat_low_A / cat_low_B / ...) and drops every raw user column as "extra" - leaving
    # a (N, 0) frame that crashes the extensions transform with ``Found array with 0 sample(s)`` before any model can run. Surfaced by fuzz iter#189 (binary
    # classification x lgb,linear,ridge x cat_enc=onehot x dim_reducer=TruncatedSVD x 1M rows).
    if train_df is not None and hasattr(train_df, "columns"):
        if isinstance(train_df, pl.DataFrame):
            _raw_cols = list(train_df.columns)
        else:
            _raw_cols = train_df.columns.tolist()
        # SKEW-COL-ORDER: write both the explicit "raw_input_columns" key (post-fix canonical) and the legacy "input_columns" alias.
        # ``_validate_input_columns_against_metadata`` prefers the explicit name; older serialised metadata still reads via the alias.
        metadata["raw_input_columns"] = list(_raw_cols)
        metadata["input_columns"] = list(_raw_cols)


def _phase_fit_pipeline_cb_native_auto_flip(_has_cb, train_df, _declared_cats):
    """Block of _phase_fit_pipeline starting at ``if _has_cb and train_df is not None:``."""
    if _has_cb and train_df is not None:
        if isinstance(train_df, pl.DataFrame):
            # Use isinstance(d, pl.Enum) instead of str(d).startswith("Enum") so dtype detection is API-stable across polars versions and survives any repr
            # change.
            _enum_cls = getattr(pl, "Enum", None)
            _declared_cats = [
                n
                for n, d in train_df.schema.items()
                if d == pl.Categorical or (_enum_cls is not None and isinstance(d, _enum_cls)) or d == pl.Utf8 or d == pl.String
            ]
        elif hasattr(train_df, "select_dtypes"):
            try:
                _declared_cats = train_df.select_dtypes(include=["category", "object", "string"]).columns.tolist()
            except Exception as e:
                logger.debug("select_dtypes for declared categoricals failed: %s", e)
                _declared_cats = []
    return _declared_cats


def _phase_fit_pipeline_requested_config_rather_than(was_polars_input, pipeline_config, all_models_polars_native, verbose, mlframe_models):
    """Block of _phase_fit_pipeline starting at ``if was_polars_input and not pipeline_config.skip_categorical_encoding:``."""
    if was_polars_input and not pipeline_config.skip_categorical_encoding:
        if all_models_polars_native:
            pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
            if verbose:
                logger.info("  All models %s support Polars natively -- skipping categorical encoding in pipeline", mlframe_models)
    return pipeline_config


def _phase_fit_pipeline_post_fit_transform_pipeline(preprocessing_extensions):
    """Block of _phase_fit_pipeline starting at ``if preprocessing_extensions is not None and isinstance(preprocessing_e``."""
    if preprocessing_extensions is not None and isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    elif preprocessing_extensions is None:
        # A caller who never touches ``preprocessing_extensions`` still gets the config's own
        # DEFAULT-ON steps (row_wise_summary_stats_enabled / row_wise_extreme_columns_enabled) --
        # every other field (pysr_enabled, scaler, kbins, categorical_*_concat_*, ...) keeps its own
        # inert None/False default, so this only activates the generically-safe additive row-wise FE
        # steps, not the whole sklearn-bridge feature set. Explicit ``PreprocessingExtensionsConfig(...)``
        # callers are unaffected (they already construct their own instance upstream).
        preprocessing_extensions = PreprocessingExtensionsConfig()
    return preprocessing_extensions


def _phase_fit_pipeline_back_none_function_logs(preprocessing_extensions, target_by_type, verbose, train_idx, train_df, _y_train_for_ext):
    """Block of _phase_fit_pipeline starting at ``if preprocessing_extensions is not None and getattr(preprocessing_exte``."""
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "pysr_enabled", False) and target_by_type is not None:
        try:
            # target_by_type structure varies by extractor:
            #   (a) Dict[TargetTypes, Dict[str, ndarray]]  - nested
            #   (b) Dict[TargetTypes, ndarray]             - flat (single target)
            #   (c) Dict[str, ndarray]                     - "regression" -> arr
            # Iterate items() and match by str-cast to dodge any
            # StrEnum-identity-vs-string-equality quirks.
            _reg_targets = None
            if hasattr(target_by_type, "items"):
                for _k, _v in target_by_type.items():
                    if str(_k).lower().endswith("regression"):
                        _reg_targets = _v
                        break
            if _reg_targets is not None and not isinstance(_reg_targets, dict):
                # Case (b)/(c): _reg_targets is already a 1-D array-like.
                _vals_direct = _reg_targets
                _reg_targets = {"_default": _vals_direct}
            if _reg_targets:
                _first_name = next(iter(_reg_targets))
                _vals = _reg_targets[_first_name]
                # The polars/pandas -> numpy conversion here is NEEDED; PySR's
                # ``PySRRegressor.fit(X, y)`` consumes a numpy array, and the ndim / [:, 0]
                # downstream operations also assume numpy semantics. Conversion cannot be
                # pushed further.
                if hasattr(_vals, "to_numpy"):
                    _y_train_for_ext = _vals.to_numpy()
                else:
                    _y_train_for_ext = np.asarray(_vals)
                if _y_train_for_ext is not None and _y_train_for_ext.ndim > 1:
                    # Multi-output regression target -> first column for PySR.
                    # PySR is single-target by design; we surface the chosen target name so a
                    # multi-head regression doesn't silently pick the wrong head as the symbolic-FE signal.
                    if verbose:
                        logger.info(
                            "PySR symbolic FE: multi-output regression target detected (n=%d columns); " "using first column '%s' as the supervised signal.",
                            _y_train_for_ext.shape[1],
                            _first_name,
                        )
                    _y_train_for_ext = _y_train_for_ext[:, 0]
                # target_by_type carries the PRE-split full target; slice to train_idx
                # so PySR's symbolic FE only sees train-set y. Without this we hit a
                # length mismatch (e.g. full ~5M rows vs train ~4M) and PySR is silently
                # skipped.
                if _y_train_for_ext is not None and train_idx is not None and hasattr(train_df, "shape") and len(_y_train_for_ext) != train_df.shape[0]:
                    try:
                        _idx_arr = np.asarray(train_idx)
                        if len(_idx_arr) == train_df.shape[0] and int(_idx_arr.max()) < len(_y_train_for_ext):
                            _y_train_for_ext = _y_train_for_ext[_idx_arr]
                    except (TypeError, ValueError, IndexError):
                        pass
                if hasattr(train_df, "shape") and _y_train_for_ext is not None and len(_y_train_for_ext) != train_df.shape[0]:
                    if verbose:
                        logger.warning(
                            "PySR y_train length mismatch (target=%d, train rows=%d); skipping symbolic FE.",
                            len(_y_train_for_ext),
                            train_df.shape[0],
                        )
                    _y_train_for_ext = None
        except Exception as _exc:
            if verbose:
                _diag = "n/a"
                try:
                    _diag = f"keys={list(target_by_type.keys()) if hasattr(target_by_type, 'keys') else type(target_by_type).__name__}"
                except Exception as e:
                    logger.debug("swallowed exception in _phase_helpers_fit_pipeline.py: %s", e)
                    pass
                logger.warning(
                    "Could not extract y_train for PySR FE: %s: %s (target_by_type %s)",
                    type(_exc).__name__,
                    _exc,
                    _diag,
                )
            else:
                logger.debug("Could not extract y_train for PySR FE: %s: %s", type(_exc).__name__, _exc)
            _y_train_for_ext = None
    return _y_train_for_ext


def _phase_fit_pipeline_polars_native_fastpath_mrmr(
    train_df, _pre_polars_columns_snapshot, was_polars_input, train_df_polars_pre, val_df_polars_pre, test_df_polars_pre, val_df, test_df, verbose
):
    """Block of _phase_fit_pipeline starting at ``try:``."""
    # Polars-fastpath consumers (CB / XGB polars-native path) only see the
    # polars-pre frames; copy the extension-produced new columns onto them
    # so models downstream see consistent feature sets. We use a pandas
    # bridge for the new columns only (existing polars-pre columns are kept
    # as-is to preserve native dtypes / categorical metadata). Gated on
    # new columns actually existing, NOT on extensions_pipeline: row-wise
    # summary/extreme columns are added to train_df/val_df/test_df even when
    # no sklearn-bridge stage (scaler/pysr/tfidf/kbins/dim_reducer) is
    # configured, in which case extensions_pipeline stays None and this
    # back-merge would otherwise never run, silently dropping those columns
    # from the polars-native fastpath (e.g. MRMR fit on polars-sourced input).
    try:
        if isinstance(train_df, pd.DataFrame) and _pre_polars_columns_snapshot is not None and was_polars_input:
            _new_cols = [c for c in train_df.columns if c not in set(_pre_polars_columns_snapshot)]
            if _new_cols:
                # Explicit (label, pandas-frame, polars-pre-frame) triples so the
                # back-merge no longer fishes the polars-side frame out of locals()
                # by string lookup. Captures the current binding at iteration time;
                # rebinding below updates the same local name afterward.
                _polars_pre_by_label = {
                    "train": train_df_polars_pre,
                    "val": val_df_polars_pre,
                    "test": test_df_polars_pre,
                }
                for _label, _pd_df in (
                    ("train", train_df),
                    ("val", val_df),
                    ("test", test_df),
                ):
                    _pl_df = _polars_pre_by_label.get(_label)
                    if not isinstance(_pl_df, pl.DataFrame) or not isinstance(_pd_df, pd.DataFrame):
                        continue
                    if _pd_df.shape[0] != _pl_df.shape[0]:
                        # Row counts must match; otherwise the join could mis-align silently.
                        if verbose:
                            log_throttle(
                                logger,
                                "polars_pre_extension_row_mismatch",
                                logging.WARNING,
                                "polars-pre %s frame row mismatch for extension columns " "(pd=%d, pl=%d); skipping back-merge for this split.",
                                _label,
                                _pd_df.shape[0],
                                _pl_df.shape[0],
                            )
                        continue
                    # Only the new columns we want to merge.
                    _new_df_pd = _pd_df[[c for c in _new_cols if c in _pd_df.columns]]
                    if _new_df_pd.shape[1] == 0:
                        continue
                    try:
                        # Build polars columns from per-column .to_numpy() views (skips pandas block consolidation). Bench (100k x 30 mixed dtypes): 16.0ms -> 1.05ms (15x); per-split (train/val/test) this is ~3x the saving.
                        _new_pl = pl.DataFrame({c: _new_df_pd[c].to_numpy() for c in _new_df_pd.columns})
                        _merged = _pl_df.hstack(_new_pl)
                        if _label == "train":
                            train_df_polars_pre = _merged
                        elif _label == "val":
                            val_df_polars_pre = _merged
                        else:
                            test_df_polars_pre = _merged
                    except Exception as _exc:
                        if verbose:
                            log_throttle(
                                logger,
                                "polars_pre_extension_backmerge_failed",
                                logging.WARNING,
                                "Failed to back-merge extension columns into polars-pre %s frame: %s",
                                _label,
                                _exc,
                            )
                        else:
                            logger.debug("Failed to back-merge extension columns into polars-pre %s frame: %s", _label, _exc)
    except Exception as _exc:
        if verbose:
            logger.warning(
                "Polars-pre extension back-merge skipped (%s); polars-fastpath models will not see extension columns.",
                _exc,
            )
        else:
            logger.debug("Polars-pre extension back-merge skipped (%s); polars-fastpath models will not see extension columns.", _exc)
    return test_df_polars_pre, train_df_polars_pre, val_df_polars_pre


def _phase_fit_pipeline_verbose(verbose, train_df, cat_features, was_polars_input, cat_features_polars, t0_phase3):
    """Block of _phase_fit_pipeline starting at ``if verbose:``."""
    if verbose:
        logger.info("  Pipeline done -- train: %s, cat_features: %s", _df_shape_str(train_df), "(none)" if cat_features is None or len(cat_features) == 0 else cat_features)
        if was_polars_input and cat_features_polars and list(cat_features_polars) != list(cat_features or []):
            logger.info("  Pre-pipeline Polars cat_features: %s", cat_features_polars)
        logger.info("  PHASE 3 total: %s", _elapsed_str(t0_phase3))


def _detect_datetime_cols(df_: Any) -> list[str]:
    """List column names with a datetime/date dtype in ``df_`` (polars or pandas); empty list for a None frame or one exposing neither a polars schema nor pandas dtypes."""
    if df_ is None:
        return []
    if isinstance(df_, pl.DataFrame):
        return [name for name, dt in df_.schema.items() if isinstance(dt, (pl.Datetime, pl.Date))]
    if hasattr(df_, "dtypes"):
        return [c for c in df_.columns if pd.api.types.is_datetime64_any_dtype(df_[c])]
    return []


def _resolve_pipeline_config_for_models(
    *,
    train_df: Any,
    mlframe_models: list[str],
    pipeline_config: Any,
    was_polars_input: bool,
    verbose: bool,
) -> tuple[Any, bool]:
    """Apply the CB+ordinal auto-flip and the all-native auto-skip of categorical encoding; return ``(pipeline_config, all_models_polars_native)``."""
    _strategies_for_polars_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []
    all_models_polars_native = bool(_strategies_for_polars_check) and all(s.supports_polars for s in _strategies_for_polars_check)

    # CatBoost-specific footgun warning: when CB is in the model suite AND categorical_encoding="ordinal" AND the input frame carries categorical columns
    # (polars Categorical/Enum, or pandas category dtype), the ordinal encoder converts those columns to integer codes BEFORE CatBoost sees them. CatBoost then
    # loses its native categorical handling (combinations, target-statistics, one-hot small-cardinality fast-path) and treats the int codes as ordered numerics,
    # which silently degrades accuracy on high-cardinality cats. Detection here is preventive (warning only; no code-path change). Fire the WARN BEFORE the
    # polars-fastpath auto-flip below otherwise the auto-flip silences the check on the most common polars input path. Cat columns are detected directly from
    # the train_df schema because FeatureTypesConfig doesn't carry a cat_features list (the public surface is text_features + embedding_features; cat_features
    # are auto-detected downstream).
    _has_cb, _all_models_native_cat = _detect_native_cat_models(_strategies_for_polars_check)
    _ordinal = getattr(pipeline_config, "categorical_encoding", None) == "ordinal" and not getattr(pipeline_config, "skip_categorical_encoding", False)
    # Skip the cat-schema scan entirely when CatBoost isn't in the suite -- the only consumer of ``_declared_cats`` below is the CB+ordinal warning block and
    # the CB-native auto-flip; a non-CB suite cannot trip either branch so the pandas ``select_dtypes`` / polars schema iteration is pure waste.
    _declared_cats: list[str] = []
    _declared_cats = _phase_fit_pipeline_cb_native_auto_flip(_has_cb, train_df, _declared_cats)
    # Only auto-flip when EVERY suite model supports native categorical input. If a non-CB / non-native model is also in the suite (e.g. ``ridge``), the ordinal
    # encoder is required for that model to consume the cats and the auto-flip would crash the non-CB legs downstream with raw strings.
    if _has_cb and _ordinal and _declared_cats and _all_models_native_cat:
        # Previously a WARN-only check. Surfaced by the diverse-harness fuzz profile (iter#36 with cat_low + text_col + cb): the ordinal encoder turned text_col
        # into ints, which CatBoost then refused with "Invalid type for text_feature ... must have string type". Auto-flip skip_categorical_encoding=True so CB
        # sees the original categorical/text columns and uses its native handling. Caller can still force the old behaviour via
        # PreprocessingBackendConfig(skip_categorical_encoding=False).
        logger.warning(
            "  CatBoost in mlframe_models + categorical_encoding='ordinal' + %d "
            "categorical column(s) detected. Auto-flipping "
            "skip_categorical_encoding=True so CB keeps native cat-handling "
            "(combinations, target-statistics) and text_features stay string-typed. "
            "Set skip_categorical_encoding=False explicitly to restore the previous "
            "ordinal-then-CB behaviour.",
            len(_declared_cats),
        )
        pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
        _ordinal = False

    # Auto-skip categorical encoding when all models handle categoricals natively. Runs AFTER the CB+ordinal WARN above so the warning fires on the user's
    # *requested* config rather than the auto-flipped one.
    pipeline_config = _phase_fit_pipeline_requested_config_rather_than(was_polars_input, pipeline_config, all_models_polars_native, verbose, mlframe_models)
    return pipeline_config, all_models_polars_native


def _decompose_datetime_columns(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    metadata: dict,
    feature_types_config: Any,
    verbose: bool,
) -> tuple[Any, Any, Any]:
    """Decompose datetime columns of the splits (those not already owned by the FTE) and return ``(train_df, val_df, test_df)``."""
    # Datetime columns must be decomposed BEFORE the pre-pipeline clone, otherwise the cloned frames retain raw datetimes and reach downstream where
    # numpy/sklearn/CB raise.
    _dt_cols = _detect_datetime_cols(train_df)
    # Skip datetime sources already decomposed by the FTE (it keeps the original via ``delete_original_cols=False``, so the suite would otherwise re-decompose and emit duplicate / overwriting cols on top of FTE's). The raw source is still removed below so downstream consumers (LGB / XGB / sklearn) don't see a leftover datetime64 column they can't dtype-promote.
    _fte_emitted_map = metadata.get("ftextractor_emitted_columns") or {}
    _fte_owned_dt_sources = [c for c in _dt_cols if c in _fte_emitted_map] if _fte_emitted_map else []
    if _fte_emitted_map:
        _dt_cols = [c for c in _dt_cols if c not in _fte_emitted_map]
    test_df, train_df, val_df = _phase_fit_pipeline_fte_owned_dt_sources(_fte_owned_dt_sources, train_df, val_df, test_df)
    test_df, train_df, val_df = _phase_fit_pipeline_dt_cols(_dt_cols, feature_types_config, verbose, train_df, val_df, test_df, metadata)
    return train_df, val_df, test_df


def _alias_polars_pre_frames(train_df: Any, val_df: Any, test_df: Any, was_polars_input: bool) -> tuple[Any, Any, Any, Any]:
    """Return ``(train_pre, val_pre, test_pre, cat_features_polars)``: the polars pre-pipeline frames (aliases, never clones) and polars cat columns."""
    # Pre-pipeline polars-pre frames are unconditionally ALIASED to the input frames -- never cloned. Audit-time concern (CONV-HIGH-1) was that polars-ds
    # Blueprint.ordinal_encode / one_hot_encode might mutate the source frame in place. Verified non-issue: ``bp.ordinal_encode(...)`` returns a new Blueprint;
    # ``bp.materialize()`` produces a pipeline; ``pipeline.transform(df)`` returns a new DataFrame (see pipeline.py:1037). Polars frames are conceptually
    # immutable through the public API; Arrow buffers are Arc-counted (clone is a refcount bump, not a deep copy). The global string cache (memory note: "polars
    # 1.x global string cache") grows monotonically -- codes
    # for existing strings never shift, so aliasing the pre-encoding frame is safe even when the
    # encoder later sees additional strings. Downstream rebindings of ``train_df_polars_pre`` (via ``_drop_cols_df`` at L645 / ``_precast_strings`` at L674) all
    # return NEW frames and reassign, so the aliased input is never mutated either.
    if was_polars_input and isinstance(train_df, pl.DataFrame):
        train_df_polars_pre = train_df
        val_df_polars_pre = val_df if isinstance(val_df, pl.DataFrame) else None
        test_df_polars_pre = test_df if isinstance(test_df, pl.DataFrame) else None
        cat_features_polars = get_polars_cat_columns(train_df)
    else:
        train_df_polars_pre = None
        val_df_polars_pre = None
        test_df_polars_pre = None
        cat_features_polars = []
    return train_df_polars_pre, val_df_polars_pre, test_df_polars_pre, cat_features_polars


def _run_preprocessing_extensions(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    preprocessing_extensions: Any,
    *,
    target_by_type: Any,
    metadata: dict,
    train_idx: np.ndarray | None,
    verbose: bool,
) -> tuple[Any, Any, Any, Any]:
    """Extract the PySR y_train, run ``apply_preprocessing_extensions`` and stamp its replay state; return ``(train_df, val_df, test_df, extensions_pipeline)``."""
    # PySR symbolic regression (inside apply_preprocessing_extensions) needs a
    # 1-D y_train. Multi-target pipelines pass a target_by_type dict; pick the
    # first regression target as the supervised signal for symbolic feature
    # discovery. Classification-only setups: PySR is regression-only, falls
    # back to None and the function logs a warning.
    _y_train_for_ext = None
    _y_train_for_ext = _phase_fit_pipeline_back_none_function_logs(preprocessing_extensions, target_by_type, verbose, train_idx, train_df, _y_train_for_ext)
    t0_ext = timer()
    # Capture PySR's equation -> column-name map so predict can replay symbolic features against the same content-hashed column names that training emitted.
    _pysr_equations_out: dict = {}
    _row_wise_replay_out: dict = {}
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df,
        val_df,
        test_df,
        preprocessing_extensions,
        verbose=verbose,
        y_train=_y_train_for_ext,
        out_pysr_equations=_pysr_equations_out,
        out_row_wise_replay=_row_wise_replay_out,
    )
    if _pysr_equations_out:
        metadata["pysr_equations"] = dict(_pysr_equations_out)
    _rw_replay_config = _row_wise_replay_config(preprocessing_extensions, _row_wise_replay_out)
    if _rw_replay_config is not None:
        metadata["row_wise_extensions_config"] = _rw_replay_config
    if verbose and preprocessing_extensions is not None:
        logger.info("  apply_preprocessing_extensions done in %s", _elapsed_str(t0_ext))
    return train_df, val_df, test_df, extensions_pipeline


def _snapshot_pandas_pre_meta(feature_types_config: Any, was_polars_input: bool, train_df: Any) -> dict | None:
    """Mutation-immune pre-encoding snapshot of a pandas-input ``train_df`` for the downstream auto-detect phase (None when not applicable)."""
    # Snapshot a pandas-input train_df BEFORE fit_pipeline applies ordinal / one-hot encoding so the downstream auto-detect phase can see the raw string /
    # object dtypes. Without this snapshot the ordinal encoder runs first, converts all string columns to integer codes, and the subsequent auto-detect step
    # (run on the post-pipeline frame) silently classifies everything as numeric -- text columns never get promoted to text_features. Polars input already has
    # ``train_df_polars_pre`` for this purpose. Gated on ``FeatureTypesConfig.feature_types_first`` so byte-for-byte legacy reproductions can disable. fix audit
    # row FE-P1-2.
    _feature_types_first = bool(getattr(feature_types_config, "feature_types_first", True) if feature_types_config is not None else True)
    # Mutation-immune snapshot for the downstream auto-detect phase. A frame-level snapshot (even ``.copy(deep=False)``) shares the source block-manager and
    # leaks any in-place numpy poke (``df[col].values[i] = x``) into the snapshot, silently corrupting auto-detect's view of pre-encoding dtypes / cardinality.
    # The dict captures every datum auto-detect needs (column names, dtype strings, cardinality + non-null counts on text-candidate cols, embedding-shape sniff
    # result on object cols) at snapshot time so the recorded values are baked and immune to any later mutation on ``train_df``. 100 GB frame discipline: only
    # string/object/category columns get nunique / count scans; numeric blocks are never touched.
    train_df_pandas_pre_meta: dict | None = None
    train_df_pandas_pre_meta = _phase_fit_pipeline_string_object_category_columns(_feature_types_first, was_polars_input, train_df, train_df_pandas_pre_meta)
    return train_df_pandas_pre_meta


def _stamp_post_pipeline_metadata(metadata: dict, pipeline: Any, extensions_pipeline: Any, cat_features: Any, train_df: Any) -> None:
    """Record the fitted pipelines, categorical features, provenance and the post-pipeline column schema into ``metadata``."""
    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    try:
        from mlframe.training.provenance import record_provenance as _record_provenance

        _record_provenance(
            metadata,
            "preprocessing_pipeline",
            source="train",
            n_rows=int(train_df.shape[0]) if hasattr(train_df, "shape") else None,
            extra={"n_features_out": int(train_df.shape[1]) if hasattr(train_df, "shape") and len(train_df.shape) > 1 else None},
        )
    except Exception as e:
        logger.debug("swallowed exception in _phase_helpers_fit_pipeline.py: %s", e)
        pass
    _post_cols = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else list(train_df.columns)
    # On a polars run the pandas-tier frame here does not carry the categorical columns (polars-native models read them from
    # the polars frames), so they were missing from the recorded schema although the model is fitted on them; list them.
    _post_cols = _post_cols + [c for c in (cat_features or []) if c not in set(_post_cols)]
    # SKEW-COL-ORDER: write the explicit "post_pipeline_columns" name AND the legacy "columns" alias. ``_post_cols`` is already a freshly built list; reuse
    # the same reference under both keys so an in-place mutation by one downstream consumer is visible under the other (the historical aliasing contract).
    metadata["post_pipeline_columns"] = _post_cols
    metadata["columns"] = _post_cols
