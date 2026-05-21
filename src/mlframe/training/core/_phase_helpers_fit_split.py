"""``_phase_fit_pipeline`` + ``_phase_train_val_test_split`` -- the heavy training phases.

Wave 105 (2026-05-21): split out from ``training/core/_phase_helpers.py`` to
keep that file below the 1k-line monolith threshold. Behaviour preserved
bit-for-bit; both functions are re-exported from ``_phase_helpers`` so
existing imports continue to work.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional,
    Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd

from ..phases import phase

try:
    import polars as pl
except ImportError:
    pl = None

# 2026-05-21: wave-105 split-out forgot to mirror the parent's imports +
# NamedTuple defs, so every call into ``_phase_fit_pipeline`` /
# ``_phase_train_val_test_split`` raised NameError. Mirroring the parent
# module's imports here so this file is genuinely self-contained.
if TYPE_CHECKING:
    from ._training_context import TrainingContext

from ._misc_helpers import (
    _auto_detect_feature_types, _cfg_get, _df_shape_str, _drop_cols_df,
    _elapsed_str, _validate_feature_type_exclusivity,
)
from ..configs import PreprocessingExtensionsConfig, TargetTypes
from ..preprocessing import (
    create_split_dataframes, save_split_artifacts,
)
from ..utils import (
    get_process_rss_mb, log_phase, log_ram_usage, maybe_clean_ram_and_gpu,
)
from ..strategies import get_strategy, get_polars_cat_columns
from ..splitting import make_train_test_split
from ..pipeline import (
    apply_preprocessing_extensions, fit_and_transform_pipeline,
)
from ._setup_helpers import _compute_fairness_subgroups


class TrainValTestSplitResult(NamedTuple):
    """Return shape for ``_phase_train_val_test_split`` (mirror of the
    parent module's definition; required here because the split moved
    the function body but left its consumers in both modules).
    """
    train_idx: Any
    val_idx: Any
    test_idx: Any
    train_details: Any
    val_details: Any
    test_details: Any
    train_df: Any
    val_df: Any
    test_df: Any
    fairness_subgroups: Any
    fairness_features: Any
    train_sequences: Any
    val_sequences: Any
    test_sequences: Any
    baseline_rss_mb: Any


class FitPipelineResult(NamedTuple):
    """Return shape for ``_phase_fit_pipeline`` (see comment on
    ``TrainValTestSplitResult``)."""
    train_df: Any
    val_df: Any
    test_df: Any
    pipeline: Any
    extensions_pipeline: Any
    cat_features: Any
    cat_features_polars: Any
    was_polars_input: Any
    all_models_polars_native: Any
    polars_pipeline_applied: Any
    train_df_polars_pre: Any
    val_df_polars_pre: Any
    test_df_polars_pre: Any
    pipeline_config: Any
    preprocessing_extensions: Any
    train_df_pandas_pre_meta: Any

logger = logging.getLogger(__name__)

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

    # Capture the RAW input column list BEFORE the main pipeline transform or
    # extensions stage runs. ``metadata["columns"]`` (set later, post-pipeline +
    # post-extensions) is the *output* schema the trained model was fit on; the
    # predict-time input-validation step needs the *input* schema instead. Without
    # this snapshot, ``_validate_input_columns_against_metadata`` filters predict
    # inputs against post-pipeline column names (truncatedsvd0..9 from a
    # PreprocessingExtensionsConfig dim_reducer, or sklearn one-hot expansions
    # cat_low_A / cat_low_B / ...) and drops every raw user column as "extra" -
    # leaving a (N, 0) frame that crashes the extensions transform with
    # ``Found array with 0 sample(s)`` before any model can run. Surfaced by
    # fuzz iter#189 (binary classification x lgb,linear,ridge x cat_enc=onehot
    # x dim_reducer=TruncatedSVD x 1M rows).
    if train_df is not None and hasattr(train_df, "columns"):
        if isinstance(train_df, pl.DataFrame):
            _raw_cols = list(train_df.columns)
        else:
            _raw_cols = train_df.columns.tolist()
        # SKEW-COL-ORDER: write both the explicit "raw_input_columns" key (post-fix canonical) and
        # the legacy "input_columns" alias. ``_validate_input_columns_against_metadata`` prefers the
        # explicit name; older serialised metadata still reads via the alias.
        metadata["raw_input_columns"] = list(_raw_cols)
        metadata["input_columns"] = list(_raw_cols)

    _strategies_for_polars_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []
    all_models_polars_native = bool(_strategies_for_polars_check) and all(
        s.supports_polars for s in _strategies_for_polars_check
    )

    # CatBoost-specific footgun warning: when CB is in the model suite AND categorical_encoding="ordinal"
    # AND the input frame carries categorical columns (polars Categorical/Enum, or pandas category dtype),
    # the ordinal encoder converts those columns to integer codes BEFORE CatBoost sees them. CatBoost then
    # loses its native categorical handling (combinations, target-statistics, one-hot small-cardinality
    # fast-path) and treats the int codes as ordered numerics, which silently degrades accuracy on
    # high-cardinality cats. Detection here is preventive (warning only; no code-path change). Fire the
    # WARN BEFORE the polars-fastpath auto-flip below otherwise the auto-flip silences the check on the
    # most common polars input path. Cat columns are detected directly from the train_df schema because
    # FeatureTypesConfig doesn't carry a cat_features list (the public surface is text_features +
    # embedding_features; cat_features are auto-detected downstream).
    _suite_models_lower = {str(m).lower() for m in (mlframe_models or [])}
    _has_cb = bool(_suite_models_lower & {"cb", "catboost"})
    _ordinal = (
        getattr(pipeline_config, "categorical_encoding", None) == "ordinal"
        and not getattr(pipeline_config, "skip_categorical_encoding", False)
    )
    _declared_cats: list[str] = []
    if train_df is not None:
        if isinstance(train_df, pl.DataFrame):
            # Use isinstance(d, pl.Enum) instead of str(d).startswith("Enum")
            # so dtype detection is API-stable across polars versions and survives any repr change.
            _enum_cls = getattr(pl, "Enum", None)
            _declared_cats = [
                n for n, d in train_df.schema.items()
                if d == pl.Categorical
                or (_enum_cls is not None and isinstance(d, _enum_cls))
                or d == pl.Utf8 or d == pl.String
            ]
        elif hasattr(train_df, "select_dtypes"):
            try:
                _declared_cats = train_df.select_dtypes(include=["category", "object", "string"]).columns.tolist()
            except Exception:
                _declared_cats = []
    if _has_cb and _ordinal and _declared_cats:
        # Previously a WARN-only check. Surfaced by the diverse-harness
        # fuzz profile (iter#36 with cat_low + text_col + cb): the ordinal
        # encoder turned text_col into ints, which CatBoost then refused
        # with "Invalid type for text_feature ... must have string type".
        # Auto-flip skip_categorical_encoding=True so CB sees the original
        # categorical/text columns and uses its native handling. Caller can
        # still force the old behaviour via
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

    # Auto-skip categorical encoding when all models handle categoricals natively. Runs AFTER the CB+ordinal
    # WARN above so the warning fires on the user's *requested* config rather than the auto-flipped one.
    if was_polars_input and not pipeline_config.skip_categorical_encoding:
        if all_models_polars_native:
            pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
            if verbose:
                logger.info("  All models %s support Polars natively -- skipping categorical encoding in pipeline", mlframe_models)

    # Datetime columns must be decomposed BEFORE the pre-pipeline clone, otherwise the
    # cloned frames retain raw datetimes and reach downstream where numpy/sklearn/CB raise.
    def _detect_datetime_cols(df_):
        if df_ is None:
            return []
        if isinstance(df_, pl.DataFrame):
            return [name for name, dt in df_.schema.items()
                    if isinstance(dt, (pl.Datetime, pl.Date))]
        if hasattr(df_, "dtypes"):
            return [c for c in df_.columns
                    if pd.api.types.is_datetime64_any_dtype(df_[c])]
        return []

    _dt_cols = _detect_datetime_cols(train_df)
    # Skip datetime sources already decomposed by the FTE (it keeps the original via ``delete_original_cols=False``, so the suite would otherwise re-decompose and emit duplicate / overwriting cols on top of FTE's). The raw source is still removed below so downstream consumers (LGB / XGB / sklearn) don't see a leftover datetime64 column they can't dtype-promote.
    _fte_emitted_map = metadata.get("ftextractor_emitted_columns") or {}
    _fte_owned_dt_sources = [c for c in _dt_cols if c in _fte_emitted_map] if _fte_emitted_map else []
    if _fte_emitted_map:
        _dt_cols = [c for c in _dt_cols if c not in _fte_emitted_map]
    if _fte_owned_dt_sources:
        # FTE already produced derived cols for these; the raw datetime source must still be dropped so downstream model libs don't choke on a Datetime64 column. Match the ``delete_original_cols=True`` behaviour of the suite's own create_date_features call below.
        def _drop_source_cols(_frame, _cols):
            if _frame is None:
                return _frame
            _present = [c for c in _cols if c in _frame.columns]
            if not _present:
                return _frame
            if isinstance(_frame, pl.DataFrame):
                return _frame.drop(_present)
            return _frame.drop(columns=_present)
        train_df = _drop_source_cols(train_df, _fte_owned_dt_sources)
        val_df = _drop_source_cols(val_df, _fte_owned_dt_sources)
        test_df = _drop_source_cols(test_df, _fte_owned_dt_sources)
    if _dt_cols:
        from mlframe.feature_engineering.basic import create_date_features
        # Configurable set of dt accessors (year / ordinal_day / minute / ...).
        # Backward-compat default {day, weekday, month, hour} kept by
        # FeatureTypesConfig; callers opt into richer decomposition by passing
        # datetime_methods in their FeatureTypesConfig.
        _configured_methods = (
            set(feature_types_config.datetime_methods)
            if feature_types_config is not None and getattr(feature_types_config, "datetime_methods", None)
            else {"day", "weekday", "month", "hour"}
        )
        # ``create_date_features`` expects {accessor: np_dtype}. int8 fits most
        # cyclical fields; year exceeds int8 range so it needs int32. Pick
        # dtype per-method so the user doesn't have to know polars dtype rules.
        _wide_int_methods = {"year"}
        _dt_methods = {
            m: (np.int32 if m in _wide_int_methods else np.int8)
            for m in sorted(_configured_methods)
        }
        if verbose:
            logger.info(
                "Decomposing %d datetime column(s) into numeric features "
                "(%s) before pre-pipeline clone: %s",
                len(_dt_cols), "/".join(sorted(_dt_methods.keys())), _dt_cols,
            )
        train_df = create_date_features(
            train_df, cols=_dt_cols, delete_original_cols=True,
            methods=_dt_methods,
        )
        if val_df is not None:
            v_cols = [c for c in _dt_cols if c in val_df.columns]
            if v_cols:
                val_df = create_date_features(
                    val_df, cols=v_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )
        if test_df is not None:
            t_cols = [c for c in _dt_cols if c in test_df.columns]
            if t_cols:
                test_df = create_date_features(
                    test_df, cols=t_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )
        # Persist the resolved methods keyed by source column so predict can replay the same expansion deterministically. Stored as the accessor-name -> numpy-dtype-name map (json-friendly); the predict side resolves the name back to the numpy dtype.
        _persisted_methods = {m: _dt_methods[m].__name__ for m in _dt_methods}
        _store = metadata.setdefault("datetime_methods", {})
        for _src in _dt_cols:
            _store[_src] = dict(_persisted_methods)

    # Pre-pipeline polars-pre frames are unconditionally ALIASED to the input frames -- never cloned.
    # Audit-time concern (CONV-HIGH-1) was that polars-ds Blueprint.ordinal_encode / one_hot_encode
    # might mutate the source frame in place. Verified non-issue: ``bp.ordinal_encode(...)`` returns a
    # new Blueprint; ``bp.materialize()`` produces a pipeline; ``pipeline.transform(df)`` returns a
    # new DataFrame (see pipeline.py:1037). Polars frames are conceptually immutable through the
    # public API; Arrow buffers are Arc-counted (clone is a refcount bump, not a deep copy). The
    # global string cache (memory note: "polars 1.x global string cache") grows monotonically -- codes
    # for existing strings never shift, so aliasing the pre-encoding frame is safe even when the
    # encoder later sees additional strings. Downstream rebindings of ``train_df_polars_pre`` (via
    # ``_drop_cols_df`` at L645 / ``_precast_strings`` at L674) all return NEW frames and reassign,
    # so the aliased input is never mutated either.
    if was_polars_input:
        train_df_polars_pre = train_df
        val_df_polars_pre = val_df if isinstance(val_df, pl.DataFrame) else None
        test_df_polars_pre = test_df if isinstance(test_df, pl.DataFrame) else None
        cat_features_polars = get_polars_cat_columns(train_df)
    else:
        train_df_polars_pre = None
        val_df_polars_pre = None
        test_df_polars_pre = None
        cat_features_polars = []

    # Snapshot a pandas-input train_df BEFORE fit_pipeline applies ordinal /
    # one-hot encoding so the downstream auto-detect phase can see the raw
    # string / object dtypes. Without this snapshot the ordinal encoder runs
    # first, converts all string columns to integer codes, and the subsequent
    # auto-detect step (run on the post-pipeline frame) silently classifies
    # everything as numeric -- text columns never get promoted to text_features.
    # Polars input already has ``train_df_polars_pre`` for this purpose.
    # Gated on ``FeatureTypesConfig.feature_types_first`` so byte-for-byte
    # legacy reproductions can disable. fix audit row FE-P1-2.
    _feature_types_first = bool(
        getattr(feature_types_config, "feature_types_first", True)
        if feature_types_config is not None else True
    )
    # Mutation-immune snapshot for the downstream auto-detect phase. A frame-level snapshot (even ``.copy(deep=False)``) shares
    # the source block-manager and leaks any in-place numpy poke (``df[col].values[i] = x``) into the snapshot, silently
    # corrupting auto-detect's view of pre-encoding dtypes / cardinality. The dict captures every datum auto-detect needs
    # (column names, dtype strings, cardinality + non-null counts on text-candidate cols, embedding-shape sniff result on
    # object cols) at snapshot time so the recorded values are baked and immune to any later mutation on ``train_df``.
    # 100 GB frame discipline: only string/object/category columns get nunique / count scans; numeric blocks are never touched.
    train_df_pandas_pre_meta: dict | None = None
    if _feature_types_first and (not was_polars_input) and isinstance(train_df, pd.DataFrame):
        try:
            _text_cand_cols = [
                c for c in train_df.columns
                if train_df[c].dtype.kind in "OUSb" or isinstance(train_df[c].dtype, pd.CategoricalDtype)
            ]
            _n_unique: dict[str, int] = {}
            _non_null: dict[str, int] = {}
            for _c in _text_cand_cols:
                _s = train_df[_c]
                _n_unique[_c] = int(_s.nunique(dropna=True))
                _non_null[_c] = int(_s.notna().sum())
            # Embedding-shape sniff: object-dtype cells holding ndarray / list (e.g. sentence-transformer vectors) cannot
            # be auto-detected from dtype alone. Probe the first 8 non-null cells per object column at snapshot time so the
            # downstream consumer can route them to embedding_features without touching the (potentially-mutated) source.
            _embedding_object_cols: list[str] = []
            for _c in train_df.columns:
                if str(train_df[_c].dtype).startswith("object"):
                    try:
                        _first = next((v for v in train_df[_c].head(8) if v is not None), None)
                    except Exception:
                        _first = None
                    if _first is not None and (
                        hasattr(_first, "shape")
                        or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
                    ):
                        _embedding_object_cols.append(_c)
            # Wave 54 (2026-05-20): pandas allows duplicate column names; the prior
            # {c: ...} comprehension silently collapsed dupes to one entry, so the
            # downstream schema-hash would mis-flag a "matching" schema and drop
            # auto-detect coverage for the duplicate columns. Refuse explicitly.
            _cols_list = list(train_df.columns)
            if len(set(_cols_list)) != len(_cols_list):
                from collections import Counter as _Counter
                _dupes = [_c for _c, _n in _Counter(_cols_list).items() if _n > 1]
                raise ValueError(
                    f"train_df has {len(_dupes)} duplicate column name(s) "
                    f"({_dupes[:5]}); deduplicate before fit() to keep schema-hash honest."
                )
            train_df_pandas_pre_meta = {
                "columns": _cols_list,
                "dtypes": {c: str(train_df[c].dtype) for c in _cols_list},
                "n_unique": _n_unique,
                "non_null": _non_null,
                "embedding_object_cols": _embedding_object_cols,
                "shape": tuple(train_df.shape),
            }
        except Exception:
            train_df_pandas_pre_meta = None
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

    if preprocessing_extensions is not None and isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    # PySR symbolic regression (inside apply_preprocessing_extensions) needs a
    # 1-D y_train. Multi-target pipelines pass a target_by_type dict; pick the
    # first regression target as the supervised signal for symbolic feature
    # discovery. Classification-only setups: PySR is regression-only, falls
    # back to None and the function logs a warning.
    _y_train_for_ext = None
    if (
        preprocessing_extensions is not None
        and getattr(preprocessing_extensions, "pysr_enabled", False)
        and target_by_type is not None
    ):
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
                # Audit D P2-5 (2026-05-18): polars/pandas в†’ numpy is NEEDED here; PySR's
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
                            "PySR symbolic FE: multi-output regression target detected (n=%d columns); "
                            "using first column '%s' as the supervised signal.",
                            _y_train_for_ext.shape[1], _first_name,
                        )
                    _y_train_for_ext = _y_train_for_ext[:, 0]
                # target_by_type carries the PRE-split full target; slice to train_idx
                # so PySR's symbolic FE only sees train-set y. Without this we hit a
                # length mismatch (e.g. full ~5M rows vs train ~4M) and PySR is silently
                # skipped.
                if (
                    _y_train_for_ext is not None
                    and train_idx is not None
                    and hasattr(train_df, "shape")
                    and len(_y_train_for_ext) != train_df.shape[0]
                ):
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
                            len(_y_train_for_ext), train_df.shape[0],
                        )
                    _y_train_for_ext = None
        except Exception as _exc:
            if verbose:
                _diag = "n/a"
                try:
                    _diag = f"keys={list(target_by_type.keys()) if hasattr(target_by_type, 'keys') else type(target_by_type).__name__}"
                except Exception:
                    pass
                logger.warning(
                    "Could not extract y_train for PySR FE: %s: %s (target_by_type %s)",
                    type(_exc).__name__, _exc, _diag,
                )
            _y_train_for_ext = None
    t0_ext = timer()
    # Snapshot the train_df_polars_pre column set so we can detect which new
    # columns the extensions produced and back-merge them into the polars-pre
    # frames. fix audit row FE-P1-3.
    _pre_polars_columns_snapshot = (
        list(train_df_polars_pre.columns) if isinstance(train_df_polars_pre, pl.DataFrame) else None
    )
    # Capture PySR's equation -> column-name map so predict can replay symbolic features against the same content-hashed column names that training emitted.
    _pysr_equations_out: dict = {}
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df, val_df, test_df, preprocessing_extensions, verbose=verbose, y_train=_y_train_for_ext,
        out_pysr_equations=_pysr_equations_out,
    )
    if _pysr_equations_out:
        metadata["pysr_equations"] = dict(_pysr_equations_out)
    if verbose and preprocessing_extensions is not None:
        logger.info("  apply_preprocessing_extensions done in %s", _elapsed_str(t0_ext))
    if extensions_pipeline is not None:
        cat_features = []
        # Polars-fastpath consumers (CB / XGB polars-native path) only see the
        # polars-pre frames; copy the extension-produced new columns onto them
        # so models downstream see consistent feature sets. We use a pandas
        # bridge for the new columns only (existing polars-pre columns are kept
        # as-is to preserve native dtypes / categorical metadata).
        try:
            if (
                isinstance(train_df, pd.DataFrame)
                and _pre_polars_columns_snapshot is not None
                and was_polars_input
            ):
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
                                logger.warning(
                                    "polars-pre %s frame row mismatch for extension columns "
                                    "(pd=%d, pl=%d); skipping back-merge for this split.",
                                    _label, _pd_df.shape[0], _pl_df.shape[0],
                                )
                            continue
                        # Only the new columns we want to merge.
                        _new_df_pd = _pd_df[[c for c in _new_cols if c in _pd_df.columns]]
                        if _new_df_pd.shape[1] == 0:
                            continue
                        try:
                            _new_pl = pl.from_pandas(_new_df_pd)
                            _merged = _pl_df.hstack(_new_pl)
                            if _label == "train":
                                train_df_polars_pre = _merged
                            elif _label == "val":
                                val_df_polars_pre = _merged
                            else:
                                test_df_polars_pre = _merged
                        except Exception as _exc:
                            if verbose:
                                logger.warning(
                                    "Failed to back-merge extension columns into polars-pre %s frame: %s",
                                    _label, _exc,
                                )
        except Exception as _exc:
            if verbose:
                logger.warning(
                    "Polars-pre extension back-merge skipped (%s); polars-fastpath models will not see extension columns.",
                    _exc,
                )

    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    _post_cols = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else list(train_df.columns)
    # SKEW-COL-ORDER: write the explicit "post_pipeline_columns" name AND the legacy "columns" alias.
    # Predict-side validators prefer the explicit name; old metadata files still resolve via "columns".
    metadata["post_pipeline_columns"] = list(_post_cols)
    metadata["columns"] = _post_cols

    if verbose:
        logger.info("  Pipeline done -- train: %s, cat_features: %s", _df_shape_str(train_df), cat_features or '(none)')
        if was_polars_input and cat_features_polars and list(cat_features_polars) != list(cat_features or []):
            logger.info("  Pre-pipeline Polars cat_features: %s", cat_features_polars)
        logger.info("  PHASE 3 total: %s", _elapsed_str(t0_phase3))

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


def _phase_train_val_test_split(
    *,
    df: pl.DataFrame | pd.DataFrame | None,
    target_by_type: dict,
    timestamps: np.ndarray | None,
    group_ids: np.ndarray | pd.Series | None,
    group_ids_raw: np.ndarray | pd.Series | None,
    artifacts: Any,
    sequences: list[np.ndarray] | None,
    split_config: Any,
    behavior_config: Any,
    metadata: dict,
    data_dir: str,
    models_dir: str,
    target_name: str,
    model_name: str,
    df_size_mb: float,
    verbose: bool,
) -> "TrainValTestSplitResult":
    """Train/val/test splitting with auto-stratification + group-aware splitting.

    Mutates ``metadata`` in-place with split sizes + per-split details.
    """
    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    # Auto-stratify by target when no timestamps are present (without stratification,
    # rare-imbalance shuffles produce all-class-0 val slices). Three regimes:
    #   (a) single classification target  -> stratify on its ndarray directly
    #   (b) multiple classification targets (e.g. several binary heads) -> stratify on
    #       a composite key built from the row-tuple, encoded as an int class id.
    #       Gated on combined-cardinality <= MAX_COMPOSITE_CARDINALITY so the
    #       sklearn StratifiedShuffleSplit doesn't reject for sparse classes.
    #   (c) multilabel target (N, K)      -> if iterative-stratification is installed,
    #       pass its ndarray through; otherwise fall back to first-label stratification
    #       as a best-effort over the all-classes-fully-balanced corner case.
    _MAX_COMPOSITE_CARDINALITY = 200  # caps the (b) regime at ~200 distinct row-tuples
    _stratify_y = None
    if timestamps is None and isinstance(target_by_type, dict):
        _classification_targets = []
        _multilabel_target = None
        for _tt, _named in target_by_type.items():
            _tt_name = getattr(_tt, "name", str(_tt)).upper()
            if "MULTILABEL" in _tt_name:
                # Multilabel arrives as (N, K) ndarray under one key; capture and stop.
                if isinstance(_named, dict):
                    _ml_vals = next(iter(_named.values()), None)
                else:
                    _ml_vals = _named
                _multilabel_target = _ml_vals
                continue
            if "CLASS" in _tt_name and isinstance(_named, dict):
                for _tn, _tv in _named.items():
                    if _tv is not None:
                        _classification_targets.append(_tv)
        if _multilabel_target is not None:
            try:
                _ml_arr = np.asarray(_multilabel_target)
                if _ml_arr.ndim == 2 and _ml_arr.shape[1] >= 1:
                    # Prefer the proper iterative-stratification path when available.
                    try:
                        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # noqa: F401
                        _stratify_y = _ml_arr
                    except ImportError:
                        # Best-effort fallback: stratify on the first label column. Better
                        # than nothing when one of the K labels is the rare class.
                        _first = _ml_arr[:, 0]
                        _u, _c = np.unique(_first, return_counts=True)
                        if len(_u) >= 2 and _c.min() >= 2:
                            _stratify_y = _first
                        else:
                            logger.warning(
                                "Auto-stratify: multilabel first-label fallback has %d "
                                "unique values with min-count=%d (need >=2 + min-count>=2); "
                                "stratification disabled. val/test slices may be class-degenerate.",
                                len(_u), int(_c.min()) if len(_c) else 0,
                            )
            except Exception as _strat_err:
                logger.warning(
                    "Auto-stratify: multilabel build failed (%s: %s); shuffled-only splits.",
                    type(_strat_err).__name__, _strat_err,
                )
                _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
                    else:
                        # Surface so the rare-imbalance scenario (single-class slice OR
                        # rare-class with one sample) isn't misdiagnosed as random-seed
                        # flakiness when val ends up all-class-0. Pre-fix this branch
                        # silently flipped to shuffled-only.
                        logger.warning(
                            "Auto-stratify: single classification target has %d unique "
                            "classes with min-count=%d (need >=2 + min-count>=2); "
                            "stratification disabled. val/test slices may be class-degenerate.",
                            len(_u), int(_c.min()) if len(_c) else 0,
                        )
            except Exception as _strat_err:
                logger.warning(
                    "Auto-stratify: single-target build failed (%s: %s); shuffled-only splits.",
                    type(_strat_err).__name__, _strat_err,
                )
                _stratify_y = None
        elif len(_classification_targets) > 1:
            try:
                _arrs = [np.asarray(_t) for _t in _classification_targets]
                _n = len(_arrs[0])
                if all(_a.ndim == 1 and len(_a) == _n for _a in _arrs):
                    # Composite key: each row maps to an integer class id from
                    # (val_t0, val_t1, ..., val_tK) tuple. np.unique on stacked (N, K)
                    # returns_inverse for the encoding in one pass.
                    _stack = np.stack(_arrs, axis=1)
                    _, _composite_ids = np.unique(_stack, axis=0, return_inverse=True)
                    _u, _c = np.unique(_composite_ids, return_counts=True)
                    if 2 <= len(_u) <= _MAX_COMPOSITE_CARDINALITY and _c.min() >= 2:
                        _stratify_y = _composite_ids
                    elif len(_u) < 2 or _c.min() < 2:
                        logger.warning(
                            "Auto-stratify: composite key has %d distinct row-tuples with "
                            "min-count=%d (need >=2 + min-count>=2); stratification disabled. "
                            "val/test slices may be class-degenerate.",
                            len(_u), int(_c.min()) if len(_c) else 0,
                        )
                    elif len(_u) > _MAX_COMPOSITE_CARDINALITY:
                        # Surface the silent fallback to shuffled-only splits so operators
                        # know auto-stratification was abandoned on multi-head targets that
                        # exceed the composite-cardinality cap; otherwise they re-discover
                        # the all-class-0-val-slice bug under heavy class imbalance.
                        logger.warning(
                            "Auto-stratify: composite key has %d distinct row-tuples > "
                            "_MAX_COMPOSITE_CARDINALITY=%d; falling back to UNstratified "
                            "shuffle splits. Rare-class imbalance may produce all-one-class "
                            "val/test slices. Reduce the number of classification heads or "
                            "pre-compute a stratify_y manually to restore stratification.",
                            len(_u), _MAX_COMPOSITE_CARDINALITY,
                        )
            except Exception:
                _stratify_y = None
    # Group-aware splitting opt-in: when the extractor produced ``group_ids`` and
    # ``split_config.use_groups`` is set, route through GroupShuffleSplit.
    _groups = group_ids if (split_config.use_groups and group_ids is not None and len(group_ids) > 0) else None
    with phase("split_data"):
        # ``calib_size`` is a TrainingSplitConfig field for opt-in
        # post-calibration; ``make_train_test_split`` does not accept it
        # (the calibration slice is carved by downstream code, not the
        # splitter). Exclude alongside ``use_groups`` (which is
        # consumed above to derive ``_groups``).
        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df=df,
            timestamps=timestamps,
            stratify_y=_stratify_y,
            groups=_groups,
            **split_config.model_dump(exclude={"use_groups", "calib_size"}),
        )
    if verbose:
        log_ram_usage()

    if data_dir:
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=group_ids_raw,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name=target_name,
            model_name=model_name,
        )

    metadata.update(
        {
            "train_details": train_details,
            "val_details": val_details,
            "test_details": test_details,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
    )

    # Compute fairness subgroups from full df BEFORE splitting.
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info("Computed %d fairness subgroups", len(fairness_subgroups))

    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    if verbose:
        logger.info("  Split shapes -- train: %s, val: %s, test: %s", _df_shape_str(train_df), _df_shape_str(val_df), _df_shape_str(test_df))
        logger.info("  PHASE 2 total: %s", _elapsed_str(t0_phase2))

    # Split sequences by train/val/test indices (for recurrent models).
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))

    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    # Refresh baseline so the next maybe_clean_ram_and_gpu in the caller sees the post-del state.
    baseline_rss_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")
    if verbose:
        log_ram_usage()

    return TrainValTestSplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        fairness_subgroups=fairness_subgroups,
        fairness_features=fairness_features,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        baseline_rss_mb=baseline_rss_mb,
    )


def _phase_auto_detect_feature_types(
    *,
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    train_df_polars_pre: pl.DataFrame | None,
    val_df_polars_pre: pl.DataFrame | None,
    test_df_polars_pre: pl.DataFrame | None,
    cat_features: list[str],
    cat_features_polars: list[str],
    was_polars_input: bool,
    all_models_polars_native: bool,
    pipeline_config: Any,
    feature_types_config: Any,
    metadata: dict,
    verbose: bool,
    train_df_pandas_pre_meta: dict | None = None,
) -> tuple:
    """Auto-detect text + embedding features, optionally drop high-card columns, validate exclusivity, one-time Polars string->Categorical cast.

    Mutates ``metadata`` in-place with ``columns`` and ``cat_features``.
    """
    # Use pre-pipeline view so auto-detection sees original dtypes BEFORE the ordinal encoder converts strings to int codes.
    # Polars: ``train_df_polars_pre`` (frame alias, always populated; polars frames are conceptually immutable through the
    # public API so the alias is safe). Pandas: ``train_df_pandas_pre_meta`` dict, mutation-immune by construction
    # (column names / dtype-strings / cardinality / non-null counts baked at snapshot time). Fallback to post-pipeline
    # ``train_df`` only when both pre-views are absent (legacy callers / feature_types_first=False).
    if was_polars_input:
        detect_df = train_df_polars_pre
    else:
        detect_df = train_df
    raw_cat_features = list(set((cat_features or []) + (cat_features_polars or [])))
    # Honor only strictly-user-declared pl.Categorical columns as already-assigned.
    if was_polars_input:
        user_polars_cats = [
            c for c, dt in zip(detect_df.columns, detect_df.dtypes)
            if dt == pl.Categorical
        ]
    else:
        user_polars_cats = []
    text_features, embedding_features, auto_high_card_drop = _auto_detect_feature_types(
        detect_df, feature_types_config, user_polars_cats, verbose=verbose,
        pandas_meta=train_df_pandas_pre_meta if not was_polars_input else None,
    )

    # Capture pre-drop column data so dummy_baselines per_group_mean can use these as group
    # keys downstream (tree models drop them to avoid XGB QuantileDMatrix OOM).
    # Audit D P1-6 (2026-05-18): pre-fix loop ran ``_frame[c].to_numpy()`` per column per
    # split -- N independent Arrow batches per split. Now we do ONE ``_frame.select(cols)``
    # per split, materialise that 2D matrix through ``get_pandas_view_of_polars_df`` (split-
    # blocks Arrow bridge, ~32x faster than naive to_pandas on multi-col selects), then
    # peel the per-column numpy arrays from the resulting DataFrame view. Pandas-branch is
    # unchanged because pandas ``_frame[col]`` is already a Series view (no extra copy).
    dropped_high_card_data = {}
    if auto_high_card_drop:
        # Per-split single-pass materialisation for polars frames; pandas branch is per-col
        # (cheap Series view).
        _per_split_views: dict[str, Any] = {}
        for _label, _frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            if _frame is None:
                continue
            _cols = _frame.columns if hasattr(_frame, "columns") else []
            _present = [c for c in auto_high_card_drop if c in _cols]
            if not _present:
                continue
            if isinstance(_frame, pl.DataFrame):
                try:
                    # Single select over ALL needed columns; Arrow split-blocks bridge.
                    from mlframe.training.utils import get_pandas_view_of_polars_df as _get_pd_view
                    _per_split_views[_label] = _get_pd_view(_frame.select(_present))
                except Exception:
                    # Fallback to bare to_pandas on the multi-col select; still 1 batch vs N.
                    try:
                        _per_split_views[_label] = _frame.select(_present).to_pandas()
                    except Exception:
                        _per_split_views[_label] = None
            else:
                _per_split_views[_label] = _frame
        for _col in auto_high_card_drop:
            _col_frames = {}
            for _label in ("train", "val", "test"):
                _view = _per_split_views.get(_label)
                if _view is None:
                    continue
                if _col not in getattr(_view, "columns", []):
                    continue
                try:
                    _col_frames[_label] = np.asarray(_view[_col])
                except Exception:
                    continue
            if _col_frames:
                dropped_high_card_data[_col] = _col_frames
        train_df = _drop_cols_df(train_df, auto_high_card_drop)
        val_df = _drop_cols_df(val_df, auto_high_card_drop)
        test_df = _drop_cols_df(test_df, auto_high_card_drop)
        if was_polars_input:
            if train_df_polars_pre is not None:
                train_df_polars_pre = _drop_cols_df(train_df_polars_pre, auto_high_card_drop)
            if val_df_polars_pre is not None:
                val_df_polars_pre = _drop_cols_df(val_df_polars_pre, auto_high_card_drop)
            if test_df_polars_pre is not None:
                test_df_polars_pre = _drop_cols_df(test_df_polars_pre, auto_high_card_drop)
        raw_cat_features = [c for c in raw_cat_features if c not in auto_high_card_drop]
        metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    text_emb_set = set(text_features) | set(embedding_features)
    effective_cat_features = [c for c in raw_cat_features if c not in text_emb_set]
    _validate_feature_type_exclusivity(text_features, embedding_features, effective_cat_features)
    cat_features = effective_cat_features
    metadata["cat_features"] = cat_features

    # One-time Polars string->Enum cast so XGB's arrow bridge doesn't choke on large_string.
    # Use pl.Enum (per-Series, no global cache impact) keyed off the train-only unique set;
    # val/test cast non-strict so OOV becomes null (matches the alignment semantics elsewhere
    # in the suite). pl.Categorical would widen the process-wide string cache (memory rule:
    # reference_polars_global_string_cache). Fixes audit B-P0-3 / Low-B11.
    if was_polars_input and all_models_polars_native and pipeline_config.skip_categorical_encoding:
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        _keep_as_string = text_emb_set
        _str_cols = [c for c, dt in zip(train_df.columns, train_df.dtypes)
                     if dt in _string_types and c not in _keep_as_string]
        if _str_cols:
            # Wave 72 (2026-05-21): build per-column Enum domain from train+val
            # uniques (NOT train-only). val is the early-stopping detector --
            # if a val-only categorical value gets cast to null silently, ES is
            # biased away from val-rare-cat-sensitive splits. test stays
            # unseen (built from train+val ONLY). Symmetric with dict-alignment
            # in _phase_polars_fixes.py.
            _enum_domains: dict[str, list[str]] = {}
            for _c in _str_cols:
                try:
                    _u_train = train_df.select(pl.col(_c).drop_nulls().unique())[_c].to_list()
                    _u_val: list = []
                    if val_df is not None and _c in set(val_df.columns):
                        try:
                            _u_val = val_df.select(pl.col(_c).drop_nulls().unique())[_c].to_list()
                        except Exception:
                            _u_val = []
                    _enum_domains[_c] = sorted(set(_u_train) | set(_u_val), key=str)
                except Exception:
                    pass

            def _enum_cast(df, strict: bool, split_name: str | None = None):
                if df is None:
                    return df
                _existing = set(df.columns)
                _exprs = []
                _affected_cols = []
                for _c in _str_cols:
                    if _c not in _existing or _c not in _enum_domains:
                        continue
                    _exprs.append(pl.col(_c).cast(pl.Enum(_enum_domains[_c]), strict=strict))
                    _affected_cols.append(_c)
                if not _exprs:
                    return df
                # Wave 72 (2026-05-21): quantify silent OOV-nulling so operators
                # can see how many rows got cast-failed (was invisible before).
                _null_pre = {c: int(df[c].null_count()) for c in _affected_cols}
                out = df.with_columns(_exprs)
                if split_name is not None and not strict:
                    _null_deltas = {
                        c: int(out[c].null_count()) - _null_pre[c]
                        for c in _affected_cols
                    }
                    _nonzero = {c: d for c, d in _null_deltas.items() if d > 0}
                    if _nonzero:
                        logger.info(
                            "[enum-cast] %s split: %d col(s) had OOV nulls cast-failed (cols=%s)",
                            split_name, len(_nonzero), _nonzero,
                        )
                return out

            train_df = _enum_cast(train_df, strict=True)
            # val cast is now strict-but-domain-includes-val by construction;
            # any cast failure here would be a logic bug, so strict=True.
            val_df = _enum_cast(val_df, strict=True)
            test_df = _enum_cast(test_df, strict=False, split_name="test")
            train_df_polars_pre = _enum_cast(train_df_polars_pre, strict=True)
            val_df_polars_pre = _enum_cast(val_df_polars_pre, strict=True)
            test_df_polars_pre = _enum_cast(test_df_polars_pre, strict=False, split_name="test (polars_pre)")
            if verbose:
                logger.info("  Cast Polars string columns -> Enum once (shared across model loop)")

    if verbose and (text_features or embedding_features):
        logger.info("  Feature types -- text: %s, embedding: %s, cat: %s", text_features, embedding_features, cat_features or '(none)')

    return (
        train_df, val_df, test_df,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        text_features, embedding_features, cat_features,
        text_emb_set, dropped_high_card_data,
    )


# Wave 105 (2026-05-21): _phase_fit_pipeline + _phase_train_val_test_split
# moved to sibling _phase_helpers_fit_split.py.
from ._phase_helpers_fit_split import _phase_fit_pipeline, _phase_train_val_test_split  # noqa: F401, E402
