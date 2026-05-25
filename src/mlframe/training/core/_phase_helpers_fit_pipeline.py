"""``_phase_fit_pipeline`` carved out of
``mlframe.training.core._phase_helpers_fit_split``.

Re-imported at the parent's module bottom so historical
``from mlframe.training.core._phase_helpers_fit_split import _phase_fit_pipeline``
resolves transparently.
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
from ._phase_helpers_fit_split import FitPipelineResult  # noqa: E402

logger = logging.getLogger("mlframe.training.core._phase_helpers_fit_split")




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
    # Skip the cat-schema scan entirely when CatBoost isn't in the suite -- the only consumer of ``_declared_cats`` below is the CB+ordinal warning
    # block and the CB-native auto-flip; a non-CB suite cannot trip either branch so the pandas ``select_dtypes`` / polars schema iteration is pure waste.
    _declared_cats: list[str] = []
    if _has_cb and train_df is not None:
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
    # Only auto-flip when EVERY suite model supports native categorical input.
    # If a non-CB / non-native model is also in the suite (e.g. ``ridge``),
    # the ordinal encoder is required for that model to consume the cats and
    # the auto-flip would crash the non-CB legs downstream with raw strings.
    _NATIVE_CAT_MODELS = {"cb", "catboost", "lgb", "lightgbm", "xgb", "xgboost", "hgb", "histgradientboosting"}
    _all_models_native_cat = bool(_suite_models_lower) and _suite_models_lower.issubset(_NATIVE_CAT_MODELS)
    if _has_cb and _ordinal and _declared_cats and _all_models_native_cat:
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
                            # Build polars columns from per-column .to_numpy() views (skips pandas block consolidation). Bench (100k x 30 mixed dtypes, 2026-05-24): 16.0ms -> 1.05ms (15x); per-split (train/val/test) this is ~3x the saving.
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
    try:
        from mlframe.training.provenance import record_provenance as _record_provenance
        _record_provenance(
            metadata,
            "preprocessing_pipeline",
            source="train",
            n_rows=int(train_df.shape[0]) if hasattr(train_df, "shape") else None,
            extra={"n_features_out": int(train_df.shape[1]) if hasattr(train_df, "shape") and len(train_df.shape) > 1 else None},
        )
    except Exception:
        pass
    _post_cols = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else list(train_df.columns)
    # SKEW-COL-ORDER: write the explicit "post_pipeline_columns" name AND the legacy "columns" alias. ``_post_cols`` is already a freshly built list; reuse
    # the same reference under both keys so an in-place mutation by one downstream consumer is visible under the other (the historical aliasing contract).
    metadata["post_pipeline_columns"] = _post_cols
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
