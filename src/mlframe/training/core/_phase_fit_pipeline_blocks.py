"""Datetime / pandas-schema-snapshot blocks carved verbatim out of ``_phase_fit_pipeline``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

logger = logging.getLogger("mlframe.training.core._phase_helpers_fit_split")


def _phase_fit_pipeline_fte_owned_dt_sources(_fte_owned_dt_sources, train_df, val_df, test_df):
    """Block of _phase_fit_pipeline starting at ``if _fte_owned_dt_sources:``."""
    if _fte_owned_dt_sources:
        # FTE already produced derived cols for these; the raw datetime source must still be dropped so downstream model libs don't choke on a Datetime64 column. Match the ``delete_original_cols=True`` behaviour of the suite's own create_date_features call below.
        def _drop_source_cols(_frame, _cols):
            """Drop the given columns from ``_frame`` (whichever of them are actually present), format-agnostic across polars and pandas; returns ``_frame`` unchanged when it is None or none of ``_cols`` are present."""
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
    return test_df, train_df, val_df


def _phase_fit_pipeline_dt_cols(_dt_cols, feature_types_config, verbose, train_df, val_df, test_df, metadata):
    """Block of _phase_fit_pipeline starting at ``if _dt_cols:``."""
    if _dt_cols:
        from mlframe.feature_engineering.basic import create_date_features

        # Configurable set of dt accessors (year / ordinal_day / minute / ...). Backward-compat default {day, weekday, month, hour} kept by FeatureTypesConfig;
        # callers opt into richer decomposition by passing datetime_methods in their FeatureTypesConfig.
        _configured_methods = (
            set(feature_types_config.datetime_methods)
            if feature_types_config is not None and getattr(feature_types_config, "datetime_methods", None)
            else {"day", "weekday", "month", "hour"}
        )
        # ``create_date_features`` expects {accessor: np_dtype}. Per-method width comes from the canonical ``_DEFAULT_DATE_METHODS`` map so wide fields are
        # never silently truncated: year needs int32, and day_of_year (1..366) needs int16 -- a flat int8 wraps day_of_year (pandas: silent mod-256; polars:
        # strict-cast crash mid-pipeline). Unmapped methods default to int16 (covers every date field bar year).
        from mlframe.feature_engineering.shared import DEFAULT_DATE_METHODS as _DEFAULT_DATE_METHODS

        _dt_methods = {m: _DEFAULT_DATE_METHODS.get(m, np.int16) for m in sorted(_configured_methods)}
        if verbose:
            logger.info(
                "Decomposing %d datetime column(s) into numeric features " "(%s) before pre-pipeline clone: %s",
                len(_dt_cols),
                "/".join(sorted(_dt_methods.keys())),
                _dt_cols,
            )
        train_df = create_date_features(
            train_df,
            cols=_dt_cols,
            delete_original_cols=True,
            methods=_dt_methods,
        )
        if val_df is not None:
            v_cols = [c for c in _dt_cols if c in val_df.columns]
            if v_cols:
                val_df = create_date_features(
                    val_df,
                    cols=v_cols,
                    delete_original_cols=True,
                    methods=_dt_methods,
                )
        if test_df is not None:
            t_cols = [c for c in _dt_cols if c in test_df.columns]
            if t_cols:
                test_df = create_date_features(
                    test_df,
                    cols=t_cols,
                    delete_original_cols=True,
                    methods=_dt_methods,
                )
        # Persist the resolved methods keyed by source column so predict can replay the same expansion deterministically. Stored as the accessor-name -> numpy-dtype-name map (json-friendly); the predict side resolves the name back to the numpy dtype.
        _persisted_methods = {m: _dt_methods[m].__name__ for m in _dt_methods}
        _store = metadata.setdefault("datetime_methods", {})
        for _src in _dt_cols:
            _store[_src] = dict(_persisted_methods)
        # The cyclical sin/cos encoding is recomputed at predict under the same names; record which version fitted it.
        from mlframe.feature_engineering.basic import CYCLICAL_ENCODING_VERSION

        metadata["datetime_cyclical_version"] = CYCLICAL_ENCODING_VERSION
    return test_df, train_df, val_df


def _phase_fit_pipeline_string_object_category_columns(_feature_types_first, was_polars_input, train_df, train_df_pandas_pre_meta):
    """Block of _phase_fit_pipeline starting at ``if _feature_types_first and (not was_polars_input) and isinstance(trai``."""
    if _feature_types_first and (not was_polars_input) and isinstance(train_df, pd.DataFrame):
        # pandas allows duplicate column names; the prior {c: ...} comprehension silently collapsed dupes to one entry, so the downstream schema-hash would
        # mis-flag a "matching" schema and drop auto-detect coverage for the duplicate columns. Refuse explicitly, outside the best-effort snapshot below.
        _cols_list = list(train_df.columns)
        if len(set(_cols_list)) != len(_cols_list):
            from collections import Counter as _Counter

            _dupes = [_c for _c, _n in _Counter(_cols_list).items() if _n > 1]
            raise ValueError(f"train_df has {len(_dupes)} duplicate column name(s) " f"({_dupes[:5]}); deduplicate before fit() to keep schema-hash honest.")
        try:
            _text_cand_cols = [c for c in train_df.columns if train_df[c].dtype.kind in "OUSb" or isinstance(train_df[c].dtype, pd.CategoricalDtype)]
            _n_unique: dict[str, int] = {}
            _non_null: dict[str, int] = {}
            for _c in _text_cand_cols:
                _s = train_df[_c]
                _n_unique[_c] = int(_s.nunique(dropna=True))
                _non_null[_c] = int(_s.notna().sum())
            # Embedding-shape sniff: object-dtype cells holding ndarray / list (e.g. sentence-transformer vectors) cannot be auto-detected from dtype alone.
            # Probe the first 8 non-null cells per object column at snapshot time so the downstream consumer can route them to embedding_features without
            # touching the (potentially-mutated) source.
            _embedding_object_cols: list[str] = []
            for _c in train_df.columns:
                if str(train_df[_c].dtype).startswith("object"):
                    try:
                        _first = next((v for v in train_df[_c].head(8) if v is not None), None)
                    except Exception as e:
                        logger.debug("peeking column %r's first non-null value failed: %s", _c, e)
                        _first = None
                    if _first is not None and (hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))):
                        _embedding_object_cols.append(_c)
            # A pandas 'category' dtype's categories can hold ANY value type (bool/int/float), unlike polars
            # Categorical/Enum which are always string-backed. Flag columns whose categories aren't strings so
            # the consumer (_auto_detect_feature_types) never text-auto-promotes them -- CatBoost rejects a
            # non-string category value with "text_features must have string type" (fuzz-caught).
            _non_string_category_cols = [
                c for c in _cols_list if isinstance(train_df[c].dtype, pd.CategoricalDtype) and train_df[c].dtype.categories.dtype.kind not in "OU"
            ]
            train_df_pandas_pre_meta = {
                "columns": _cols_list,
                "dtypes": {c: str(train_df[c].dtype) for c in _cols_list},
                "n_unique": _n_unique,
                "non_null": _non_null,
                "embedding_object_cols": _embedding_object_cols,
                "non_string_category_cols": _non_string_category_cols,
                "shape": tuple(train_df.shape),
            }
        except Exception as e:
            logger.debug("train_df_pandas_pre_meta computation failed: %s", e)
            train_df_pandas_pre_meta = None
    return train_df_pandas_pre_meta
