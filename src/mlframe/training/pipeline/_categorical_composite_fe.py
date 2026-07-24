"""Categorical composite FE (powerset concat / auto MI-grouped concat), applied BEFORE
categorical encoding inside ``_phase_fit_pipeline`` -- see ``PreprocessingExtensionsConfig``'s
docstring for why this can't live inside ``apply_preprocessing_extensions`` (its numeric-only
gate would drop a string composite column before any downstream step saw it).

Persists the exact groups/columns used at fit time onto ``metadata`` so ``predict.py`` can replay
byte-identical composites without re-running MI-scoring (``auto_concat_categorical_groups``) or
re-deriving the source column list (schema drift immune, matches the ``datetime_methods`` pattern).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.categorical_group_concat import auto_concat_categorical_groups, concat_categorical_group
from mlframe.feature_engineering.categorical_powerset_concat import categorical_powerset_concat
from mlframe.training.strategies import get_polars_cat_columns

logger = logging.getLogger(__name__)

COMPOSITE_SEPARATOR = "_"


def _detect_cat_columns(df: Any) -> List[str]:
    """Detect categorical/object/string column names of df, working for both polars and pandas."""
    if isinstance(df, pl.DataFrame):
        return get_polars_cat_columns(df)
    if hasattr(df, "select_dtypes"):
        try:
            return [str(c) for c in df.select_dtypes(include=["category", "object", "string"]).columns.tolist()]
        except Exception as exc:
            logger.debug("_detect_cat_columns: select_dtypes probe failed: %s", exc)
            return []
    return []


def _attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    """Append ``new_cols`` (a pandas frame, same row count/order as ``df``) onto ``df``, preserving ``df``'s own frame type."""
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)


def _to_pandas_view(df: Any, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """Zero-copy-ish pandas view of just ``columns`` (falls back to a full conversion for polars)."""
    if df is None:
        return None
    if isinstance(df, pl.DataFrame):
        return df.select(columns).to_pandas()
    return df[list(columns)]


def apply_categorical_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    y_train: Optional[np.ndarray],
    metadata: dict,
    verbose: int = 0,
) -> tuple:
    """Generate powerset / auto-grouped categorical composite columns on train/val/test, opt-in per ``config``.

    No-op (returns inputs unchanged) when neither step is enabled, when ``train_df`` carries no
    categorical columns, or when the detected categorical column count exceeds
    ``config.categorical_composite_max_source_columns`` (WARN, not raise -- a wide-schema caller
    just doesn't get this optional step rather than hitting a 2^n blowup or an O(n^2) MI-scoring pass).
    """
    _powerset_on = bool(getattr(config, "categorical_powerset_concat_enabled", False))
    _auto_group_on = bool(getattr(config, "categorical_group_concat_auto_enabled", False))
    if not (_powerset_on or _auto_group_on) or train_df is None:
        return train_df, val_df, test_df

    cat_cols = _detect_cat_columns(train_df)
    if len(cat_cols) < 2:
        return train_df, val_df, test_df

    _max_source_cols = int(getattr(config, "categorical_composite_max_source_columns", 12) or 12)
    if len(cat_cols) > _max_source_cols:
        logger.warning(
            "apply_categorical_composite_fe: %d categorical column(s) exceeds "
            "categorical_composite_max_source_columns=%d; skipping (avoids a powerset blowup / "
            "O(n^2) MI-scoring pass). Narrow the schema or raise the cap explicitly.",
            len(cat_cols), _max_source_cols,
        )
        return train_df, val_df, test_df

    train_cat_pd = _to_pandas_view(train_df, cat_cols)
    if train_cat_pd is None:
        return train_df, val_df, test_df

    if _auto_group_on and y_train is not None and len(y_train) == train_cat_pd.shape[0]:
        try:
            _min_mi_gain = float(getattr(config, "categorical_group_concat_min_mi_gain", 0.0) or 0.0)
            _max_group_size = getattr(config, "categorical_group_concat_max_group_size", None)
            _, groups = auto_concat_categorical_groups(
                train_cat_pd, columns=cat_cols, y=np.asarray(y_train),
                separator=COMPOSITE_SEPARATOR, min_mi_gain=_min_mi_gain, max_group_size=_max_group_size,
            )
            _multi_col_groups = [g for g in groups if len(g) >= 2]
            metadata["categorical_group_concat_auto_groups"] = _multi_col_groups
            if _multi_col_groups:
                for _split_name, _df in (("train", train_df), ("val", val_df), ("test", test_df)):
                    if _df is None:
                        continue
                    _pd_view = train_cat_pd if _split_name == "train" else _to_pandas_view(_df, cat_cols)
                    if _pd_view is None:
                        continue
                    _new_cols = pd.DataFrame(index=_pd_view.index)
                    for _group in _multi_col_groups:
                        if not all(c in _pd_view.columns for c in _group):
                            continue
                        _fname = f"concat_group__{COMPOSITE_SEPARATOR.join(_group)}"
                        _composed = concat_categorical_group(_pd_view, columns=_group, separator=COMPOSITE_SEPARATOR, feature_name=_fname)
                        _new_cols[_fname] = _composed[_fname].to_numpy()
                    if _split_name == "train":
                        train_df = _attach_new_columns(train_df, _new_cols)
                    elif _split_name == "val":
                        val_df = _attach_new_columns(val_df, _new_cols)
                    else:
                        test_df = _attach_new_columns(test_df, _new_cols)
                if verbose:
                    logger.info("apply_categorical_composite_fe: auto-discovered %d composite group(s): %s", len(_multi_col_groups), _multi_col_groups)
        except Exception:
            logger.warning("apply_categorical_composite_fe: categorical_group_concat_auto step failed; skipping.", exc_info=True)

    if _powerset_on:
        try:
            _max_order = int(getattr(config, "categorical_powerset_concat_max_order", 2) or 2)
            metadata["categorical_powerset_concat_columns"] = list(cat_cols)
            metadata["categorical_powerset_concat_max_order"] = _max_order
            for _split_name, _df in (("train", train_df), ("val", val_df), ("test", test_df)):
                if _df is None:
                    continue
                _pd_view = train_cat_pd if _split_name == "train" else _to_pandas_view(_df, cat_cols)
                if _pd_view is None:
                    continue
                _powerset_out = categorical_powerset_concat(_pd_view, columns=cat_cols, separator=COMPOSITE_SEPARATOR, max_order=_max_order)
                _new_cols = _powerset_out.drop(columns=list(cat_cols))
                if _split_name == "train":
                    train_df = _attach_new_columns(train_df, _new_cols)
                elif _split_name == "val":
                    val_df = _attach_new_columns(val_df, _new_cols)
                else:
                    test_df = _attach_new_columns(test_df, _new_cols)
            if verbose:
                logger.info("apply_categorical_composite_fe: powerset-concat over %d source column(s), max_order=%d", len(cat_cols), _max_order)
        except Exception:
            logger.warning("apply_categorical_composite_fe: categorical_powerset_concat step failed; skipping.", exc_info=True)

    return train_df, val_df, test_df


def replay_categorical_composite_fe(df: Any, metadata: dict, verbose: int = 0) -> Any:
    """Predict-time replay: re-materialize the exact composite columns fit-time persisted, without
    re-running MI-scoring (auto groups) or re-deriving the source column list (powerset)."""
    if df is None:
        return df

    _auto_groups = metadata.get("categorical_group_concat_auto_groups") or []
    _powerset_cols = metadata.get("categorical_powerset_concat_columns") or []
    if not _auto_groups and not _powerset_cols:
        return df

    _all_source_cols = sorted({c for g in _auto_groups for c in g} | set(_powerset_cols))
    _present = [c for c in _all_source_cols if c in (df.columns if hasattr(df, "columns") else [])]
    if not _present:
        return df
    _pd_view = _to_pandas_view(df, _present)
    if _pd_view is None:
        return df

    new_cols = pd.DataFrame(index=_pd_view.index)
    for _group in _auto_groups:
        if not all(c in _pd_view.columns for c in _group):
            continue
        _fname = f"concat_group__{COMPOSITE_SEPARATOR.join(_group)}"
        _composed = concat_categorical_group(_pd_view, columns=_group, separator=COMPOSITE_SEPARATOR, feature_name=_fname)
        new_cols[_fname] = _composed[_fname].to_numpy()

    _powerset_present = [c for c in _powerset_cols if c in _pd_view.columns]
    if len(_powerset_present) >= 2:
        _max_order = int(metadata.get("categorical_powerset_concat_max_order", 2) or 2)
        _powerset_out = categorical_powerset_concat(_pd_view, columns=_powerset_present, separator=COMPOSITE_SEPARATOR, max_order=_max_order)
        for c in _powerset_out.columns:
            if c not in _powerset_present:
                new_cols[c] = _powerset_out[c].to_numpy()

    if verbose:
        logger.info("replay_categorical_composite_fe: replayed %d composite column(s)", new_cols.shape[1])
    return _attach_new_columns(df, new_cols)


__all__ = ["apply_categorical_composite_fe", "replay_categorical_composite_fe"]
