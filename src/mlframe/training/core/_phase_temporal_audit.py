"""Temporal target audit batch precompute. Single polars multi-aggregation pass over all (target_type, target_name) pairs (~5x faster than per-call for N>1 targets on >1M rows)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify

from ..target_temporal_audit import (
    audit_targets_over_time as _audit_targets_over_time,
    format_temporal_audit_report as _format_temporal_audit_report,
    plot_target_over_time as _plot_target_over_time,
)

logger = logging.getLogger(__name__)


def run_temporal_audit_batch(
    *,
    behavior_config,
    features_and_targets_extractor,
    df,  # may be None if del'd earlier
    timestamps,  # FTE-returned ndarray, main fallback when df column dropped
    target_by_type: dict,
    verbose: bool = True,
) -> dict[Any, dict[str, Any]]:
    """Precompute temporal target audit for all targets.

    Timestamp resolution order:
    1. behavior_config.target_temporal_audit_column = "<col>" -> explicit opt-in
    2. behavior_config.target_temporal_audit_column = ""      -> explicit opt-out
    3. behavior_config.target_temporal_audit_column = None    -> fall through to FTE.ts_field
    4. FTE.ts_field set + timestamps present                  -> auto-detect (audit fires)
    5. neither                                                -> audit silent

    Returns dict keyed by (target_type, target_name) -> audit result.
    """
    _all_target_audits: dict[Any, dict[str, Any]] = {}

    _audit_ts_override = getattr(behavior_config, "target_temporal_audit_column", None) if behavior_config else None
    if _audit_ts_override is None:
        # df may have been deleted earlier; FTE-returned ``timestamps`` ndarray is the primary fallback.
        _fte_ts = getattr(features_and_targets_extractor, "ts_field", None)
        _ts_in_df = (
            df is not None and hasattr(df, "columns") and _fte_ts in df.columns
        )
        if _fte_ts and (timestamps is not None or _ts_in_df):
            _audit_ts_col = _fte_ts
            logger.info(
                "target_temporal_audit: auto-detected timestamp column '%s' "
                "from features_and_targets_extractor.ts_field. To override, "
                "set behavior_config.target_temporal_audit_column='<col>'; "
                "to disable, set it to ''.",
                _audit_ts_col,
            )
        else:
            _audit_ts_col = None
    else:
        _audit_ts_col = _audit_ts_override

    if not _audit_ts_col:
        return _all_target_audits

    try:
        # Prefer df column, fall back to the FTE-returned ``timestamps`` ndarray (the prod norm where ts_field is in columns_to_drop).
        _audit_ts_values = None
        _audit_src_kind = None
        if df is not None and hasattr(df, "columns") and _audit_ts_col in df.columns:
            _audit_ts_values = df[_audit_ts_col]
            _audit_src_kind = "df_column"
        elif timestamps is not None:
            _audit_ts_values = timestamps
            _audit_src_kind = "fte_timestamps"
            logger.info(
                "target_temporal_audit: column '%s' was dropped from df "
                "(likely via columns_to_drop) — using FTE-returned "
                "timestamps ndarray as fallback.",
                _audit_ts_col,
            )
        if _audit_ts_values is None:
            logger.warning(
                "target_temporal_audit: column '%s' not found in df and "
                "FTE returned no timestamps — audit skipped. Either "
                "set behavior_config.target_temporal_audit_column to a "
                "column present in df, or configure ts_field on your "
                "FeaturesAndTargetsExtractor.",
                _audit_ts_col,
            )
            return _all_target_audits

        # audit_key is unique across target_types so same target_name under two target_types doesn't collide.
        _audit_input_cols: dict[str, np.ndarray] = {}
        _audit_targets_spec: dict[str, tuple[str, str]] = {}
        _audit_keys_by_pair: dict[tuple[Any, str], str] = {}
        for _tt_outer, _named in target_by_type.items():
            for _tname, _tvals in _named.items():
                _audit_key = f"{slugify(str(_tt_outer).lower())}__{slugify(_tname)}"
                _audit_col = f"__audit_target_{_audit_key}"
                _arr = np.asarray(_tvals)
                if _arr.ndim != 1:
                    continue  # multilabel unsupported; needs per-label decomposition
                _audit_input_cols[_audit_col] = _arr
                _audit_targets_spec[_audit_key] = (
                    _audit_col,
                    "regression" if str(_tt_outer) == "regression" else "binary_classification",
                )
                _audit_keys_by_pair[(_tt_outer, _tname)] = _audit_key

        if _audit_targets_spec:
            if _audit_src_kind == "df_column" and isinstance(df, pl.DataFrame):
                _batch_input = df.select([_audit_ts_col]).with_columns([
                    pl.Series(name, arr) for name, arr in _audit_input_cols.items()
                ])
            elif _audit_src_kind == "df_column":
                _batch_input = pd.DataFrame({
                    _audit_ts_col: df[_audit_ts_col].values,
                    **_audit_input_cols,
                })
            else:
                _ts_arr = np.asarray(_audit_ts_values)
                _batch_input = pd.DataFrame({
                    _audit_ts_col: _ts_arr,
                    **_audit_input_cols,
                })
            _gran = getattr(behavior_config, "target_temporal_audit_granularity", "auto")
            _batch_results = _audit_targets_over_time(
                _batch_input,
                timestamp_col=_audit_ts_col,
                targets=_audit_targets_spec,
                granularity=_gran,
            )
            for (_tt_pair, _tname_pair), _key in _audit_keys_by_pair.items():
                _all_target_audits.setdefault(_tt_pair, {})[_tname_pair] = _batch_results.get(_key)
            logger.info(
                "target_temporal_audit: batched %d target(s) in one polars multi-aggregation pass.",
                len(_audit_targets_spec),
            )
    except Exception as _audit_err:
        logger.warning(
            "target_temporal_audit batch failed (timestamp_col='%s'): %s. Training continues without audit.",
            _audit_ts_col, _audit_err,
        )

    return _all_target_audits
