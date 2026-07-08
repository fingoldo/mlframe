"""Temporal target audit batch precompute. Single polars multi-aggregation pass over all (target_type, target_name) pairs (~5x faster than per-call for N>1 targets on >1M rows)."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pyutilz.strings import slugify

from ..targets import (
    audit_targets_over_time as _audit_targets_over_time,
    coerce_timestamps_for_audit as _coerce_timestamps_for_audit_lib,
    format_temporal_audit_report as _format_temporal_audit_report,
    plot_target_over_time as _plot_target_over_time,
)

logger = logging.getLogger(__name__)


# Re-export the lib-level helper under the legacy name so the test suite (and any
# external callers of the now-private symbol introduced in commit 9a1855e) keeps
# working unchanged. The canonical home is now ``target_temporal_audit`` (one
# level up) so the public ``RegimeAuditResult.segment_mask`` / ``_pick_granularity``
# can use the same helper without an upward import.
_coerce_timestamps_for_audit = _coerce_timestamps_for_audit_lib


def run_temporal_audit_batch(
    *,
    behavior_config,
    features_and_targets_extractor,
    timestamps,  # FTE-returned ndarray; df was already del'd by the caller, so this is the only source
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
        # df was already del'd before this batch is called; ``timestamps`` is the only source.
        _fte_ts = getattr(features_and_targets_extractor, "ts_field", None)
        if _fte_ts and timestamps is not None:
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
        if timestamps is None:
            logger.warning(
                "target_temporal_audit: timestamp column '%s' requested but "
                "FTE returned no timestamps -- audit skipped. Configure "
                "ts_field on your FeaturesAndTargetsExtractor or unset "
                "behavior_config.target_temporal_audit_column.",
                _audit_ts_col,
            )
            return _all_target_audits
        _audit_ts_values = timestamps

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
            _ts_arr = np.asarray(_audit_ts_values)
            # Route to the polars multi-target single-pass aggregation path
            # (~5x faster on N>1 targets at >1M rows per target_temporal_audit
            # module docstring). audit_targets_over_time only takes that path
            # when ``isinstance(df, pl.DataFrame)``. Numeric timestamps are
            # pre-coerced via pd.to_datetime so the polars Datetime semantics
            # match the pandas fallback (pandas interprets bare int64 as ns
            # since the unix epoch).
            try:
                import polars as _pl
                # Coerce numeric timestamps to a usable datetime array, with auto-detected unit.
                # pandas pd.to_datetime() interprets bare int64 as NANOSECONDS since the unix epoch
                # by default; for the very common case of epoch-seconds input this collapses every
                # row into the first nanosecond after epoch, producing a one-bin audit with no
                # change-point coverage. Caller can override via
                # behavior_config.target_temporal_audit_unit ('s' / 'ms' / 'us' / 'ns');
                # otherwise we try each unit and pick the COARSEST one that yields a fully-in-range
                # [1970, 2200] result (coarsest = widest span = highest audit resolution).
                _override_unit = getattr(behavior_config, "target_temporal_audit_unit", None) if behavior_config is not None else None
                _ts_for_pl = _coerce_timestamps_for_audit(_ts_arr, explicit_unit=_override_unit)
                _batch_input = _pl.DataFrame(
                    {
                        _audit_ts_col: _ts_for_pl,
                        **_audit_input_cols,
                    }
                )
            except ImportError:
                # Same coercion as the polars path: building the audit frame from raw int64
                # timestamps lets pandas interpret them as nanoseconds-since-epoch, collapsing
                # epoch-seconds input into a single bin (no change-point coverage).
                _override_unit = getattr(behavior_config, "target_temporal_audit_unit", None) if behavior_config is not None else None
                _ts_for_pd = _coerce_timestamps_for_audit(_ts_arr, explicit_unit=_override_unit)
                _batch_input = pd.DataFrame(
                    {
                        _audit_ts_col: _ts_for_pd,
                        **_audit_input_cols,
                    }
                )
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
