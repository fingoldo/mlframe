"""Stages of ``audit_targets_over_time``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.training.targets.target_temporal_audit import _HAS_POLARS
from mlframe.training.targets.target_temporal_audit import pl
# --- end imports ---


def _aggregate_targets_by_period(df, timestamp_col, target_specs, chosen, aggregate_polars_multi, aggregate_pandas):
    """Stage of ``audit_targets_over_time``."""
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        agg = aggregate_polars_multi(
            df,
            timestamp_col,
            target_specs,
            chosen,
        )
    else:
        # Pandas fallback: still N passes, but at least one place to
        # change later if we add a multi-agg pandas path.
        agg = aggregate_pandas(
            df,
            timestamp_col,
            target_specs[0][0],
            chosen,
            target_type=target_specs[0][1],
        )
        agg = agg.rename(columns={"target_rate": target_specs[0][2]})
        for _col, _ttype, _alias in target_specs[1:]:
            _extra_agg = aggregate_pandas(
                df,
                timestamp_col,
                _col,
                chosen,
                target_type=_ttype,
            )
            agg[_alias] = _extra_agg["target_rate"].values
    return agg


def _collect_target_specs(targets):
    """Stage of ``audit_targets_over_time``."""
    target_specs: list[tuple[str, str, str]] = []  # (col, type, alias)
    name_to_alias: dict[str, str] = {}
    for name, spec in targets.items():
        if isinstance(spec, str):
            _col, _ttype = spec, "binary_classification"
        elif isinstance(spec, tuple) and len(spec) == 2:
            _col, _ttype = spec[0], spec[1]
        else:
            raise ValueError(f"targets[{name!r}] must be str or (col, target_type) tuple; " f"got {type(spec).__name__}")
        _alias = f"__rate__{name}"
        target_specs.append((_col, _ttype, _alias))
        name_to_alias[name] = _alias
    return name_to_alias, target_specs
