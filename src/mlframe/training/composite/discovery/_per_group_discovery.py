"""Predict-time row routing for opt-in per-group composite discovery.

``CompositeTargetDiscovery.fit`` (see ``_fit.py``'s trailing block, gated on
``config.per_group_discovery_enabled``) delegates to ``_per_group.run_per_group_discovery``
to populate ``self.specs_by_group_: dict[group_value, list[CompositeSpec]]`` -- one
independently-discovered spec list per group with >= ``config.per_group_min_rows`` rows,
alongside the always-present global ``self.specs_``. Groups absent from ``specs_by_group_``
(too few rows, or unseen at predict time) have no entry, and must fall back to ``specs_``.

This module supplies the row-routing counterpart to that discovery-time split, mirroring
``GroupedBlockStacker``'s "one submodel per group, stitched back by a row mask" pattern
(``grouped_block_stacking.py``) but applied to composite-spec forward-transform application
instead of a trained submodel: for a given spec NAME, resolve each row's spec instance from
its OWN group's spec list, falling back to the global spec of that name when the row's group
has no per-group discovery result.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery

from ._grouped_causal_bases import _extract_raw

logger = logging.getLogger(__name__)


def route_spec_column_by_group(
    discovery: "CompositeTargetDiscovery",
    df: Any,
    spec_name: str,
) -> np.ndarray:
    """Compute a single per-row T-scale column for ``spec_name``, resolving EACH row's spec of that
    name from its OWN group's spec list (``discovery.specs_by_group_``), falling back to the global
    ``discovery.specs_`` for rows whose group is absent from ``specs_by_group_`` (below
    ``per_group_min_rows`` at fit time, or an unseen group at predict time).

    Requires ``discovery.config.per_group_discovery_enabled`` and ``discovery.config.per_group_column``
    to have been set for the fit that produced ``discovery``. Rows whose resolved spec set contains no
    spec named ``spec_name`` (e.g. that group's own discovery converged on a different spec) get ``NaN``,
    same convention as ``CompositeTargetDiscovery.iter_transform``.
    """
    from .screening import _extract_column_array
    from ..transforms import get_transform

    group_column = getattr(discovery.config, "per_group_column", None)
    if not group_column:
        raise ValueError("route_spec_column_by_group requires config.per_group_column to be set")

    specs_by_group: Mapping[Any, list] = getattr(discovery, "specs_by_group_", {}) or {}
    global_specs = list(getattr(discovery, "specs_", []) or [])
    global_by_name = {s.name: s for s in global_specs}

    target_col = discovery._target_col
    y_full = _extract_column_array(df, target_col)
    n = y_full.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    raw_groups = _extract_raw(df, group_column)
    for group_value in np.unique(raw_groups):
        row_mask = raw_groups == group_value
        group_specs = specs_by_group.get(group_value)
        spec = next((s for s in group_specs if s.name == spec_name), None) if group_specs else None
        if spec is None:
            spec = global_by_name.get(spec_name)
        if spec is None:
            continue  # no spec of this name for this row's group AND none globally; leave NaN.

        transform = get_transform(spec.transform_name)
        extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
        if not transform.requires_base:
            base_rows = None
        elif extra:
            base_rows = np.column_stack([_extract_column_array(df, c)[row_mask] for c in (spec.base_column, *extra)])
        else:
            base_rows = _extract_column_array(df, spec.base_column)[row_mask]
        y_rows = y_full[row_mask]
        valid = transform.domain_check(y_rows, base_rows)  # type: ignore[arg-type]
        t_rows = np.full(y_rows.shape[0], np.nan, dtype=np.float64)
        if valid.any():
            _base_valid = None if base_rows is None else base_rows[valid]
            t_rows[valid] = transform.forward(y_rows[valid], _base_valid, spec.fitted_params)
        out[row_mask] = t_rows
    return out
