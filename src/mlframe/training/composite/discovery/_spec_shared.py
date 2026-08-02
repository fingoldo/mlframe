"""Shared spec-inspection helpers for the discovery/ honest-gate family (_honest_holdout,
_honest_oof_select, _honest_rmse_gate, _yscale_holdout_gate): small utilities independently
duplicated across those modules, consolidated here so a fix can't silently drift out of sync
across copies.
"""
from __future__ import annotations

from typing import Any


def spec_base_columns(spec: Any) -> list:
    """Full ordered base-column list for a spec: primary ``base_column`` plus any ``extra_base_columns`` (multi-base); empty for a unary (base-free) spec."""
    extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
    if not spec.base_column:
        return []
    return [spec.base_column, *extra]
