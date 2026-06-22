"""Regression: a schema-drifted composite spec lacking 'target_col' must not KeyError.

Pre-fix ``_matching_spec["target_col"]`` was a hard index; on older/drifted cached specs it raised
KeyError, swallowed by the surrounding broad except, so the y-scale dummy baseline went silently
missing. ``_resolve_spec_raw_target`` uses ``.get`` + explicit skip-with-log.
"""
from __future__ import annotations

from mlframe.training.core._phase_dummy_baselines import _resolve_spec_raw_target


def test_missing_target_col_returns_none_pair_no_keyerror():
    spec = {"name": "comp", "transform_name": "linear_residual"}  # no 'target_col'
    target_by_type = {"regression": {"y": [1.0, 2.0]}}
    # Pre-fix: spec["target_col"] -> KeyError. Post-fix: clean (None, None).
    raw_col, raw_y = _resolve_spec_raw_target(spec, "regression", target_by_type, "comp")
    assert raw_col is None
    assert raw_y is None


def test_present_target_col_resolves_y():
    spec = {"name": "comp", "target_col": "y"}
    target_by_type = {"regression": {"y": [1.0, 2.0, 3.0]}}
    raw_col, raw_y = _resolve_spec_raw_target(spec, "regression", target_by_type, "comp")
    assert raw_col == "y"
    assert raw_y == [1.0, 2.0, 3.0]
