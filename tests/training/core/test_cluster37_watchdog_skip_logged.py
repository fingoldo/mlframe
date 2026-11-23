"""Cluster 37 regression: a watchdog check skipped for a missing base column is logged at WARNING (audit 2026-11-20, finding 37-F25)."""

import logging
from types import SimpleNamespace

import pandas as pd

from mlframe.training.composite.transforms import get_transform
from mlframe.training.core._composite_wrap_watchdog import run_wrap_watchdog


def _base_requiring_transform_name():
    """Name of a registered transform that consumes a base column."""
    for name in ("linear_residual", "monotonic_residual"):
        try:
            if get_transform(name).requires_base:
                return name
        except Exception:
            continue
    raise RuntimeError("no base-requiring transform registered")


def test_missing_base_column_skip_is_logged(caplog):
    """When the split lacks the spec's base column the watchdog skips with a WARNING, not silently (37-F25)."""
    wrapper = SimpleNamespace(estimator_=object(), _extract_base_for_transform=lambda *a, **k: None, group_column=None)
    spec = {"transform_name": _base_requiring_transform_name(), "fitted_params": {}, "base_column": "absent_base"}
    split_df = pd.DataFrame({"x": [1.0, 2.0]})
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, split_df, [1.0, 2.0], composite_name="c", split_name="val")
    assert any("base columns" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
