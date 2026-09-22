"""The composite value report must build for a run that has no group column.

`build_composite_value_report(..., group_ids=None)` raised "len() of unsized object" inside `factorize`, the caller
caught it and logged "report build failed; continuing", so every ungrouped run silently lost the report.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite._value_report import build_composite_value_report


def _bed(n: int = 60):
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, n)
    raw = y + rng.normal(0, 1.0, n)
    comp = y + rng.normal(0, 0.5, n)
    return y, raw, comp


def test_no_groups_builds_a_report():
    y, raw, comp = _bed()
    report = build_composite_value_report(y, raw, comp, None)
    assert isinstance(report, dict) and report


def test_no_groups_equals_one_explicit_group():
    """None must mean "one group spanning the split", exactly as passing that group would."""
    y, raw, comp = _bed()
    a = build_composite_value_report(y, raw, comp, None)
    b = build_composite_value_report(y, raw, comp, np.zeros(len(y), dtype=np.int64))
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float):
            assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k
