"""Cluster 12 regression: the sweep writes strict JSON, so NaN and infinity become null (audit 2026-11-20, finding 12-F101).

Python's json writes NaN by default, which strict parsers reject. _strict_json maps non-finite floats to None, and the dump refuses
any value it misses (allow_nan=False).
"""

import json
import math

from mlframe.feature_selection._benchmarks.fs_hybrid.sweep_recovery_vs_n import _strict_json


def test_nan_and_inf_become_null_and_round_trip_strictly():
    """Non-finite floats become None and the result dumps under allow_nan=False (12-F101)."""
    rows = [{"recall": float("nan"), "auc": float("inf"), "n": 3, "nested": [1.0, float("-inf")]}]
    cleaned = _strict_json(rows)
    text = json.dumps(cleaned, allow_nan=False)
    back = json.loads(text)
    assert back[0]["recall"] is None
    assert back[0]["auc"] is None
    assert back[0]["nested"] == [1.0, None]
    assert back[0]["n"] == 3


def test_finite_values_are_unchanged():
    """Control: finite floats and other values pass through untouched (12-F101)."""
    rows = [{"x": 0.25, "y": "ok", "z": None}]
    assert _strict_json(rows) == rows
    assert math.isfinite(_strict_json([{"x": 1.5}])[0]["x"])
