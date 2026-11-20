"""An enrichment sweep that stops at the wall-clock deadline says so (audit 2026-11-20 B-F04)."""

import logging

import numpy as np

from mlframe.feature_selection.filters import _fe_deadline as dl
from mlframe.utils.log_throttle import reset_throttle_counts


def test_warn_deadline_truncation_names_the_stage_and_the_counts(caplog):
    """The warning carries the stage name and how many units were skipped of the total."""
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        dl.warn_deadline_truncation("binned_numeric_agg", 4, 9)
    assert any("binned_numeric_agg" in r.getMessage() and "4 of 9" in r.getMessage() for r in caplog.records)


def test_gate_sweep_with_an_expired_deadline_warns(monkeypatch, caplog):
    """The conditional-gate column sweep breaks at an expired deadline with a warning naming the skipped gate columns."""
    from mlframe.feature_selection.filters import _conditional_gate_fe_helpers as h

    reset_throttle_counts()
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_deadline.fe_deadline_passed", lambda: True)
    arrs = {"g1": np.arange(10.0), "g2": np.arange(10.0), "a": np.arange(10.0)}
    with caplog.at_level(logging.WARNING):
        h._scan_gate_column(["g1", "g2"], arrs, ["g1", "g2", "a"], lambda *a, **k: None)
    assert any("conditional_gate" in r.getMessage() and "2 of 2" in r.getMessage() for r in caplog.records)


def test_integer_verdict_cache_rescans_a_column_edited_outside_the_sampled_rows():
    """Audit 2026-11-20 B-F06: two frames of equal length and names that agree on the 64 sampled rows but differ elsewhere do not share an integer verdict."""
    import pandas as pd

    from mlframe.feature_selection.filters import _integer_columns as ic

    n = 1000
    base = pd.DataFrame({"a": np.arange(n, dtype=np.float64)})
    assert ic.eligible_integer_columns(base) == ["a"]
    sampled = set(np.linspace(0, n - 1, num=64).astype(np.int64).tolist())
    edited = base.copy()
    middle = next(i for i in range(5, n) if i not in sampled)
    edited.iloc[middle, 0] = middle + 0.37
    edited.iloc[[i for i in range(10, n) if i not in sampled][:400], 0] += 0.41  # a visible share of non-integer values, none on a sampled row
    assert ic._fingerprint(edited, ["a"]) == ic._fingerprint(base, ["a"])
    assert ic.eligible_integer_columns(edited) == []
    assert ic.eligible_integer_columns(base) == ["a"]
