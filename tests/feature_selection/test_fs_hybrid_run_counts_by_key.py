"""Audit 2026-11-20 finding 12-F39: the shared run-count helper reports completed and failed runs per key, so means are never silently over fewer runs."""

import sys
from pathlib import Path

import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.fixture(scope="module")
def shared():
    """Import the shared bench helpers with the bench folder on the path."""
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import _downstream_shared as mod
    finally:
        sys.path.remove(str(_FS_HYBRID))
    return mod


def test_failed_runs_are_counted_per_key(shared):
    """A config with one failed seed reports 2 completed and 1 failed; the other config is fully completed."""
    rows = [
        {"config": "A", "seed": 0, "auc_mean": 0.8},
        {"config": "A", "seed": 1, "auc_mean": 0.8},
        {"config": "A", "seed": 2, "error": "ValueError: x"},
        {"config": "B", "seed": 0, "auc_mean": 0.7},
    ]
    counts = shared.run_counts_by_key(rows)
    assert (int(counts.loc["A", "completed"]), int(counts.loc["A", "failed"])) == (2, 1)
    assert (int(counts.loc["B", "completed"]), int(counts.loc["B", "failed"])) == (1, 0)


def test_key_can_be_the_case_column(shared):
    """The probe keys its rows by case and config; the helper groups by whichever key is given."""
    rows = [{"case": "c1", "error": "boom"}, {"case": "c1", "auc_mean": 0.5}]
    counts = shared.run_counts_by_key(rows, key="case")
    assert (int(counts.loc["c1", "completed"]), int(counts.loc["c1", "failed"])) == (1, 1)
