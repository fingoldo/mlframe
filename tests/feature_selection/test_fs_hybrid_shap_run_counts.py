"""Audit 2026-11-20 finding 12-F32: the shap bench summary reports, per config, how many runs completed and how many failed."""

import sys
from pathlib import Path

import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.fixture(scope="module")
def shap_bench():
    """Import the shap bench with its sibling modules on the path."""
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import round2_shap_bench as mod
    finally:
        sys.path.remove(str(_FS_HYBRID))
    return mod


def test_failed_runs_are_counted_next_to_completed_ones(shap_bench):
    """A config that failed on one of three seeds reports 2 completed and 1 failed, so its mean is visibly over fewer runs."""
    rows = [
        {"config": "A", "seed": 0, "auc_mean": 0.8},
        {"config": "A", "seed": 1, "auc_mean": 0.8},
        {"config": "A", "seed": 2, "error": "ValueError: boom"},
        {"config": "B", "seed": 0, "auc_mean": 0.7},
    ]
    counts = shap_bench.config_run_counts(rows)
    assert int(counts.loc["A", "completed"]) == 2 and int(counts.loc["A", "failed"]) == 1
    assert int(counts.loc["B", "completed"]) == 1 and int(counts.loc["B", "failed"]) == 0
