"""FS_BENCHMARKS_B-2: row["auc_mean"] was gated on any(aucs.values()) instead of
any(v is not None for v in aucs.values()); if one model produces a legitimate AUC of exactly 0.0 while
the others failed (None), the truthy check misfires and the real result is silently dropped as None."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment import compute_auc_mean


def test_zero_auc_with_other_models_failed_is_not_dropped():
    """A legitimate AUC of exactly 0.0, with every other model failed to None, must NOT be silently
    replaced by None -- the old any(aucs.values()) truthy check treated 0.0 the same as all-None."""
    aucs = {"model_a": 0.0, "model_b": None, "model_c": None}
    assert compute_auc_mean(aucs) == 0.0


def test_all_none_returns_none():
    """Every model failed -> no result to average -> None."""
    aucs = {"model_a": None, "model_b": None}
    assert compute_auc_mean(aucs) is None


def test_normal_mixed_values_averages_the_non_none_ones():
    """Sanity: the common case still averages only the surviving (non-None) AUC values."""
    aucs = {"model_a": 0.8, "model_b": None, "model_c": 0.6}
    assert compute_auc_mean(aucs) == 0.7
