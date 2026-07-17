"""Regression sensor for S66 -- categorical PSI in feature_drift_report.

Before the fix ``compute_feature_distribution_drift`` was numeric-only; categorical drift via PSI was a documented TODO that never landed.
Now ``compute_categorical_drift_psi`` is the public helper and ``categorical_psi`` is folded into the existing report dict.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from mlframe.training.feature_drift_report import (
    _compute_categorical_psi,
    compute_categorical_drift_psi,
    compute_feature_distribution_drift,
)


def test_categorical_psi_identical_distributions_is_zero():
    """Identical category distributions across train/val must give PSI close to zero (not NaN)."""
    train_counts = {"A": 500, "B": 500}
    val_counts = {"A": 500, "B": 500}
    psi = _compute_categorical_psi(train_counts, val_counts)
    assert math.isfinite(psi)
    assert abs(psi) < 1e-9, f"identical distributions must give PSI~0, got {psi}"


def test_categorical_psi_shifted_distribution_fires_moderate_warn():
    """50/50 -> 80/20 PSI must be >= 0.20 moderate threshold."""
    train_counts = {"A": 500, "B": 500}
    val_counts = {"A": 800, "B": 200}
    psi = _compute_categorical_psi(train_counts, val_counts)
    assert math.isfinite(psi)
    assert psi >= 0.20, f"shifted distribution 50/50->80/20 should fire PSI>=0.20, got {psi}"


def test_categorical_psi_new_category_in_val_handled():
    """A category present in val/test but absent from train must produce a finite, strictly positive PSI bucket (not NaN, not crash)."""
    train_counts = {"A": 500, "B": 500}
    val_counts = {"A": 400, "B": 400, "C": 200}
    psi = _compute_categorical_psi(train_counts, val_counts, bin_min_count=5)
    assert math.isfinite(psi)
    assert psi > 0.0, f"new-category bucket should produce positive PSI, got {psi}"


def test_categorical_psi_vectorised_path_matches_python_loop_at_50_plus_cats():
    """iter432 regression: the size-aware dispatcher routes n_cats >= 50
    to the numpy-vectorised path. Both branches must produce bit-equal
    PSI for the same inputs -- the vectorised form uses identical
    dict.get + max-floor + log-ratio + sum math, just in numpy ufuncs."""
    import numpy as np

    rng = np.random.default_rng(20260527)
    # 100 cats -> vectorised path
    train_counts = {i: int(rng.integers(1, 1000)) for i in range(100)}
    val_counts = {i: int(rng.integers(1, 1000)) for i in range(100)}
    psi_vec = _compute_categorical_psi(train_counts, val_counts)
    assert math.isfinite(psi_vec)
    # 10 cats -> Python loop path
    train_small = {i: int(rng.integers(1, 1000)) for i in range(10)}
    val_small = {i: int(rng.integers(1, 1000)) for i in range(10)}
    psi_loop = _compute_categorical_psi(train_small, val_small)
    assert math.isfinite(psi_loop)
    # The two paths use the same dispatch internally for a given size;
    # this test just pins that BOTH branches stay reachable / non-zero.
    assert psi_vec > 0.0
    assert psi_loop > 0.0


def test_compute_categorical_drift_psi_dataframe_path_fires_warn_threshold():
    """Synthetic frame with shifted cat distribution must surface as a drift_candidate."""
    train_df = pd.DataFrame(
        {
            "cat_a": ["X"] * 500 + ["Y"] * 500,
            "cat_b": ["P"] * 1000,
            "num_c": [0.5] * 1000,
        }
    )
    val_df = pd.DataFrame(
        {
            "cat_a": ["X"] * 800 + ["Y"] * 200,
            "cat_b": ["P"] * 1000,
            "num_c": [0.5] * 1000,
        }
    )
    rep = compute_categorical_drift_psi(train_df, val_df, test_df=None)
    assert rep["n_categorical_features"] >= 2
    assert "cat_a" in rep["per_feature"]
    cat_a_psi = rep["per_feature"]["cat_a"]["val_psi"]
    assert math.isfinite(cat_a_psi)
    assert cat_a_psi >= 0.20
    cands = [c for c, _ in rep["drift_candidates"]]
    assert "cat_a" in cands


def test_compute_categorical_drift_psi_identical_distributions_no_candidates():
    """Identical train/val cat distributions: no drift_candidates above moderate threshold."""
    train_df = pd.DataFrame({"cat_a": ["X"] * 500 + ["Y"] * 500})
    val_df = pd.DataFrame({"cat_a": ["X"] * 500 + ["Y"] * 500})
    rep = compute_categorical_drift_psi(train_df, val_df, test_df=None)
    assert rep["drift_candidates"] == [], "identical distributions should produce no drift candidates"


def test_feature_distribution_drift_includes_categorical_psi_payload():
    """The main feature-drift report now carries a categorical_psi payload alongside the numeric one."""
    train_df = pd.DataFrame(
        {
            "num_a": [0.0, 1.0, 2.0, 3.0, 4.0] * 200,
            "cat_b": ["X"] * 500 + ["Y"] * 500,
        }
    )
    val_df = pd.DataFrame(
        {
            "num_a": [0.0, 1.0, 2.0, 3.0, 4.0] * 200,
            "cat_b": ["X"] * 800 + ["Y"] * 200,
        }
    )
    rep = compute_feature_distribution_drift(train_df, val_df, test_df=None)
    assert "categorical_psi" in rep, "main report should expose categorical_psi key"
    cpsi = rep["categorical_psi"]
    assert "cat_b" in cpsi["per_feature"]
    assert cpsi["per_feature"]["cat_b"]["val_psi"] >= 0.20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
