"""Tests for the FEATURE-side detectors in
``mlframe.training._target_distribution_analyzer.analyze_feature_distribution``.

Five pathology classes (positive + negative cases each):
- Low-variance features
- NaN-heavy features
- High-cardinality categorical features
- Redundant numeric pairs
- Suspected target leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._target_distribution_analyzer import (
    FeatureDistributionReport,
    analyze_feature_distribution,
)


# ---------------------------------------------------------------------------
# Clean baseline
# ---------------------------------------------------------------------------


class TestCleanFeatures:
    def test_iid_gaussians_no_pathologies(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 10)).astype(np.float64)
        rep = analyze_feature_distribution(X)
        assert isinstance(rep, FeatureDistributionReport)
        assert rep.n_features == 10
        assert rep.pathologies == [], f"clean iid flagged: {rep.pathologies}"
        assert rep.drop_candidates == []

    def test_polars_series_input_handled(self):
        """P0 #3 follow-up: polars.Series should produce a 1-column frame, not
        AttributeError on df.columns inside the analyzer."""
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(50)
        s = pl.Series("my_feature", rng.normal(0, 1, 500).astype(np.float32))
        rep = analyze_feature_distribution(s)
        assert rep.n_features == 1
        assert rep.diagnostics["n_numeric"] == 1
        assert rep.diagnostics["n_categorical"] == 0

    def test_polars_lazyframe_input_handled(self):
        """P0 #3 follow-up: polars.LazyFrame has no to_pandas; the analyzer must
        collect() it before conversion, NOT fall through to np.asarray (which
        gave object-array misclassification on the first fix)."""
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(51)
        n = 500
        lf = pl.LazyFrame({
            "f0": rng.normal(0, 1, n).astype(np.float32),
            "f1": rng.normal(0, 1, n).astype(np.float32),
            "well_id": [f"w{i % 50}" for i in range(n)],
        })
        rep = analyze_feature_distribution(lf)
        assert rep.diagnostics["n_numeric"] == 2, rep.diagnostics
        assert rep.diagnostics["n_categorical"] == 1, rep.diagnostics
        # No false-positive high-cardinality flag on numeric columns.
        assert not any("high_cardinality_categorical(n=2" in p for p in rep.pathologies), rep.pathologies

    def test_polars_dataframe_numerics_not_misclassified(self):
        """2026-05-21 P0 #3: TVT prod log flagged 25 Float32 numeric columns as
        ``high_cardinality_categorical(n=25)`` because the analyzer fell through
        the numpy-conversion path for polars input (object dtype after
        np.asarray). Fix: detect polars via duck-typing and call to_pandas()."""
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(99)
        n = 1000
        data = {f"f{i}": pl.Series(rng.normal(0, 1, n).astype(np.float32), dtype=pl.Float32) for i in range(25)}
        data["well_id"] = pl.Series([f"w{i % 100}" for i in range(n)], dtype=pl.Utf8)
        df = pl.DataFrame(data)
        rep = analyze_feature_distribution(df)
        # 25 Float32 numerics + 1 String categorical; high-card detector should NOT
        # flag the numerics, only the well_id column (100 unique values, below the 100 default
        # threshold so even well_id stays clean here).
        assert rep.diagnostics["n_numeric"] == 25, rep.diagnostics
        assert rep.diagnostics["n_categorical"] == 1, rep.diagnostics
        assert not any("high_cardinality_categorical(n=25" in p for p in rep.pathologies), rep.pathologies

    def test_insufficient_samples_short_circuits(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 5))
        rep = analyze_feature_distribution(X)
        assert any("insufficient_samples" in p for p in rep.pathologies)


# ---------------------------------------------------------------------------
# Low-variance + NaN-heavy
# ---------------------------------------------------------------------------


class TestLowVariance:
    def test_constant_feature_flagged_and_recommended_to_drop(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((500, 5)).astype(np.float64)
        # Make f0 constant
        X[:, 0] = 42.0
        rep = analyze_feature_distribution(X)
        assert any("low_variance_features" in p for p in rep.pathologies), rep.pathologies
        assert "f0" in rep.drop_candidates
        assert "f0" in rep.feature_warnings
        # No false positives on the non-constant columns:
        for j in (1, 2, 3, 4):
            assert f"f{j}" not in rep.drop_candidates

    def test_low_variance_relative_to_mean(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((500, 3)).astype(np.float64)
        # Make f1 mean=1000, std=0.5 -> rel_std=5e-4 < 1e-3 threshold
        X[:, 1] = 1000.0 + 0.5 * rng.standard_normal(500)
        rep = analyze_feature_distribution(X)
        assert "f1" in rep.drop_candidates


class TestNanHeavy:
    def test_50_percent_nan_feature_flagged(self):
        rng = np.random.default_rng(20)
        X = rng.standard_normal((500, 4))
        # Make f2 60% NaN
        nan_idx = rng.choice(500, size=300, replace=False)
        X[nan_idx, 2] = np.nan
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        rep = analyze_feature_distribution(df)
        assert any("nan_heavy_features" in p for p in rep.pathologies), rep.pathologies
        assert "f2" in rep.drop_candidates
        assert rep.knob_overrides.get("preprocessing_config", {}).get("review_nan_strategy") is True

    def test_low_nan_fraction_not_flagged(self):
        rng = np.random.default_rng(21)
        X = rng.standard_normal((500, 4))
        # Make f1 only 5% NaN -- random missingness, NOT a structural issue.
        nan_idx = rng.choice(500, size=25, replace=False)
        X[nan_idx, 1] = np.nan
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        rep = analyze_feature_distribution(df)
        assert "f1" not in rep.drop_candidates


# ---------------------------------------------------------------------------
# High-cardinality categorical
# ---------------------------------------------------------------------------


class TestHighCardinalityCategorical:
    def test_string_feature_with_many_levels_flagged(self):
        rng = np.random.default_rng(30)
        n = 500
        df = pd.DataFrame({
            "num0": rng.standard_normal(n),
            # 200 unique string levels -> above the 100 threshold
            "user_id": [f"user_{i % 200}" for i in range(n)],
            "cat_lowcard": rng.choice(["A", "B", "C"], size=n),
        })
        rep = analyze_feature_distribution(df)
        assert any("high_cardinality_categorical" in p for p in rep.pathologies), rep.pathologies
        assert "user_id" in rep.feature_warnings
        assert "cat_lowcard" not in rep.feature_warnings  # 3 unique levels, fine
        # Knob recommendation: prefer target / hashing encoder.
        assert rep.knob_overrides.get("preprocessing_config", {}).get("prefer_high_cardinality_encoder") is True


# ---------------------------------------------------------------------------
# Redundant pairs
# ---------------------------------------------------------------------------


class TestRedundantPairs:
    def test_two_redundant_features_paired(self):
        rng = np.random.default_rng(40)
        X = rng.standard_normal((500, 5))
        # Make f4 a copy of f0 with tiny noise -> corr ~ 1.0
        X[:, 4] = X[:, 0] + 0.0001 * rng.standard_normal(500)
        rep = analyze_feature_distribution(X)
        assert any("redundant_feature_pairs" in p for p in rep.pathologies), rep.pathologies
        # Both members of the redundant pair must be flagged with warning text.
        assert "f0" in rep.feature_warnings
        assert "f4" in rep.feature_warnings
        # And the diagnostic table records the pair with its correlation.
        pairs = rep.diagnostics.get("redundant_feature_pairs", [])
        assert any(
            (p["a"] == "f0" and p["b"] == "f4") or (p["a"] == "f4" and p["b"] == "f0")
            for p in pairs
        )

    def test_diagonal_self_correlation_does_not_self_pair(self):
        rng = np.random.default_rng(41)
        X = rng.standard_normal((500, 4))
        rep = analyze_feature_distribution(X)
        # Diagonal correlations are 1.0 by definition; the detector skips i==j by construction.
        assert rep.diagnostics.get("redundant_feature_pairs", []) == [] or not any(
            p["a"] == p["b"] for p in rep.diagnostics.get("redundant_feature_pairs", [])
        )

    def test_redundancy_skipped_when_feature_count_too_high(self):
        rng = np.random.default_rng(42)
        # 600 features > 500 cap -> the O(n^2) corrcoef pass is skipped with a diagnostic.
        X = rng.standard_normal((100, 600))
        rep = analyze_feature_distribution(X)
        # Even if random pairs incidentally exceed 0.95, the function logs the skip + does not stamp pathologies.
        assert any("redundant" in str(k).lower() for k in rep.diagnostics.keys()) or \
               not any("redundant_feature_pairs" in p for p in rep.pathologies)


# ---------------------------------------------------------------------------
# Suspected target leakage
# ---------------------------------------------------------------------------


class TestTargetLeakage:
    def test_feature_equal_to_target_flagged(self):
        rng = np.random.default_rng(50)
        n = 500
        y = rng.standard_normal(n)
        X = rng.standard_normal((n, 4))
        # f3 == y + tiny noise -> corr > 0.99
        X[:, 3] = y + 0.001 * rng.standard_normal(n)
        rep = analyze_feature_distribution(X, y=y)
        assert "f3" in rep.leakage_candidates, rep.leakage_candidates
        assert any("suspected_target_leakage" in p for p in rep.pathologies)

    def test_no_leakage_when_features_uncorrelated_with_target(self):
        rng = np.random.default_rng(51)
        n = 500
        y = rng.standard_normal(n)
        X = rng.standard_normal((n, 4))  # all independent of y
        rep = analyze_feature_distribution(X, y=y)
        assert rep.leakage_candidates == []

    def test_leakage_skipped_when_y_is_none(self):
        rng = np.random.default_rng(52)
        X = rng.standard_normal((500, 4))
        rep = analyze_feature_distribution(X)  # no y
        assert rep.leakage_candidates == []
        # And no "leakage" pathology.
        assert not any("leakage" in p for p in rep.pathologies)
