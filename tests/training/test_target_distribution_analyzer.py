"""Tests for :mod:`mlframe.training._target_distribution_analyzer`.

Each pathology in the analyzer's contract is exercised against a synthetic
distribution where the pathology is unambiguously present (positive case)
plus a clean distribution where it must NOT trigger (negative case).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._target_distribution_analyzer import (
    TargetDistributionReport,
    _detect_multi_modal,
    _excess_kurtosis,
    _lag1_autocorr,
    _skewness,
    _within_between_group_variance_ratio,
    analyze_target_distribution,
)


# ---------------------------------------------------------------------------
# helper detectors -- direct numeric sanity
# ---------------------------------------------------------------------------


class TestHelperDetectors:
    def test_excess_kurtosis_gaussian_near_zero(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(20_000)
        # Sample excess kurtosis of N(0,1) over 20k samples typically |k| < 0.1.
        assert abs(_excess_kurtosis(y)) < 0.5

    def test_excess_kurtosis_heavy_tail_large(self):
        rng = np.random.default_rng(1)
        y = rng.standard_t(df=3, size=20_000)
        # Student-t with df=3 has infinite kurtosis; sample value > 5 reliably.
        assert _excess_kurtosis(y) > 5.0

    def test_skewness_symmetric_near_zero(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(20_000)
        assert abs(_skewness(y)) < 0.2

    def test_skewness_lognormal_positive(self):
        rng = np.random.default_rng(3)
        y = np.exp(rng.standard_normal(20_000))
        assert _skewness(y) > 2.0

    def test_lag1_autocorr_iid_near_zero(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(10_000)
        assert abs(_lag1_autocorr(y)) < 0.05

    def test_lag1_autocorr_strong_AR_high(self):
        rng = np.random.default_rng(5)
        n = 10_000
        ar = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            ar[i] = 0.9 * ar[i - 1] + rng.standard_normal()
        # AR(1) with phi=0.9 -> lag-1 autocorr ~ 0.9
        assert _lag1_autocorr(ar) > 0.8

    def test_multi_modal_unimodal_returns_false(self):
        rng = np.random.default_rng(6)
        y = rng.standard_normal(5000)
        is_mm, n_peaks, sep = _detect_multi_modal(y)
        assert is_mm is False, f"unimodal gaussian flagged as multi-modal (n_peaks={n_peaks}, sep={sep})"

    def test_multi_modal_bimodal_returns_true(self):
        rng = np.random.default_rng(7)
        y = np.concatenate([
            rng.normal(loc=-5, scale=0.5, size=2500),
            rng.normal(loc=+5, scale=0.5, size=2500),
        ])
        is_mm, n_peaks, sep = _detect_multi_modal(y)
        assert is_mm is True, f"clean bimodal not flagged: n_peaks={n_peaks}, sep={sep}"
        assert n_peaks >= 2
        # Symmetric bimodal pegs around 2.0 global stds by construction; assert above the threshold.
        assert sep >= 1.8

    def test_within_between_group_variance_strongly_clustered(self):
        rng = np.random.default_rng(8)
        groups = np.repeat(np.arange(10), 200)
        group_means = rng.uniform(0, 100, 10)
        y = group_means[groups] + rng.normal(0, 0.5, 2000)
        # Within std ~ 0.5; between std ~ 30 -> ratio ~ 0.017
        ratio = _within_between_group_variance_ratio(y, groups)
        assert ratio < 0.1

    def test_within_between_group_variance_uniform_groups(self):
        rng = np.random.default_rng(9)
        groups = np.repeat(np.arange(10), 200)
        y = rng.normal(0, 1, 2000)  # target unrelated to group
        ratio = _within_between_group_variance_ratio(y, groups)
        # Within ~ 1, between ~ 1/sqrt(200) ~ 0.07 -> ratio >> 1
        assert ratio > 1.0


# ---------------------------------------------------------------------------
# analyze_target_distribution -- regression scenarios
# ---------------------------------------------------------------------------


class TestRegressionAnalyzer:
    def test_clean_gaussian_no_pathologies(self):
        rng = np.random.default_rng(100)
        y = rng.standard_normal(5000)
        rep = analyze_target_distribution(y, has_time_axis=False)
        assert isinstance(rep, TargetDistributionReport)
        assert rep.target_type == "regression"
        assert rep.pathologies == [], f"clean gaussian flagged: {rep.pathologies}"
        assert rep.knob_overrides == {}

    def test_heavy_tail_recommends_huber(self):
        rng = np.random.default_rng(101)
        y = rng.standard_t(df=3, size=5000)
        rep = analyze_target_distribution(y, has_time_axis=False)
        assert any("heavy_tail" in p for p in rep.pathologies), rep.pathologies
        assert rep.knob_overrides.get("mlp_kwargs", {}).get("model_params", {}).get("loss_fn") == "huber"
        assert rep.knob_overrides.get("lgb_kwargs", {}).get("objective") == "huber"
        assert "reg:pseudohubererror" in str(rep.knob_overrides.get("xgb_kwargs", {}).get("objective", ""))

    def test_strong_AR_recommends_no_layernorm(self):
        rng = np.random.default_rng(102)
        n = 4000
        ar = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            ar[i] = 0.9 * ar[i - 1] + rng.standard_normal() * 0.5
        rep = analyze_target_distribution(ar, has_time_axis=True)
        assert any("strong_AR" in p for p in rep.pathologies), rep.pathologies
        np_overrides = rep.knob_overrides.get("mlp_kwargs", {}).get("network_params", {})
        assert np_overrides.get("use_layernorm") is False

    def test_strong_AR_skipped_when_time_axis_false(self):
        rng = np.random.default_rng(103)
        n = 4000
        ar = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            ar[i] = 0.9 * ar[i - 1] + rng.standard_normal() * 0.5
        rep = analyze_target_distribution(ar, has_time_axis=False)
        # AR detector should not fire when caller says rows aren't ordered.
        assert not any("strong_AR" in p for p in rep.pathologies), rep.pathologies

    def test_multi_modal_flag(self):
        rng = np.random.default_rng(104)
        y = np.concatenate([
            rng.normal(-5, 0.5, 2500),
            rng.normal(+5, 0.5, 2500),
        ])
        rep = analyze_target_distribution(y, has_time_axis=False)
        assert any("multi_modal" in p for p in rep.pathologies), rep.pathologies

    def test_skewed_target_flag(self):
        rng = np.random.default_rng(105)
        y = np.exp(rng.standard_normal(5000)).astype(np.float64)
        rep = analyze_target_distribution(y, has_time_axis=False)
        assert any("skewed_target" in p for p in rep.pathologies), rep.pathologies

    def test_near_constant_target_hard_warn(self):
        # Constant float target with explicit regression type so the auto-classify
        # heuristic (which treats single-unique-value floats as classification)
        # doesn't intercept. A truly constant target should hit the regression-side
        # near_constant detector and short-circuit before kurtosis/skew run.
        y = np.full(5000, 42.0)
        rep = analyze_target_distribution(y, has_time_axis=False, target_type="regression")
        assert any("near_constant" in p for p in rep.pathologies), rep.pathologies
        # And the analyzer must NOT have run downstream detectors that would crash on sigma=0.
        assert "excess_kurtosis" not in rep.diagnostics

    def test_clustered_target_with_group_ids(self):
        rng = np.random.default_rng(106)
        groups = np.repeat(np.arange(20), 100)
        means = rng.uniform(0, 50, 20)
        y = means[groups] + rng.normal(0, 0.3, 2000)
        rep = analyze_target_distribution(y, group_ids=groups, has_time_axis=False)
        assert any("clustered_target" in p for p in rep.pathologies), rep.pathologies
        assert rep.knob_overrides.get("split_config", {}).get("prefer_group_aware") is True

    def test_clustered_target_skipped_without_group_ids(self):
        rng = np.random.default_rng(107)
        groups = np.repeat(np.arange(20), 100)
        means = rng.uniform(0, 50, 20)
        y = means[groups] + rng.normal(0, 0.3, 2000)
        rep = analyze_target_distribution(y, has_time_axis=False)  # NO group_ids
        # Without group_ids the clustered-target detector can't trip, regardless of underlying structure.
        assert not any("clustered_target" in p for p in rep.pathologies), rep.pathologies


# ---------------------------------------------------------------------------
# analyze_target_distribution -- classification scenarios
# ---------------------------------------------------------------------------


class TestClassificationAnalyzer:
    def test_balanced_two_class_no_pathology(self):
        rng = np.random.default_rng(200)
        y = rng.integers(0, 2, size=4000)
        rep = analyze_target_distribution(y)
        assert rep.target_type == "classification"
        assert rep.pathologies == [], rep.pathologies

    def test_class_imbalance_recommends_balanced_weights(self):
        # 95% class 0, 5% class 1 -> ratio 19x > 10x threshold
        y = np.zeros(4000, dtype=np.int32)
        y[:200] = 1  # 5%
        rep = analyze_target_distribution(y)
        assert any("class_imbalance" in p for p in rep.pathologies), rep.pathologies
        assert rep.knob_overrides.get("lgb_kwargs", {}).get("class_weight") == "balanced"
        assert rep.knob_overrides.get("cb_kwargs", {}).get("auto_class_weights") == "Balanced"

    def test_rare_class_flag(self):
        # Three classes, third one with only 50 samples (below the default 100 threshold).
        y = np.concatenate([
            np.zeros(2000, dtype=np.int32),
            np.ones(1950, dtype=np.int32),
            np.full(50, 2, dtype=np.int32),
        ])
        rep = analyze_target_distribution(y)
        assert any("rare_classes" in p for p in rep.pathologies), rep.pathologies

    def test_near_singleton_class_flag(self):
        # 99.5% class 0
        y = np.zeros(2000, dtype=np.int32)
        y[-10:] = 1
        rep = analyze_target_distribution(y)
        assert any("near_singleton_class" in p for p in rep.pathologies), rep.pathologies

    def test_single_class_short_circuits(self):
        y = np.zeros(2000, dtype=np.int32)
        rep = analyze_target_distribution(y)
        assert any("single_class" in p for p in rep.pathologies), rep.pathologies


# ---------------------------------------------------------------------------
# merge_into_config
# ---------------------------------------------------------------------------


class TestMergeIntoConfig:
    def test_recommendations_fill_gaps_but_preserve_user_values(self):
        rng = np.random.default_rng(300)
        # Heavy-tail target so we know which knobs are recommended.
        y = rng.standard_t(df=3, size=5000)
        rep = analyze_target_distribution(y, has_time_axis=False)
        # User config has mlp_kwargs with their own loss_fn -- must be preserved.
        user_config = {"mlp_kwargs": {"model_params": {"loss_fn": "mse", "learning_rate": 1e-3}}}
        merged = rep.merge_into_config(user_config, override_existing=False)
        # User's loss_fn (mse) wins; learning_rate untouched.
        assert merged["mlp_kwargs"]["model_params"]["loss_fn"] == "mse"
        assert merged["mlp_kwargs"]["model_params"]["learning_rate"] == 1e-3
        # But the recommendation for lgb (objective=huber) lands because user had no lgb_kwargs.
        assert merged["lgb_kwargs"]["objective"] == "huber"

    def test_override_existing_lets_recommendation_win(self):
        rng = np.random.default_rng(301)
        y = rng.standard_t(df=3, size=5000)
        rep = analyze_target_distribution(y, has_time_axis=False)
        user_config = {"mlp_kwargs": {"model_params": {"loss_fn": "mse"}}}
        merged = rep.merge_into_config(user_config, override_existing=True)
        # With override_existing=True, the recommendation (huber) wins.
        assert merged["mlp_kwargs"]["model_params"]["loss_fn"] == "huber"

    def test_non_dict_user_slot_preserved(self):
        # Caller had ``mlp_kwargs=None`` (or some non-dict sentinel): the merger
        # must NOT crash; it bails on that slot. The slot stays as the caller put it.
        rng = np.random.default_rng(302)
        y = rng.standard_t(df=3, size=5000)
        rep = analyze_target_distribution(y, has_time_axis=False)
        user_config = {"mlp_kwargs": None}
        merged = rep.merge_into_config(user_config)
        assert merged["mlp_kwargs"] is None
