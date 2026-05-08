"""Tests for ``QuantileRegressionConfig`` validators + TargetTypes.is_quantile."""

from __future__ import annotations

import pytest

from mlframe.training.configs import (
    QuantileRegressionConfig, ReportingConfig, TargetTypes,
)


class TestEnum:
    def test_enum_value(self):
        assert TargetTypes.QUANTILE_REGRESSION.value == "quantile_regression"

    def test_is_quantile_predicate(self):
        assert TargetTypes.QUANTILE_REGRESSION.is_quantile is True
        assert TargetTypes.REGRESSION.is_quantile is False
        assert TargetTypes.BINARY_CLASSIFICATION.is_quantile is False
        assert TargetTypes.LEARNING_TO_RANK.is_quantile is False

    def test_quantile_is_not_regression_or_classification(self):
        # QR is its own class; distinguishes from plain regression because
        # the consumer needs to know to expect (N, K) preds, not (N,).
        qr = TargetTypes.QUANTILE_REGRESSION
        assert qr.is_regression is False
        assert qr.is_classification is False
        assert qr.is_ranking is False
        assert qr.is_multi_output is False  # multi_output is for classification


class TestConfig:
    def test_defaults(self):
        cfg = QuantileRegressionConfig()
        assert cfg.alphas == (0.1, 0.5, 0.9)
        assert cfg.crossing_fix == "sort"
        assert cfg.point_estimate_alpha == 0.5
        assert cfg.coverage_pairs == ((0.1, 0.9),)
        assert cfg.wrapper_n_jobs == "auto"

    def test_custom_alphas(self):
        cfg = QuantileRegressionConfig(
            alphas=(0.05, 0.25, 0.5, 0.75, 0.95),
            point_estimate_alpha=0.5,
            coverage_pairs=((0.05, 0.95),),
        )
        assert cfg.alphas == (0.05, 0.25, 0.5, 0.75, 0.95)

    def test_empty_alphas_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuantileRegressionConfig(alphas=())

    def test_unsorted_alphas_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            QuantileRegressionConfig(alphas=(0.5, 0.1, 0.9))

    def test_out_of_range_alphas_rejected(self):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            QuantileRegressionConfig(alphas=(0.1, 0.5, 1.5))
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            QuantileRegressionConfig(alphas=(0.0, 0.5))
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            QuantileRegressionConfig(alphas=(0.5, 1.0))

    def test_duplicate_alphas_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            QuantileRegressionConfig(alphas=(0.1, 0.5, 0.5, 0.9))

    def test_invalid_crossing_fix_rejected(self):
        with pytest.raises(ValueError, match="crossing_fix"):
            QuantileRegressionConfig(crossing_fix="quantile-snap")

    def test_point_estimate_alpha_snaps_to_nearest(self):
        # User asks for 0.45; closest alpha in defaults is 0.5.
        cfg = QuantileRegressionConfig(
            alphas=(0.1, 0.5, 0.9), point_estimate_alpha=0.45,
        )
        assert cfg.point_estimate_alpha == 0.5

    def test_coverage_pair_must_be_in_alphas(self):
        with pytest.raises(ValueError, match="not in alphas"):
            QuantileRegressionConfig(
                alphas=(0.1, 0.5, 0.9),
                coverage_pairs=((0.05, 0.95),),
            )

    def test_coverage_pair_lo_lt_hi_enforced(self):
        with pytest.raises(ValueError, match="lo=.*must be < hi"):
            QuantileRegressionConfig(
                alphas=(0.1, 0.5, 0.9),
                coverage_pairs=((0.9, 0.1),),
            )


class TestReportingConfigQuantilePanels:
    def test_default_quantile_panels(self):
        cfg = ReportingConfig()
        assert "RELIABILITY" in cfg.quantile_panels
        assert "PIT_HIST" in cfg.quantile_panels

    def test_unknown_quantile_token_rejected(self):
        with pytest.raises(ValueError, match="Unknown quantile"):
            ReportingConfig(quantile_panels="RELIABILITY NOPE")

    def test_subset_quantile_template(self):
        cfg = ReportingConfig(quantile_panels="RELIABILITY WIDTH_DIST")
        assert cfg.quantile_panels == "RELIABILITY WIDTH_DIST"
