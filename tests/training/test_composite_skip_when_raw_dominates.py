"""#10 biz_val: skip composite training when raw model already dominates.

Production TVT log: Ridge on raw TVT achieved MAE=7.89 (better than CB / XGB / LGB); the two discovered composites (monres-Y, monresYj-Y) produced IDENTICAL metrics to raw -- pure compute loss.

Pack #10 adds ``composite_skip_when_raw_dominates_ratio`` to ``CompositeTargetDiscoveryConfig``. Default 0.0 (off, back-compat). When > 0, discovery measures ``raw_baseline / y_std`` -- the fraction of y's variance the raw model could not explain -- and short-circuits to "no kept specs" if the fraction is below the threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig


class TestComposiSkipConfig:
    def test_config_flag_exists_with_zero_default(self) -> None:
        """``composite_skip_when_raw_dominates_ratio`` MUST default to 0.0 (off / back-compat)."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.composite_skip_when_raw_dominates_ratio == 0.0

    def test_config_flag_accepts_positive_ratio(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(composite_skip_when_raw_dominates_ratio=0.05)
        assert cfg.composite_skip_when_raw_dominates_ratio == 0.05


class TestDiscoveryHonoursSkipFlag:
    """Wiring smoke: discovery sees the config knob without crash."""

    def test_discovery_with_skip_flag_does_not_crash(self) -> None:
        from mlframe.training.composite_discovery import CompositeTargetDiscovery
        rng = np.random.default_rng(2)
        n = 1500
        x_a = rng.normal(50.0, 10.0, n)
        x_b = rng.normal(0.0, 5.0, n)
        # Mild multi-feature signal so raw is good but not trivial.
        y = 0.5 * x_a + 0.3 * x_b + rng.normal(0.0, 1.0, n)
        df = pd.DataFrame({
            "x_a": x_a, "x_b": x_b,
            "n0": rng.standard_normal(n), "n1": rng.standard_normal(n),
            "y": y,
        })
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=1000,
            composite_skip_when_raw_dominates_ratio=0.5,  # very lenient
        )
        disc = CompositeTargetDiscovery(config=cfg).fit(
            df=df, target_col="y", feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
        )
        # Just verify the discovery completed (no crash on the new code path).
        assert hasattr(disc, "specs_")
        assert hasattr(disc, "report_")
