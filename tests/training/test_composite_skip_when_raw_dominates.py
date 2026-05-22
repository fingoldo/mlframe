"""Composite-discovery skip gates: config wiring smoke + default assertions.

Two skip gates short-circuit ``CompositeTargetDiscovery.fit`` to "no kept
specs" when the raw model already dominates:
  - ``composite_skip_when_raw_dominates_ratio``: raw_baseline / y_std
  - ``composite_skip_when_ablation_delta_pct``: BD ablation delta% of top hint

Both default to 0.0 (always run discovery). Setting them >0 re-enables
the heuristic that an Identity-MLP or other mis-configured downstream
model wouldn't benefit from a residual composite -- an assumption that
holds for the Ridge/CB/XGB/LGB baseline but breaks for model-mix
suites that include neural / pathological configs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig


class TestComposiSkipConfig:
    def test_raw_dominates_ratio_default_is_off(self) -> None:
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.composite_skip_when_raw_dominates_ratio == 0.0

    def test_ablation_delta_pct_default_is_off(self) -> None:
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.composite_skip_when_ablation_delta_pct == 0.0

    def test_config_flag_accepts_positive_ratio(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(composite_skip_when_raw_dominates_ratio=0.05)
        assert cfg.composite_skip_when_raw_dominates_ratio == 0.05

    def test_config_flag_accepts_positive_ablation_pct(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(composite_skip_when_ablation_delta_pct=500.0)
        assert cfg.composite_skip_when_ablation_delta_pct == 500.0


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
