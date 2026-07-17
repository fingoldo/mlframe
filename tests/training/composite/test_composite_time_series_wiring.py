"""M9: time-series transform auto-wiring.

When ``time_series_transforms_enabled`` is set, the three chronological-order
transforms (ewma_residual / rolling_quantile_ratio / frac_diff) join the
discovery candidate set; combined with ``time_column`` (M6) the screening sample
is time-ordered so they evaluate on a genuine forward sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig
from mlframe.training.composite import CompositeTargetDiscovery

_TS = ["ewma_residual", "rolling_quantile_ratio", "frac_diff"]


class TestM9Config:
    def test_default_excludes_time_series_transforms(self) -> None:
        cfg = CompositeTargetDiscoveryConfig()
        assert not any(t in cfg.transforms for t in _TS)

    def test_enabled_appends_all_three(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(time_series_transforms_enabled=True)
        for t in _TS:
            assert t in cfg.transforms

    def test_append_is_idempotent_and_order_preserving(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(time_series_transforms_enabled=True)
        assert cfg.transforms.count("frac_diff") == 1
        # The original core transforms still lead the list.
        assert cfg.transforms[0] == "diff"


class TestM9Discovery:
    def test_time_series_discovery_runs_clean_on_temporal_frame(self) -> None:
        """End-to-end: enabling the TS transforms + a time_column does not crash
        discovery and time-orders the screen."""
        rng = np.random.default_rng(0)
        n = 4000
        ts = np.arange(n)
        # AR(1) target so a frac_diff / ewma residual has something to model.
        y = np.empty(n)
        y[0] = 0.0
        for i in range(1, n):
            y[i] = 0.8 * y[i - 1] + rng.normal(0.0, 1.0)
        lag = np.concatenate([[0.0], y[:-1]])
        feat = rng.normal(0.0, 1.0, size=n)
        perm = rng.permutation(n)  # shuffle rows so the frame is not pre-sorted
        df = pd.DataFrame(
            {
                "y": y[perm],
                "lag": lag[perm],
                "feat": feat[perm],
                "ts": ts[perm],
            }
        )
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            mi_sample_n=800,
            base_candidates=["lag"],
            time_series_transforms_enabled=True,
            time_column="ts",
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, "y", ["lag", "feat"], np.arange(n), time_ordering=df["ts"].to_numpy())
        assert getattr(disc, "_screen_time_ordered_", False) is True
        # Discovery completed and produced a (possibly empty) spec list.
        assert isinstance(disc.specs_, list)
