"""Left-skewed targets (e.g. a 0..5 score piled up at 5) must not get right-tail compressors."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._skew_gate import RIGHT_TAIL_COMPRESSORS, left_skewed_right_tail_skips


def test_left_skewed_score_skips_compressors():
    rng = np.random.default_rng(0)
    y = np.clip(5.0 - rng.exponential(0.3, 50000), 0.0, 5.0)
    assert left_skewed_right_tail_skips(y) == RIGHT_TAIL_COMPRESSORS
    assert "yeo_johnson_y" not in RIGHT_TAIL_COMPRESSORS


def test_right_skewed_target_keeps_them():
    rng = np.random.default_rng(1)
    assert left_skewed_right_tail_skips(rng.lognormal(0, 1.5, 50000)) == frozenset()


def test_discovery_skips_them_on_left_skewed_target():
    import pandas as pd

    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=n)
    y = np.clip(5.0 - rng.exponential(0.3, n) - 0.2 * np.abs(x), 0.0, 5.0)
    df = pd.DataFrame({"x": x, "b": rng.normal(size=n), "y": y})
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="mi", base_candidates=["b"], transforms=["log_y", "cbrt_y", "yeo_johnson_y"],
        require_beats_raw_baseline=False, random_state=0,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, "y", ["x", "b"], np.arange(n))
    names = {r.get("transform_name") for r in disc.report()}
    assert not names & {"log_y", "cbrt_y"}
