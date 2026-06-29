"""alpha-drift gate must require a practically-meaningful effect size, not z alone. At large n the z-test SE shrinks
(~1/sqrt(n)) so a negligible slope shift trips z >> 3 and cascade-drops good specs (prod TVT: 23/62). The gate now
drops only when z > threshold AND |a1-a2|*std(base)/std(y) >= alpha_drift_min_effect_size."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery._eval_stats import apply_alpha_drift_gate
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _run_gate(delta: float, *, min_effect: float, n: int = 200_000):
    """Fit y = (1+delta-on-second-half)*base, run the alpha-drift gate, return (kept_names, flags)."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, size=n)
    y = base.copy()
    half = n // 2
    y[half:] = (1.0 + delta) * base[half:]
    y += rng.normal(0.0, 0.01, size=n)  # tiny noise so the OLS SE is small -> z is large at this n
    spec = SimpleNamespace(transform_name="linear_residual", base_column="b", name="lr-b")
    cfg = CompositeTargetDiscoveryConfig(
        detect_linear_residual_alpha_drift=True, alpha_drift_z_threshold=3.0,
        reject_on_alpha_drift=True, alpha_drift_min_effect_size=min_effect,
    )
    fake = SimpleNamespace(config=cfg, _auto_base_pool={"b": base})
    df = pd.DataFrame({"b": base, "y": y})
    kept = apply_alpha_drift_gate(
        fake, [spec], df=df, train_idx=np.arange(n), y_full=y,
        extract_column_array=lambda d, c: d[c].to_numpy(),
        linear_residual_fit=None,
    )
    return [s.name for s in kept], fake._alpha_drift_flags


def test_negligible_slope_drift_high_z_is_kept_with_effect_floor():
    kept, flags = _run_gate(delta=0.004, min_effect=0.01)
    assert flags["lr-b"]["z_score"] > 3.0, "negligible drift should still trip the z-test at this n"
    assert flags["lr-b"]["effect_size"] < 0.01, "but its effect size must be below the floor"
    assert kept == ["lr-b"], "a statistically-but-not-practically drifted spec must be kept"


def test_large_slope_drift_is_still_dropped():
    kept, flags = _run_gate(delta=0.6, min_effect=0.01)
    assert flags["lr-b"]["z_score"] > 3.0 and flags["lr-b"]["effect_size"] >= 0.01
    assert kept == [], "a practically-meaningful slope drift must still be dropped"


def test_legacy_z_only_behaviour_when_floor_zero():
    # min_effect=0 reproduces the pre-fix z-only gate: the negligible-drift spec is dropped.
    kept, _ = _run_gate(delta=0.004, min_effect=0.0)
    assert kept == [], "with the floor disabled the z-only gate drops even negligible drift (legacy)"
