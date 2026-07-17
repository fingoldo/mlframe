"""E1: the pre-discovery base-target leakage guard drops same-time re-encodings of y but spares genuine lags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _ar1(n, rng, phi=0.7):
    """Generates an AR(1) series with autocorrelation phi, used as the base signal for leakage-guard fixtures."""
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


def test_leakage_guard_drops_monotone_reencoding_keeps_lag():
    """Leakage guard drops a same-time monotone re-encoding of the target (high Spearman, escapes Pearson) but keeps a genuine 1-step lag."""
    rng = np.random.default_rng(0)
    n = 500
    y = _ar1(n, rng)
    base_leaky = np.exp(y * 0.5)  # monotone same-time re-encoding: high Spearman, Pearson < 0.99999 (escapes corr guard)
    base_lag1 = np.empty(n)
    base_lag1[0] = y[0]
    base_lag1[1:] = y[:-1]  # genuine 1-step lag of y -> the canonical composite base, must be KEPT
    df = pd.DataFrame(
        {
            "base_leaky": base_leaky,
            "base_lag1": base_lag1,
            "f0": rng.standard_normal(n),
            "y": y,
        }
    )
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=400, detect_base_leakage=True)
    disc = CompositeTargetDiscovery(config=cfg)
    disc.fit(
        df=df,
        target_col="y",
        feature_cols=["base_leaky", "base_lag1", "f0"],
        train_idx=np.arange(n),
        time_ordering=np.arange(n),
    )
    dropped = {name for name, _reason in getattr(disc, "_leaky_bases_dropped_", [])}
    assert "base_leaky" in dropped, f"monotone re-encoding not flagged; dropped={dropped}"
    assert "base_lag1" not in dropped, "genuine lag(y) base was wrongly dropped by the guard"


def test_leakage_guard_noop_without_time_ordering():
    """Without time_ordering the guard must NOT run (autocorrelation must not be mistaken for leakage)."""
    rng = np.random.default_rng(1)
    n = 400
    y = _ar1(n, rng)
    df = pd.DataFrame({"base_lag1": np.r_[y[0], y[:-1]], "f0": rng.standard_normal(n), "y": y})
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=300, detect_base_leakage=True)
    disc = CompositeTargetDiscovery(config=cfg)
    disc.fit(df=df, target_col="y", feature_cols=["base_lag1", "f0"], train_idx=np.arange(n))  # no time_ordering
    assert not getattr(disc, "_leaky_bases_dropped_", [])  # guard inert -> nothing dropped
