"""Unit + biz_value tests for ``composite.autoconfig.suggest_discovery_config``.

The biz_value claim: on a synthetic TEMPORAL, RIGHT-SKEWED frame the suggester
must (a) detect the monotone time column, (b) enable the chronological-order
transforms, (c) add a tail-compressing skew y-transform, and (d) scale
``mi_sample_n`` to the row count. On a plain small frame it must return
CONSERVATIVE defaults (no time column, no TS transforms, full-train MI screen,
no skew transform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.autoconfig import (
    _MI_SAMPLE_LARGE,
    _MI_SAMPLE_SMALL_N,
    suggest_discovery_config,
)


def _temporal_right_skewed_frame(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    """Frame with a monotone integer timestamp, a strong autoregressive lag
    base, a couple of plain features, and a heavily right-skewed target."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)            # strictly monotone time index
    lag = np.cumsum(rng.normal(size=n)) * 0.3     # slow-moving AR base
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Right-skewed positive target: exponentiate a signal so the tail blows up.
    signal = 0.5 * lag + 0.4 * x1 + 0.1 * x2 + rng.normal(size=n) * 0.2
    y = np.exp(signal)                            # lognormal -> strong right skew
    return pd.DataFrame(
        {"ts": t, "lag": lag, "x1": x1, "x2": x2, "y": y}
    )


def _plain_small_frame(n: int = 800, seed: int = 1) -> pd.DataFrame:
    """Small shuffled frame, symmetric target, no time / structural column."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 0.5 * x1 + 0.3 * x2 + rng.normal(size=n) * 0.5   # symmetric, light-tail
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


# ----------------------------------------------------------------------
# biz_value: temporal right-skewed frame triggers every data-driven choice.
# ----------------------------------------------------------------------
def test_biz_val_autoconfig_temporal_right_skewed_triggers_all():
    df = _temporal_right_skewed_frame()
    cfg, rationale = suggest_discovery_config(
        df, "y", ["ts", "lag", "x1", "x2"],
    )

    # (a) time column detected + (b) TS transforms enabled.
    assert cfg.time_column == "ts", rationale
    assert cfg.time_series_transforms_enabled is True
    # The config validator appends the three chronological transforms.
    for ts_t in ("ewma_residual", "rolling_quantile_ratio", "frac_diff"):
        assert ts_t in cfg.transforms

    # (c) a skew / tail-compressing y-transform got added.
    assert "signed_power_y" in cfg.transforms, rationale
    # target is strictly positive (exp(...)) -> log_y added too.
    assert "log_y" in cfg.transforms

    # heavy tail -> stratified sampling with boosted strata.
    assert cfg.mi_sample_strategy == "stratified_quantile"
    assert cfg.mi_n_strata >= 30

    # (d) mi_sample_n scaled to the (mid-band) row count.
    assert cfg.mi_sample_n == _MI_SAMPLE_LARGE, rationale

    assert cfg.enabled is True
    # rationale must explain each steered choice.
    for key in ("time_column", "transforms", "mi_sample_n", "mi_sample_strategy"):
        assert key in rationale and rationale[key]


# ----------------------------------------------------------------------
# biz_value: plain small frame returns conservative defaults.
# ----------------------------------------------------------------------
def test_biz_val_autoconfig_plain_small_frame_conservative():
    df = _plain_small_frame()
    cfg, rationale = suggest_discovery_config(df, "y", ["x1", "x2", "x3"])

    # No chronological column -> no time wiring.
    assert cfg.time_column is None
    assert cfg.time_series_transforms_enabled is False
    for ts_t in ("ewma_residual", "rolling_quantile_ratio", "frac_diff"):
        assert ts_t not in cfg.transforms

    # Symmetric light-tail target -> no skew transform, no heavy-tail boost.
    assert "signed_power_y" not in cfg.transforms
    assert cfg.mi_n_strata < 30

    # Small frame -> full-train MI screen (no sampling).
    assert cfg.mi_sample_n is None, rationale
    assert "mi_sample_n" in rationale


# ----------------------------------------------------------------------
# mi_sample_n scales across the three row-count bands.
# ----------------------------------------------------------------------
def test_autoconfig_mi_sample_n_bands():
    # Small band -> None.
    small = _plain_small_frame(n=1000)
    cfg_s, _ = suggest_discovery_config(small, "y", ["x1", "x2", "x3"])
    assert cfg_s.mi_sample_n is None

    # Mid band -> capped at _MI_SAMPLE_LARGE (frame above the small threshold).
    mid_n = _MI_SAMPLE_SMALL_N + 5_000
    mid = _plain_small_frame(n=mid_n)
    cfg_m, _ = suggest_discovery_config(mid, "y", ["x1", "x2", "x3"])
    assert cfg_m.mi_sample_n == _MI_SAMPLE_LARGE


# ----------------------------------------------------------------------
# Structural hint: a near-affine predictor is surfaced as a dominant base hint.
# ----------------------------------------------------------------------
def test_autoconfig_structural_hint_near_affine_base():
    rng = np.random.default_rng(7)
    n = 5000
    base = rng.normal(size=n)
    # y is almost a pure affine function of `base` -> near-affine detector fires.
    y = 3.0 * base + 1.0 + rng.normal(size=n) * 0.01
    noise = rng.normal(size=n)
    df = pd.DataFrame({"base": base, "noise": noise, "y": y})
    cfg, rationale = suggest_discovery_config(df, "y", ["base", "noise"])
    assert cfg.dominant_features_hint is not None
    assert "base" in cfg.dominant_features_hint, rationale
    assert "dominant_features_hint" in rationale


# ----------------------------------------------------------------------
# Edge cases: empty frame + absent target return conservative defaults.
# ----------------------------------------------------------------------
def test_autoconfig_empty_frame_conservative():
    df = pd.DataFrame({"x1": [], "y": []})
    cfg, rationale = suggest_discovery_config(df, "y", ["x1"])
    assert cfg.enabled is True
    assert cfg.time_column is None
    assert cfg.time_series_transforms_enabled is False


def test_autoconfig_absent_target_conservative():
    df = _plain_small_frame(n=500)
    cfg, rationale = suggest_discovery_config(df, "not_a_col", ["x1", "x2"])
    assert cfg.time_column is None
    assert "enabled" in rationale


# ----------------------------------------------------------------------
# config_overrides win over suggested fields.
# ----------------------------------------------------------------------
def test_autoconfig_overrides_take_precedence():
    df = _temporal_right_skewed_frame()
    cfg, _ = suggest_discovery_config(
        df, "y", ["ts", "lag", "x1", "x2"],
        time_series_transforms_enabled=False,
        mi_sample_n=12345,
    )
    assert cfg.time_series_transforms_enabled is False
    assert cfg.mi_sample_n == 12345


# ----------------------------------------------------------------------
# polars parity: same temporal frame as polars triggers the same wiring.
# ----------------------------------------------------------------------
def test_autoconfig_polars_parity_temporal():
    pl = pytest.importorskip("polars")
    pdf = _temporal_right_skewed_frame(n=60_000)
    df = pl.from_pandas(pdf)
    cfg, _ = suggest_discovery_config(df, "y", ["ts", "lag", "x1", "x2"])
    assert cfg.time_column == "ts"
    assert cfg.time_series_transforms_enabled is True
    assert "signed_power_y" in cfg.transforms
    assert cfg.mi_sample_n == _MI_SAMPLE_LARGE
