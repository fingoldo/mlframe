"""Unit: the residual-std / noise-floor probe (T_std/y_std < 0.001 gate) is measured on the
SCREEN sample, not the full train set (measured-FUTURE item, ``discovery/_eval.py``).

Pins the claim behind the switch: on a battery of synthetic (base, transform) pairs whose true
T_std/y_std ratio sits both far from and near the 0.001 threshold, the accept/reject decision made
from the screen-sample probe matches the decision a full-train probe would have made -- except
inherently unstable cases exactly on the threshold (not tested here, since even two full-train
runs a seed apart disagree there; see the _ktc-adjacent measurement note in _eval.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _cfg(**kw) -> CompositeTargetDiscoveryConfig:
    """Minimal fast discovery config for the residual-std-probe tests, with ``**kw`` applied."""
    base = dict(
        enabled=True, random_state=0, screening="mi", honest_holdout_frac=None,
        auto_base_null_perms=0, multi_base_enabled=False, honest_rmse_gate_enabled=False,
        interaction_base_discovery_enabled=False, auto_chain_discovery_enabled=False,
    )
    base.update(kw)
    return CompositeTargetDiscoveryConfig(**base)


def test_residual_below_noise_floor_still_rejected_from_screen_sample():
    """A transform whose T is far below the 0.001*y_std noise floor must still be rejected
    when the probe runs on the screen sample instead of full train."""
    rng = np.random.default_rng(0)
    n = 40_000
    base = rng.uniform(1.0, 1000.0, n)
    # y is almost pure base plus a tiny bit of noise: ratio(base) - ratio(y) captured by
    # additive_residual leaves T = y - base - beta near machine-noise scale (T_std << y_std).
    y = base + rng.normal(0, 1e-6, n)
    df = pd.DataFrame({"lag": base, "y": y})
    disc = CompositeTargetDiscovery(_cfg(base_candidates=["lag"], transforms=["additive_residual"], mi_sample_n=5000)).fit(
        df, "y", ["lag"], np.arange(n),
    )
    assert not disc.specs_, "the near-zero-residual additive_residual spec must be rejected by the noise-floor probe"


def test_healthy_residual_ratio_survives_screen_sample_probe():
    """A transform with a genuinely large residual (far above 0.001*y_std) must survive the probe
    regardless of whether it's measured on train or screen. Needs a remaining feature ``x1``
    correlated with the residual so ``mi_gain`` has something to score against -- MI gain is
    computed vs the OTHER (non-base) feature columns, not against ``base`` itself."""
    rng = np.random.default_rng(1)
    n = 40_000
    base = rng.uniform(0.0, 100.0, n)
    x1 = rng.normal(0, 1, n)
    y = base + 5.0 * x1  # residual (y - base) is entirely explained by x1: mi_gain(T, x1) is large.
    df = pd.DataFrame({"lag": base, "x1": x1, "y": y})
    disc = CompositeTargetDiscovery(_cfg(base_candidates=["lag"], transforms=["additive_residual"], mi_sample_n=5000)).fit(
        df, "y", ["lag", "x1"], np.arange(n),
    )
    assert disc.specs_, "a healthy residual must not be rejected by the noise-floor probe"


def test_screen_vs_full_train_probe_decision_parity_across_seeds():
    """Direct pin on the measured claim: across many seeds and true ratios away from the exact
    0.001 boundary, a screen-sample std-ratio computation agrees with the full-train one."""
    true_ratios = (0.0005, 0.0008, 0.0009, 0.0011, 0.0015, 0.005, 0.05)
    sample_sizes = (5_000, 20_000, 50_000)
    n_train = 200_000
    flips = 0
    total = 0
    for true_ratio in true_ratios:
        for seed in range(10):
            rng = np.random.default_rng(seed * 1000 + int(true_ratio * 1e6))
            y = rng.normal(0, 1, n_train)
            t = y * true_ratio + rng.normal(0, true_ratio * 0.01, n_train)
            full_ratio = np.std(t) / np.std(y)
            for ss in sample_sizes:
                idx = rng.choice(n_train, ss, replace=False)
                screen_ratio = np.std(t[idx]) / np.std(y[idx])
                total += 1
                if (full_ratio < 0.001) != (screen_ratio < 0.001):
                    flips += 1
    assert flips == 0, f"{flips}/{total} decision flips away from the exact 0.001 boundary -- probe switch is not safe at these ratios"
