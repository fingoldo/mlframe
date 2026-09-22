"""Every selection routine picks the null on data with no signal, in at least 9 of 10 seeds.

A selector that never prefers "nothing" turns noise into a shipped choice: the seasonal period picked the largest
candidate on pure noise, the power transform never had the identity on its grid, a spec found in one stability run of three
survived a truncated majority, and the drift check could not fire under the default threshold.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

_SEEDS = range(10)


def _majority(hits: list[bool]) -> bool:
    """At least 9 of the 10 seeded runs chose the null."""
    return sum(hits) >= 9


def test_seasonal_period_is_none_on_noise_and_the_true_period_on_a_season():
    """No seasonality (period 1) on noise; 12, not a multiple of it, on a period-12 signal."""
    t = TRANSFORMS_REGISTRY["seasonal_residual"]
    noise = [t.fit(np.random.default_rng(s).normal(size=400), np.zeros(400)).get("period") in (None, 1) for s in _SEEDS]
    x = np.arange(480)
    season = [t.fit(np.sin(2 * np.pi * x / 12) + np.random.default_rng(s).normal(0.0, 0.3, 480), np.zeros(480)).get("period") == 12 for s in _SEEDS]
    assert _majority(noise) and _majority(season), (noise, season)


def test_signed_power_is_the_identity_on_a_symmetric_target():
    """``signed_power_y`` fits p == 1 on symmetric y."""
    t = TRANSFORMS_REGISTRY["signed_power_y"]
    assert _majority([float(t.fit(np.random.default_rng(s).normal(size=400), None).get("p")) == 1.0 for s in _SEEDS])


def test_the_stability_check_drops_a_spec_found_in_one_run_of_three(monkeypatch):
    """With n_bootstrap_runs=3 and the default 0.6 majority, a spec seen once is dropped and a spec seen twice kept."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    runs = iter([["always", "once"], ["always", "twice"], ["always", "twice"]])

    def fake_fit(self, *a, **k):
        """Each run finds the next scripted spec set."""
        self.specs_ = [types.SimpleNamespace(name=n) for n in next(runs)]
        return self

    monkeypatch.setattr(CompositeTargetDiscovery, "fit", fake_fit)
    df = pd.DataFrame({"b": np.arange(200.0), "y": np.arange(200.0)})
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True, random_state=0))
    disc.fit_with_stability_check(df, "y", ["b"], np.arange(200), n_bootstrap_runs=3)
    assert disc.stability_counts_ == {"always": 3, "once": 1, "twice": 2}, disc.stability_counts_
    assert {s.name for s in disc.specs_} == {"always", "twice"}, disc.stability_counts_


@pytest.mark.slow
def test_default_discovery_emits_no_spec_on_pure_noise():
    """A target independent of every feature yields no spec under the default config, on every one of the 10 fixed seeds.

    The honest RMSE gate compared a composite only with the raw tiny model, which overfits noise and loses to the constant
    train mean: on seed 1 ten noise specs "beat raw" by 2-3 standard errors. The constant is now part of the null.
    """
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    empty = []
    for s in _SEEDS:
        rng = np.random.default_rng(s)
        n = 600
        df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)}).assign(y=rng.normal(size=n))
        disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True, random_state=s)).fit(df, "y", [f"x{i}" for i in range(4)], np.arange(n))
        empty.append(not disc.specs_)
    # The seeds are fixed, so the result is deterministic: one noise seed shipping specs is the defect, not variance.
    assert all(empty), f"noise specs emitted on seeds {[s for s, e in zip(_SEEDS, empty) if not e]}"
