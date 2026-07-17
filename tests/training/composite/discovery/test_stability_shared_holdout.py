"""Unit: ``fit_with_stability_check`` carves the honest holdout ONCE and shares it across replicates (G6).

Pre-fix each replicate's ``fit`` carved its own 20% holdout from that run's subsample, so every
replicate's holdout landed inside other replicates' screening pools -- no row set stayed
"never touched" sweep-wide. Pins:

* every replicate reuses the SAME holdout index set;
* every replicate's screening pool is disjoint from that shared holdout;
* the shared-holdout marker is cleared after the sweep (standalone fits carve normally again);
* stability selection still returns the stable spec set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery import _honest_holdout as hh
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(n: int = 1200, seed: int = 0):
    """Small additive-signal frame for the shared-holdout stability sweep test."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    x0 = rng.normal(size=n)
    y = base + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"base": base, "x0": x0, "x1": rng.normal(size=n), "y": y})


def _cfg() -> CompositeTargetDiscoveryConfig:
    """screening="mi" discovery config used by the shared-holdout stability sweep test."""
    return CompositeTargetDiscoveryConfig(
        enabled=True,
        random_state=0,
        screening="mi",
        base_candidates=["base"],
        transforms=["linear_residual"],
        honest_holdout_frac=0.2,
        auto_base_null_perms=0,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
    )


def test_stability_sweep_shares_one_holdout(monkeypatch):
    """Every replicate in a stability sweep reuses the SAME honest holdout, never overlapping its own screen pool."""
    df = _frame()
    carves: list[tuple[np.ndarray, np.ndarray | None]] = []
    real_carve = hh.carve_screening_holdout

    def _spy(self, train_idx):
        """Wrap ``carve_screening_holdout`` to record every (screen_idx, holdout_idx) call it produces."""
        out = real_carve(self, train_idx)
        carves.append(out)
        return out

    monkeypatch.setattr(hh, "carve_screening_holdout", _spy)
    disc = CompositeTargetDiscovery(_cfg())
    disc.fit_with_stability_check(df, "y", ["base", "x0", "x1"], np.arange(len(df)), n_bootstrap_runs=3)

    assert len(carves) == 3, "one carve call per replicate (via the shared-holdout fast path)"
    holdouts = [h for _s, h in carves]
    assert all(h is not None for h in holdouts), "the shared honest holdout must be non-empty"
    ref = holdouts[0]
    for h in holdouts[1:]:
        assert np.array_equal(ref, h), "every replicate must reuse the SAME holdout indices"
    for screen, h in carves:
        assert np.intersect1d(screen, h).size == 0, "replicate screening pools must never touch the shared holdout"

    assert getattr(disc, "_stability_shared_holdout_idx", "unset") is None, "marker must be cleared after the sweep"
    assert disc.stability_counts_, "the sweep must still produce stability counts"
    assert disc.specs_, "the stable linear_residual spec must survive on this clean AR frame"

    # Standalone fit after the sweep carves its own holdout again (marker cleared).
    carves.clear()
    disc.fit(df, "y", ["base", "x0", "x1"], np.arange(len(df)))
    assert len(carves) == 1 and carves[0][1] is not None
