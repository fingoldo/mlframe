"""Unit: knn-MI budget guard (auto-downgrade knn -> bin when the estimated sweep cost blows the budget, G8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._knn_budget import estimate_knn_sweep_seconds
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(n: int = 1500, seed: int = 0):
    """Small additive-signal frame for the knn-budget-guard tests."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    x0 = rng.normal(size=n)
    y = base + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"base": base, "x0": x0, "x1": rng.normal(size=n), "y": y})


def _cfg(**kw) -> CompositeTargetDiscoveryConfig:
    """Minimal screening="mi" / mi_estimator="knn" discovery config for the budget-guard tests."""
    base = dict(
        enabled=True,
        random_state=0,
        screening="mi",
        mi_estimator="knn",
        base_candidates=["base"],
        transforms=["linear_residual"],
        honest_holdout_frac=None,
        honest_rmse_gate_enabled=False,
        auto_base_null_perms=0,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
    )
    base.update(kw)
    return CompositeTargetDiscoveryConfig(**base)


def test_estimate_formula():
    """``estimate_knn_sweep_seconds`` is per_col_seconds * n_cols * (n_work_items + 1); zero columns gives zero."""
    # one sweep per work item plus the upfront per-feature y baseline.
    assert estimate_knn_sweep_seconds(0.5, 10, 7) == pytest.approx(0.5 * 10 * 8)
    assert estimate_knn_sweep_seconds(0.5, 0, 7) == 0.0


def test_downgrade_fires_on_tiny_budget():
    """A near-zero budget must downgrade the per-fit config to bin without mutating the caller's config."""
    df = _frame()
    caller_cfg = _cfg(knn_mi_budget_seconds=1e-9)
    disc = CompositeTargetDiscovery(caller_cfg)
    disc.fit(df, "y", ["base", "x0", "x1"], np.arange(len(df)))
    assert disc.config.mi_estimator == "bin", "the per-fit config must be downgraded to the bin estimator"
    assert caller_cfg.mi_estimator == "knn", "the caller's shared config object must never be mutated"
    assert disc.specs_, "discovery must still find the linear_residual spec on the bin estimator"


def test_no_downgrade_when_budget_ample_or_disabled():
    """An ample budget, and ``knn_mi_auto_downgrade=False`` even under a tiny budget, must keep knn."""
    df = _frame()
    disc = CompositeTargetDiscovery(_cfg(knn_mi_budget_seconds=1e9))
    disc.fit(df, "y", ["base", "x0", "x1"], np.arange(len(df)))
    assert disc.config.mi_estimator == "knn", "an ample budget must keep the configured knn estimator"

    disc2 = CompositeTargetDiscovery(_cfg(knn_mi_auto_downgrade=False, knn_mi_budget_seconds=1e-9))
    disc2.fit(df, "y", ["base", "x0", "x1"], np.arange(len(df)))
    assert disc2.config.mi_estimator == "knn", "auto_downgrade=False must be a hard opt-out"


def test_the_guard_covers_auto_base_ranking(monkeypatch):
    """With auto bases and the permutation null, a tiny budget downgrades before auto-base runs any Kraskov estimate."""
    import sklearn.feature_selection as fs

    calls = {"n": 0}
    at_auto_base = {"n": None}
    real_mi, real_auto = fs.mutual_info_regression, CompositeTargetDiscovery._auto_base

    def mi_spy(*a, **k):
        """Count every MI call so the budget can be checked against what was actually spent."""
        calls["n"] += 1
        return real_mi(*a, **k)

    def auto_spy(self, *a, **k):
        """Record the MI tally at the moment base ranking starts, which is the point the budget applies to."""
        at_auto_base["n"] = calls["n"]
        return real_auto(self, *a, **k)

    monkeypatch.setattr(fs, "mutual_info_regression", mi_spy)
    monkeypatch.setattr(CompositeTargetDiscovery, "_auto_base", auto_spy)
    disc = CompositeTargetDiscovery(_cfg(knn_mi_budget_seconds=1e-9, base_candidates="auto", auto_base_null_perms=20))
    disc.fit(_frame(), "y", ["base", "x0", "x1"], np.arange(1500))
    assert disc.config.mi_estimator == "bin"
    assert at_auto_base["n"] is not None, "auto-base ranking did not run"
    assert calls["n"] == at_auto_base["n"], f"auto-base ran {calls['n'] - at_auto_base['n']} Kraskov estimates the guard should have prevented"


def test_the_estimate_counts_the_auto_base_sweeps():
    """21 auto-base sweeps (1 + 20 null draws) are added to the per-work-item sweeps and the y baseline."""
    assert estimate_knn_sweep_seconds(0.5, 10, 7, 21) == pytest.approx(0.5 * 10 * (8 + 21))
