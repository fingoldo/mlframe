"""Unit: drift-aware rediscovery wiring (``check_and_rediscover`` + the ``on_drift`` hook, G5).

Synthetic drift: the deployed estimator was fit on ``y = base + 0.5*feat``; the drifted stream
shifts the base distribution and flips the residual relationship, so the monitor alarms. The
helper must then probe the prior discovery specs and, on a REDISCOVER verdict, run a full
``discovery.fit`` on the new frame automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.monitoring import CompositeDriftMonitor, check_and_rediscover
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _train_frame(n: int = 2000, seed: int = 0):
    """Stationary AR-plus-noise frame the estimator/discovery are fit on."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 100.0, n)
    feat = rng.normal(size=n)
    y = base + 5.0 * feat + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"lag": base, "feat": feat, "y": y}), y


def _drifted_frame(n: int = 2000, seed: int = 5):
    """Shifted base + inverted residual relation: both base-PSI and residual signals alarm."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(300.0, 400.0, n)
    feat = rng.normal(size=n)
    y = 0.2 * base - 5.0 * feat + rng.normal(0.0, 10.0, n)
    return pd.DataFrame({"lag": base, "feat": feat, "y": y}), y


def _fitted_pair():
    """(monitor, discovery) fitted on the stationary train frame."""
    df, y = _train_frame()
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="diff", base_column="lag")
    est.fit(df[["lag", "feat"]], y)
    monitor = CompositeDriftMonitor(est)
    monitor.ensure_sketch(reference=df[["lag", "feat"]], y_reference=y)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        random_state=0,
        screening="mi",
        base_candidates=["lag"],
        transforms=["diff", "linear_residual"],
        honest_holdout_frac=0.2,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        auto_base_null_perms=0,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, "y", ["lag", "feat"], np.arange(len(df)))
    assert disc.specs_, "sanity: discovery must find specs on the stationary train frame"
    return monitor, disc, df, y


def test_no_drift_no_rediscovery():
    """A stationary-continuation batch must not alarm the monitor and must keep the prior specs unchanged."""
    monitor, disc, _df, _y = _fitted_pair()
    prior = list(disc.specs_)
    new_df, new_y = _train_frame(seed=9)  # stationary continuation
    out = check_and_rediscover(monitor, disc, new_df, "y", ["lag", "feat"], y_new=new_y, train_idx=np.arange(len(new_df)))
    assert out["drift"] is False and out["refitted"] is False
    assert [s.name for s in out["specs"]] == [s.name for s in prior]


def test_drift_triggers_full_rediscovery():
    """A confirmed REDISCOVER verdict with ``train_idx`` given must run a full ``discovery.fit`` automatically."""
    monitor, disc, _df, _y = _fitted_pair()
    new_df, new_y = _drifted_frame()
    out = check_and_rediscover(
        monitor,
        disc,
        new_df,
        "y",
        ["lag", "feat"],
        y_new=new_y,
        train_idx=np.arange(len(new_df)),
        min_surviving_fraction=1.01,  # force the REDISCOVER verdict
    )
    assert out["drift"] is True, "shifted base + flipped residual must alarm the monitor"
    assert out["decision"] is not None and out["decision"].reuse is False
    assert out["refitted"] is True, "a REDISCOVER verdict with train_idx must run discovery.fit"
    assert out["specs"] == disc.specs_, "returned specs must be the freshly re-discovered set"


def test_drift_without_train_idx_reports_but_does_not_refit():
    """A REDISCOVER verdict with no ``train_idx`` reports the drift but leaves the prior specs untouched."""
    monitor, disc, _df, _y = _fitted_pair()
    prior = list(disc.specs_)
    new_df, new_y = _drifted_frame(seed=6)
    out = check_and_rediscover(
        monitor,
        disc,
        new_df,
        "y",
        ["lag", "feat"],
        y_new=new_y,
        train_idx=None,
        min_surviving_fraction=1.01,
    )
    assert out["drift"] is True and out["refitted"] is False
    assert [s.name for s in out["specs"]] == [s.name for s in prior], "without train_idx the prior specs are kept"


def test_on_drift_callback_fires_on_alarm_only():
    """The ``on_drift`` callback fires exactly once on an alarming batch and never on a stationary one."""
    monitor, _disc, _df, _y = _fitted_pair()
    calls: list[dict] = []
    monitor.on_drift = calls.append
    stat_df, stat_y = _train_frame(seed=11)
    monitor.monitor(stat_df[["lag", "feat"]], y=stat_y)
    assert calls == [], "stationary batch must not fire the on_drift hook"
    drift_df, drift_y = _drifted_frame(seed=12)
    report = monitor.monitor(drift_df[["lag", "feat"]], y=drift_y)
    assert report["recommend_update"] is True
    assert len(calls) == 1 and calls[0] is report, "the hook receives the drift report exactly once"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q", "--no-cov"])
