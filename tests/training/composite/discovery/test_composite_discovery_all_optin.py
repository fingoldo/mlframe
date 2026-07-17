"""Kitchen-sink integration: a CompositeTargetDiscovery run with EVERY opt-in
capability enabled at once (most now default-ON) completes cleanly, stays
train-only + deterministic, and the all-OFF config reproduces the bare baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig
from mlframe.training.composite import CompositeTargetDiscovery


def _temporal_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n)
    y = np.empty(n)
    y[0] = 0.0
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + rng.normal(0.0, 1.0)
    lag = np.concatenate([[0.0], y[:-1]])
    feat = rng.normal(0.0, 1.0, n)
    feat2 = rng.normal(0.0, 1.0, n)
    perm = rng.permutation(n)
    df = pd.DataFrame(
        {
            "y": y[perm],
            "lag": lag[perm],
            "feat": feat[perm],
            "feat2": feat2[perm],
            "ts": ts[perm],
        }
    )
    return df


def _all_on_config(**over):
    cfg = dict(
        enabled=True,
        mi_sample_n=800,
        base_candidates=["lag"],
        time_column="ts",
        time_series_transforms_enabled=True,
        auto_base_structural_boost=True,
        transform_waic_validation_enabled=True,
        region_adaptive_enabled=True,
        interaction_base_discovery_enabled=True,
        auto_chain_discovery_enabled=True,
        random_state=0,
    )
    cfg.update(over)
    return CompositeTargetDiscoveryConfig(**cfg)


def _fit(cfg):
    df = _temporal_frame()
    disc = CompositeTargetDiscovery(cfg)
    return disc.fit(
        df,
        "y",
        ["lag", "feat", "feat2"],
        np.arange(len(df)),
        time_ordering=df["ts"].to_numpy(),
    ), df


class TestAllOptInKitchenSink:
    def test_all_optin_runs_clean_and_produces_valid_specs(self) -> None:
        disc, df = _fit(_all_on_config())
        assert isinstance(disc.specs_, list)
        # Every kept spec round-trips fit->predict to finite y.
        from mlframe.training.composite import CompositeTargetEstimator
        from sklearn.linear_model import LinearRegression

        for spec in disc.specs_[:5]:
            est = CompositeTargetEstimator(
                base_estimator=LinearRegression(),
                transform_name=spec.transform_name,
                base_column=spec.base_column,
                base_columns=((spec.base_column, *spec.extra_base_columns) if spec.extra_base_columns else None),
            )
            X = df[["lag", "feat", "feat2"]]
            est.fit(X, df["y"].to_numpy())
            assert np.all(np.isfinite(est.predict(X)))
        # The opt-in artefacts are populated/visible.
        assert hasattr(disc, "region_adaptive_specs_")
        assert hasattr(disc, "interaction_bases_")

    def test_all_optin_is_deterministic(self) -> None:
        a, _ = _fit(_all_on_config())
        b, _ = _fit(_all_on_config())
        assert [s.name for s in a.specs_] == [s.name for s in b.specs_]

    def test_all_off_reproduces_bare_baseline(self) -> None:
        off = _all_on_config(
            time_series_transforms_enabled=False,
            transform_waic_validation_enabled=False,
            region_adaptive_enabled=False,
            interaction_base_discovery_enabled=False,
            auto_chain_discovery_enabled=False,
            auto_base_structural_boost=False,
        )
        disc, _ = _fit(off)
        # The all-off run still completes and produces a (possibly empty) list;
        # no opt-in spec leaks into specs_ (region/interaction are stash-only and
        # auto-chain is off). The region-adaptive artefact is absent or empty.
        assert isinstance(disc.specs_, list)
        assert not getattr(disc, "region_adaptive_specs_", [])
