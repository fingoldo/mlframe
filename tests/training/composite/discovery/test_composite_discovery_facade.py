"""Sensor: composite_discovery method-rebinding preserves identity + class invariants."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.training.composite import discovery as parent
from mlframe.training.composite.discovery import _filter as filter_mod
from mlframe.training.composite.discovery import _stacked as stacked_mod
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_w12b_composite_discovery_methods_rebound():
    cls = parent.CompositeTargetDiscovery
    assert cls._filter_features is filter_mod._filter_features
    assert cls.fit_stacked is stacked_mod.fit_stacked
    assert cls.fit_stacked_on_residual is stacked_mod.fit_stacked_on_residual


def test_w12b_composite_discovery_facade_under_budget():
    facade_loc = sum(1 for _ in Path(parent.__file__).open(encoding="utf-8"))
    assert facade_loc < 750, f"composite_discovery.py LOC={facade_loc} exceeds 750 budget"


def test_w12b_composite_discovery_class_identity_preserved():
    cfg = CompositeTargetDiscoveryConfig()
    disc = parent.CompositeTargetDiscovery(cfg)
    assert isinstance(disc, parent.CompositeTargetDiscovery)


def test_w12b_composite_discovery_smoke_fit_runs():
    cfg = CompositeTargetDiscoveryConfig()
    disc = parent.CompositeTargetDiscovery(cfg)
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    y = base * 2.0 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"base": base, "noise": rng.normal(size=n), "y": y})
    train_idx = np.arange(n)
    disc.fit(df, "y", ["base", "noise"], train_idx)
    # Smoke: discovery ran without raising, populated specs_ (may be empty
    # if no transforms beat the eps gates on this fixture).
    assert hasattr(disc, "specs_")
