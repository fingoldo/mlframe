"""Regression: ``fit_with_stability_check`` resamples whole GROUPS when a group key is configured.

The inline stability driver (``discovery/__init__.py``) had its OWN row-level bootstrap, group-blind even on grouped data,
so a spec that only memorised per-group levels survived (a group's rows sit in both the replicate and its complement).
These tests pin that the inline path now routes through the group-aware ``_subsample_groups`` when ``group_column`` +
``stability_group_aware`` are set, and falls back to the row draw (``np.random.Generator.choice``) otherwise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import mlframe.training.composite.discovery._stability as _stab
from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _grouped_df(n: int = 900, n_groups: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), n // n_groups)
    level = rng.uniform(0.0, 40.0, n_groups)[g]
    df = pd.DataFrame({
        "y": level + 2.0 * rng.normal(size=g.size),
        "base": level + rng.normal(0.0, 0.3, g.size),
        "well": g.astype(np.int64),
        "x1": rng.normal(size=g.size),
    })
    return df, g.size


@pytest.fixture(autouse=True)
def _silence():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        yield


def _run(monkeypatch, *, group_column, group_aware):
    df, n = _grouped_df()
    calls = {"group": 0}
    real = _stab._subsample_groups

    def _spy(*a, **k):
        calls["group"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_stab, "_subsample_groups", _spy)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="mi", mi_estimator="bin", base_candidates=["base"],
        transforms=["diff"], group_column=group_column, stability_group_aware=group_aware,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit_with_stability_check(
        df, "y", ["base", "well", "x1"], np.arange(n), n_bootstrap_runs=2, subsample_fraction=0.5,
    )
    return calls["group"]


def test_inline_uses_group_resampling_when_group_column_set(monkeypatch):
    assert _run(monkeypatch, group_column="well", group_aware=True) >= 1, (
        "inline stability must draw whole groups when a group key is configured"
    )


def test_inline_falls_back_to_row_draw_without_group_column(monkeypatch):
    assert _run(monkeypatch, group_column=None, group_aware=True) == 0, (
        "no group key -> the row draw runs, not the group draw (bit-identical to the prior behaviour)"
    )


def test_inline_group_aware_toggle_off_uses_row_draw(monkeypatch):
    assert _run(monkeypatch, group_column="well", group_aware=False) == 0, (
        "stability_group_aware=False forces the legacy row draw even with a group column"
    )
