"""biz_value: the y-scale group-aware holdout gate drops composite specs that collapse on unseen groups.

Reproduces the production TVT failure in miniature: a residual spec ``T = y - alpha*base`` whose base is a
per-GROUP level feature (like ``pf_tvt_post_mean`` per well). On a group-disjoint holdout the tree model's
``T_hat`` is clamped to the train-group range while the inverse adds ``alpha*base`` for a holdout group whose
base extrapolates beyond train -- so ``y = T_hat + alpha*base`` blows up and the prediction collapses
(``R^2 << 0`` in prod). The forward-only MI screen / i.i.d. honest-holdout never sees this; the y-scale
group-aware gate must catch it and DROP the spec before any full model is trained.

Pins:
* a large-alpha residual spec on a group-level base is REJECTED by the gate;
* a unit-alpha residual spec (stable inverse) is KEPT;
* the gate is a no-op when disabled, and a no-op without group ids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery._yscale_holdout_gate import apply_yscale_holdout_gate
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _grouped_frame(n_groups: int = 10, per: int = 150, seed: int = 0):
    """y = base + 3*x1 + noise, where ``base`` is a strong per-GROUP level (range ~100..1000).

    Holdout groups whose level lands at the extremes sit OUTSIDE the train-group base range -- the
    tree-extrapolation regime where a high-alpha inverse blows up.
    """
    rng = np.random.default_rng(seed)
    levels = rng.uniform(100.0, 1000.0, n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    base = levels[groups] + rng.normal(0.0, 5.0, groups.size)
    x1 = rng.normal(size=groups.size)
    x2 = rng.normal(size=groups.size)
    y = base + 3.0 * x1 + rng.normal(0.0, 5.0, groups.size)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y.astype(np.float64)})
    return df, groups.astype(np.int64), y.astype(np.float64)


def _spec(name: str, alpha: float, beta: float = 0.0) -> CompositeSpec:
    return CompositeSpec(
        name=name, target_col="y", transform_name="linear_residual", base_column="base",
        fitted_params={"alpha": float(alpha), "beta": float(beta)},
        mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0, n_train_rows=100,
    )


def _make_gate_ctx(group_ids):
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, yscale_holdout_gate_enabled=True,
        yscale_holdout_gate_sample_n=2_000, yscale_holdout_gate_min_groups=4,
        tiny_model_n_estimators=25,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = group_ids
    return disc


def test_biz_val_yscale_gate_drops_collapsing_high_alpha_spec():
    df, groups, y = _grouped_frame()
    disc = _make_gate_ctx(groups)
    bad = _spec("y-linres-base-BADalpha", alpha=50.0)
    good = _spec("y-linres-base-unit", alpha=1.0)
    train_idx = np.arange(len(df))

    survivors = apply_yscale_holdout_gate(
        disc, df, "y", [bad, good], ["base", "x1", "x2"], train_idx, y,
    )
    names = {s.name for s in survivors}
    assert "y-linres-base-BADalpha" not in names, "high-alpha group-collapsing spec must be dropped"
    assert "y-linres-base-unit" in names, "stable unit-alpha spec must survive"


def test_biz_val_yscale_gate_noop_when_disabled():
    df, groups, y = _grouped_frame()
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, yscale_holdout_gate_enabled=False, tiny_model_n_estimators=25,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = groups
    bad = _spec("y-linres-base-BADalpha", alpha=50.0)
    out = apply_yscale_holdout_gate(disc, df, "y", [bad], ["base", "x1", "x2"], np.arange(len(df)), y)
    assert [s.name for s in out] == ["y-linres-base-BADalpha"], "disabled gate must keep every spec"


def test_biz_val_yscale_gate_noop_without_group_ids():
    df, _groups, y = _grouped_frame()
    disc = _make_gate_ctx(None)  # no group ids -> nothing to validate
    bad = _spec("y-linres-base-BADalpha", alpha=50.0)
    out = apply_yscale_holdout_gate(disc, df, "y", [bad], ["base", "x1", "x2"], np.arange(len(df)), y)
    assert [s.name for s in out] == ["y-linres-base-BADalpha"], "no-group gate must keep every spec"
