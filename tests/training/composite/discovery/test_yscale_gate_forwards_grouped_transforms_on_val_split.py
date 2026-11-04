"""A grouped transform is forwarded with its training-frame group labels on the y-scale gate's val-split path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery._yscale_holdout_gate import apply_yscale_holdout_gate
from mlframe.training.composite.transforms import get_transform
from mlframe.training.composite.transforms._call_gateway import call_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(rng, group_ids, per=150):
    """Group-level base levels with a small row-level signal; one frame per group range."""
    g = np.repeat(group_ids, per)
    base = 100.0 + 90.0 * g + rng.normal(0.0, 5.0, g.size)
    x1 = rng.normal(size=g.size)
    y = base + 3.0 * x1 + rng.normal(0.0, 5.0, g.size)
    return pd.DataFrame({"base": base, "x1": x1, "x2": rng.normal(size=g.size), "grp": g, "y": y}), g, y


def test_stable_grouped_spec_survives_the_val_split_gate():
    """It used to be rejected with 'requires groups for forward' because the fit side was called with groups=None."""
    rng = np.random.default_rng(3)
    train, g_train, y_train = _frame(rng, np.arange(8))
    val, _g_val, y_val = _frame(rng, np.arange(8, 12))
    transform = get_transform("linear_residual_grouped")
    params = call_transform(transform, "fit", y_train, train["base"].to_numpy(), groups=g_train)
    spec = CompositeSpec(
        name="y-linresgrp", target_col="y", transform_name="linear_residual_grouped", base_column="base", fitted_params=params,
        mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0, n_train_rows=100,
    )
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, yscale_holdout_gate_enabled=True, yscale_holdout_gate_sample_n=2000, yscale_holdout_gate_min_groups=4,
        tiny_model_n_estimators=25, group_column="grp",
    )
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = g_train
    survivors = apply_yscale_holdout_gate(disc, train, "y", [spec], ["base", "x1", "x2"], np.arange(len(train)), y_train, val_df=val, val_y=y_val)
    assert [s.name for s in survivors] == ["y-linresgrp"]
