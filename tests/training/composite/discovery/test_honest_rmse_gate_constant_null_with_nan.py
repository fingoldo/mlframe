"""A NaN among the fit rows must not switch off the honest gate's constant-null check.

The null a spec has to beat is the constant train mean. It was computed with ``np.mean``, so a single NaN target on the
fit rows made it NaN, and the ``np.isfinite(const_rmse) and ...`` guard then skipped the check for every spec: on a
signal-free target a composite could pass by beating an overfitting raw model while predicting nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_rmse_gate import apply_honest_rmse_gate
from mlframe.training.composite.spec import CompositeSpec
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _signal_free(n: int = 1500, seed: int = 3):
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 10.0, n)
    df = pd.DataFrame({"base": base, "x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = rng.normal(0.0, 1.0, n)
    df["y"] = y
    return df, y.astype(np.float64)


def _run(y: np.ndarray, df: pd.DataFrame):
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, screening="mi", base_candidates=["base"], honest_holdout_frac=0.2, tiny_model_n_estimators=40,
        multi_base_enabled=False, interaction_base_discovery_enabled=False, auto_chain_discovery_enabled=False, auto_base_null_perms=0,
    )
    disc = CompositeTargetDiscovery(cfg)
    spec = CompositeSpec(name="y-linres-base", target_col="y", transform_name="linear_residual", base_column="base",
                         fitted_params={"alpha": 0.0, "beta": 0.0}, mi_gain=1.0, mi_y=0.0, mi_t=1.0,
                         valid_domain_frac=1.0, n_train_rows=1000)
    out = apply_honest_rmse_gate(disc, df, "y", [spec], ["base", "x0", "x1"], np.arange(1000), np.arange(1000, 1500), y)
    return out, disc


def test_constant_null_still_applies_with_a_nan_fit_target():
    df, y = _signal_free()
    y_nan = y.copy()
    y_nan[5] = np.nan
    out_clean, _ = _run(y, df)
    out_nan, disc = _run(y_nan, df)
    assert out_clean == [], "setup: on a signal-free target the constant null must reject the spec"
    assert out_nan == [], "one NaN fit target switched the constant-null check off"
    assert any("constant" in str(row.get("reason", "")) for row in disc.rejection_ledger)
