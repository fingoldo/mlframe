"""biz_value: the honest-holdout OOS RMSE gate rejects MI-inflated-but-harmful composite specs.

MI is monotone-invariant and bias-inflated: on the additive DGP ``y = base + 30*x0 + N(0,1)`` with
``base ~ U(0, 1000)``, the ``ratio`` transform ``T = y / base = 1 + (30*x0 + noise) / base`` divides
the signal by a base that gets arbitrarily small, so T's tail rows explode and the tiny T-model
learns ~nothing -- yet mi_gain is strongly POSITIVE (measured +0.556: MI(y, x0) is drowned by the
base's dominance while T still exposes x0). The inverse ``y = T_hat * base`` then re-amplifies the
mis-fit T (measured honest y-RMSE 131.7 vs raw-y tiny baseline 10.5). The ``screening="mi"`` path
has no other OOS predictive gate, so pre-fix this spec shipped.

Rejected scenario note (measure-first): the multiplicative DGP ``y = base*(1+0.2*x0+0.1*x1)+N(0,1)``
with ``base = exp(N(0,1.5))`` was tried first as the harmful case, but there ratio is GENUINELY
helpful OOS (measured spec RMSE 1.21 vs raw 2.60 -- the raw tiny tree cannot fit the multiplicative
coupling), so it is not a valid trap fixture; the additive frame above is.

Pins:
* the harmful ratio pair (positive in-screen mi_gain) is REJECTED by the honest RMSE gate;
* a genuinely helpful ``linear_residual`` on the same dominant additive base SURVIVES and carries a
  positive ``honest_holdout_rmse_gain``;
* the gate is a no-op when disabled and when the honest holdout is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_rmse_gate import apply_honest_rmse_gate
from mlframe.training.composite.spec import CompositeSpec
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _additive_dominant_frame(n: int = 3000, seed: int = 1):
    """Additive DGP where the residual composite genuinely beats a raw-y tiny model (wide-range base)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    x0 = rng.normal(size=n)
    y = base + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"base": base, "x0": x0, "x1": rng.normal(size=n), "y": y})
    return df, y.astype(np.float64)


def _mi_config(**overrides) -> CompositeTargetDiscoveryConfig:
    """screening='mi' config (the path that previously had NO OOS predictive gate)."""
    kw = dict(
        enabled=True,
        random_state=0,
        screening="mi",
        base_candidates=["base"],
        honest_holdout_frac=0.2,
        tiny_model_n_estimators=40,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        auto_base_null_perms=0,
    )
    kw.update(overrides)
    return CompositeTargetDiscoveryConfig(**kw)


def test_biz_val_honest_rmse_gate_rejects_mi_positive_ratio_pair():
    """The MI-positive ratio spec ships with the gate off and is rejected (via the honest_rmse ledger stage) with it on."""
    df, _y = _additive_dominant_frame()
    train_idx = np.arange(len(df))

    # Control run, gate OFF: the ratio pair clears the MI screen with a POSITIVE mi_gain and ships.
    disc_off = CompositeTargetDiscovery(_mi_config(transforms=["ratio"], honest_rmse_gate_enabled=False))
    disc_off.fit(df, "y", ["base", "x0", "x1"], train_idx)
    ratio_off = [s for s in disc_off.specs_ if s.transform_name == "ratio"]
    assert ratio_off, "control: ratio spec must pass the MI screen when the RMSE gate is off"
    assert ratio_off[0].mi_gain > 0.0, "the ratio pair must be MI-POSITIVE (that is the trap the gate closes)"

    # Gate ON (default): the same MI-positive ratio pair is rejected on honest y-scale OOS RMSE.
    disc_on = CompositeTargetDiscovery(_mi_config(transforms=["ratio"]))
    disc_on.fit(df, "y", ["base", "x0", "x1"], train_idx)
    assert not [
        s for s in disc_on.specs_ if s.transform_name == "ratio"
    ], "honest RMSE gate must reject the noise-amplifying ratio spec on the mi screening path"
    stages = {row["stage"] for row in disc_on.rejection_ledger}
    assert "honest_rmse" in stages, "the rejection must be attributed to the honest_rmse ledger stage"


def test_biz_val_honest_rmse_gate_keeps_genuinely_helpful_linear_residual():
    """A linear_residual spec that genuinely beats raw y OOS survives with a positive ``honest_holdout_rmse_gain``."""
    df, _y = _additive_dominant_frame()
    disc = CompositeTargetDiscovery(_mi_config(transforms=["linear_residual"]))
    disc.fit(df, "y", ["base", "x0", "x1"], np.arange(len(df)))
    kept = [s for s in disc.specs_ if s.transform_name == "linear_residual"]
    assert kept, "helpful linear_residual on the dominant additive base must survive the gate"
    spec = kept[0]
    assert spec.honest_holdout_rmse is not None and spec.honest_holdout_raw_rmse is not None
    assert (
        spec.honest_holdout_rmse_gain is not None and spec.honest_holdout_rmse_gain > 0.0
    ), f"the surviving composite must beat raw y OOS (gain={spec.honest_holdout_rmse_gain})"
    # The wide-range base defeats the tiny tree on raw y; the residual composite should win by a wide margin.
    assert spec.honest_holdout_rmse < 0.5 * spec.honest_holdout_raw_rmse


# ---------------------------------------------------------------------------
# Unit: no-op paths of the gate helper.
# ---------------------------------------------------------------------------


def _dummy_spec(alpha: float = 50.0) -> CompositeSpec:
    """A bare linear_residual spec on ``base`` with the given alpha, for the direct-helper-call tests."""
    return CompositeSpec(
        name="y-linres-base",
        target_col="y",
        transform_name="linear_residual",
        base_column="base",
        fitted_params={"alpha": float(alpha), "beta": 0.0},
        mi_gain=1.0,
        mi_y=0.0,
        mi_t=1.0,
        valid_domain_frac=1.0,
        n_train_rows=100,
    )


def test_honest_rmse_gate_noop_when_disabled():
    """``honest_rmse_gate_enabled=False`` must keep every spec untouched."""
    df, y = _additive_dominant_frame(n=500)
    disc = CompositeTargetDiscovery(_mi_config(honest_rmse_gate_enabled=False))
    spec = _dummy_spec()
    out = apply_honest_rmse_gate(
        disc,
        df,
        "y",
        [spec],
        ["base", "x0", "x1"],
        np.arange(400),
        np.arange(400, 500),
        y,
    )
    assert out == [spec], "disabled gate must keep every spec untouched"


def test_honest_rmse_gate_noop_without_holdout():
    """The gate must no-op (keep every spec) when the honest holdout index is ``None``."""
    df, y = _additive_dominant_frame(n=500)
    disc = CompositeTargetDiscovery(_mi_config())
    spec = _dummy_spec()
    out = apply_honest_rmse_gate(disc, df, "y", [spec], ["base", "x0", "x1"], np.arange(500), None, y)
    assert out == [spec], "gate must no-op when the honest holdout is absent"


def test_honest_rmse_gate_drops_harmful_spec_directly():
    """Direct helper call: a wildly mis-fit linear_residual (alpha=50 on an alpha~1 DGP) must be dropped."""
    df, y = _additive_dominant_frame(n=1500, seed=7)
    disc = CompositeTargetDiscovery(_mi_config(tiny_model_n_estimators=25))
    bad = _dummy_spec(alpha=50.0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    out = apply_honest_rmse_gate(
        disc,
        df,
        "y",
        [bad],
        ["base", "x0", "x1"],
        np.sort(perm[:1200]),
        np.sort(perm[1200:]),
        y,
    )
    assert out == [], "alpha=50 mis-fit residual must fail the honest y-scale RMSE floor"
    assert any(r["stage"] == "honest_rmse" for r in disc.rejection_ledger)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q", "--no-cov"])
