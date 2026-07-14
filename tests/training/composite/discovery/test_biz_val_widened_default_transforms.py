"""biz_value: ``gaussian_copula_residual`` (an opt-in, explicit-``transforms=[...]`` candidate, NOT
in the production default pool -- see the REJECTED note in ``_composite_target_discovery_config_base.py``)
finds real lift a narrow old-style pool misses, when explicitly requested.

DGP: ``y = exp(1.1*base + 0.15*x0 + N(0, 0.5))`` -- a monotone but heavy-tailed / distorted-marginal
dependence on ``base``. ``gaussian_copula_residual`` maps both y and base through their train
empirical CDFs into normal-scores space before the OLS fit, which handles the marginal distortion
better than a narrow old-style pool (``diff`` / ``additive_residual`` / ``linear_residual`` /
``monotonic_residual`` / ``rank_residual``) -- measured best-old honest y-RMSE 36.308
(monotonic_residual) vs best-new honest y-RMSE 35.819 (gaussian_copula_residual), a real 1.35% gain
surviving the honest-holdout gate (G2), not just an in-sample MI artifact.

IMPORTANT scope note (see ``discovery/_benchmarks/bench_widened_default_transforms.py`` for the full
measurement): this 1.35% win is against a deliberately NARROW 5-transform comparison list, not
against the real production default (29 transforms). The full-pool A/B on this same DGP shows
``quantile_normal_y`` (already in the default pool) ties/beats ``gaussian_copula_residual`` --
0.00% net improvement from adding these 4 transforms to the default pool, at +1433% wall-clock cost.
This test therefore validates that the TRANSFORM has real value when explicitly opted into (e.g. a
caller who suspects marginal distortion and wants to try it directly), not that default-pool
membership is justified -- that was measured and rejected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_NARROW_OLD_TRANSFORMS = ["diff", "additive_residual", "linear_residual", "monotonic_residual", "rank_residual"]
_NEW_EXTRA_TRANSFORMS = ["box_cox_y", "seasonal_residual", "nadaraya_watson_residual", "gaussian_copula_residual"]


def _lognormal_frame(n: int = 4000, seed: int = 11):
    """Monotone but heavy-tailed/distorted-marginal dependence on ``base``."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 5.0, n)
    x0 = rng.normal(size=n)
    y = np.exp(1.1 * base + 0.15 * x0 + rng.normal(0.0, 0.5, n))
    df = pd.DataFrame({"base": base, "x0": x0, "noise0": rng.normal(size=n), "y": y})
    return df, y.astype(np.float64)


def _config(**overrides) -> CompositeTargetDiscoveryConfig:
    """Minimal fast discovery config for the widened-transforms tests, with ``**overrides`` applied."""
    kw = dict(
        enabled=True, random_state=0, screening="mi", base_candidates=["base", "x0"],
        honest_holdout_frac=0.2, tiny_model_n_estimators=40, auto_base_null_perms=0,
        multi_base_enabled=False, interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
    )
    kw.update(overrides)
    return CompositeTargetDiscoveryConfig(**kw)


def test_biz_val_widened_transforms_gaussian_copula_beats_narrow_old_list():
    """gaussian_copula_residual survives G2 and beats every narrow-old-list spec's honest y-RMSE."""
    df, _y = _lognormal_frame()
    feats = ["base", "x0", "noise0"]

    disc_old = CompositeTargetDiscovery(_config(transforms=list(_NARROW_OLD_TRANSFORMS)))
    disc_old.fit(df, "y", feats, np.arange(len(df)))
    old_rmses = [s.honest_holdout_rmse for s in disc_old.specs_ if s.honest_holdout_rmse is not None]
    assert old_rmses, "narrow-old-list run must produce at least one surviving spec"
    best_old_rmse = min(old_rmses)

    disc_new = CompositeTargetDiscovery(_config(transforms=list(_NARROW_OLD_TRANSFORMS) + list(_NEW_EXTRA_TRANSFORMS)))
    disc_new.fit(df, "y", feats, np.arange(len(df)))
    gc_specs = [s for s in disc_new.specs_ if s.transform_name == "gaussian_copula_residual"]
    assert gc_specs, "gaussian_copula_residual must survive the honest-holdout gate (G2) on this DGP"
    gc_rmse = min(s.honest_holdout_rmse for s in gc_specs if s.honest_holdout_rmse is not None)

    assert gc_rmse < best_old_rmse, (
        f"gaussian_copula_residual (honest y-RMSE={gc_rmse}) must beat the best narrow-old-list "
        f"spec (honest y-RMSE={best_old_rmse}) -- the widened pool must find comparable lift the "
        f"old-style narrow list misses"
    )
    new_best = min(s.honest_holdout_rmse for s in disc_new.specs_ if s.honest_holdout_rmse is not None)
    assert new_best == pytest.approx(gc_rmse), "gaussian_copula_residual must be the overall best spec in the widened run"


def _wide_range_linear_frame(n: int = 3000, seed: int = 1):
    """Plain wide-range linear DGP with NO periodic structure. ``seasonal_residual`` is
    ``requires_base=False`` (unary, phase-in-batch only) so it ignores the dominant ``base``
    signal entirely on this DGP -- a real, measured catastrophic underfit, not a synthetic strawman."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    y = 2.5 * base + 10.0 + rng.normal(0.0, 5.0, n)
    df = pd.DataFrame({"base": base, "x0": rng.normal(size=n), "noise0": rng.normal(size=n), "y": y})
    return df


def test_biz_val_widened_transforms_g2_still_rejects_seasonal_when_irrelevant():
    """seasonal_residual is present in the candidate pool but the honest-holdout gate (G2) rejects it
    on a DGP with no periodic structure -- pins that widening the pool does not let an irrelevant
    NEW transform (which ignores the dominant base signal since it is unary/requires_base=False) ship.
    Measured: honest y-RMSE=122.6 vs raw y-RMSE=12.96 (``bench_widened_default_transforms.py``)."""
    df = _wide_range_linear_frame()
    feats = ["base", "x0", "noise0"]
    disc = CompositeTargetDiscovery(_config(transforms=["seasonal_residual"], base_candidates=["base"]))
    disc.fit(df, "y", feats, np.arange(len(df)))
    assert not disc.specs_, "seasonal_residual must be rejected by the honest RMSE gate when it ignores the true (base-driven) structure"
    stages = {row["stage"] for row in disc.rejection_ledger}
    assert "honest_rmse" in stages


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q", "--no-cov"])
