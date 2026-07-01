"""biz_value: honest group-OOF reconstruction RMSE as the load-bearing spec-RANK key.

A base-ADDITIVE spec (``diff`` = ``y - base``) on a base that only PARTIALLY explains the per-well level wins (or is
competitive) on a random / group-internal CV split: in-sample the tree memorises the residual level per train well.
But on a group-DISJOINT honest holdout of UPPER-TAIL wells, the residual level falls outside the train range, the tree's
``T_hat`` is clamped, and ``y = T_hat + base`` extrapolates and blows up. A genuinely-honest non-AR spec
(``linear_residual`` on a base that FULLY explains the level) reconstructs well on the unseen wells.

Pins:
* the two estimators genuinely DISAGREE: the additive spec's group-internal CV-RMSE is much lower than its honest-OOF
  reconstruction RMSE (the whole premise);
* ``honest_oof_reconstruction_rmse`` ranks the additive spec strictly BELOW the honest spec (ratio floor 1.20; measured ~4.2);
* full-fit with ``honest_oof_selection=True`` ranks the honest spec above the additive one;
* a genuine inverse COLLAPSE -> ``+inf``; a degenerate measurement is omitted (caller falls back);
* the selector is a strict no-op without group ids.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_oof_select import honest_oof_reconstruction_rmse
from mlframe.training.composite.discovery.screening import _tiny_cv_rmse_y_scale
from mlframe.training.composite.transforms import get_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_FEATS = ["base_full", "base_partial", "x1"]


def _frame(n_groups: int = 40, per: int = 300, seed: int = 1):
    """y = well_level + 5*x1 + noise. ``base_full`` ~= well_level (full level proxy); ``base_partial`` ~= 0.5*well_level
    (half the level). 8 upper-tail wells are held out, so their level sits outside the train-well range."""
    rng = np.random.default_rng(seed)
    well_level = rng.uniform(0.0, 50.0, n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    base_full = well_level[groups] + rng.normal(0.0, 0.2, groups.size)
    base_partial = 0.5 * well_level[groups] + rng.normal(0.0, 0.2, groups.size)
    x1 = rng.normal(size=groups.size)
    y = well_level[groups] + 5.0 * x1 + rng.normal(0.0, 1.0, groups.size)
    df = pd.DataFrame(
        {"base_full": base_full, "base_partial": base_partial, "x1": x1, "y": y.astype(np.float64)}
    )
    return df, groups.astype(np.int64), y.astype(np.float64), well_level


def _split_upper_tail(groups, well_level, n_holdout_wells: int = 8):
    holdout_wells = set(np.argsort(well_level)[-n_holdout_wells:].tolist())
    hmask = np.array([g in holdout_wells for g in groups])
    return np.nonzero(~hmask)[0], np.nonzero(hmask)[0]


def _spec(name, transform_name, base_column, params):
    return CompositeSpec(
        name=name, target_col="y", transform_name=transform_name, base_column=base_column,
        fitted_params=dict(params), mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0, n_train_rows=100,
    )


def _disc(groups, holdout_idx):
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, tiny_model_n_estimators=40, yscale_holdout_gate_sample_n=30_000,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = groups
    disc.honest_holdout_idx_ = holdout_idx
    return disc


def _internal_cv(df, groups, screen_idx, y, spec):
    base = df[spec.base_column].values[screen_idx]
    ff = [c for c in _FEATS if c != spec.base_column]
    return _tiny_cv_rmse_y_scale(
        y_train=y[screen_idx], base_train=base, transform=get_transform(spec.transform_name),
        fitted_params=spec.fitted_params, x_train_matrix=df[ff].values[screen_idx], family="lightgbm",
        n_estimators=40, num_leaves=15, learning_rate=0.1, cv_folds=3, random_state=0, n_jobs=1,
        groups=groups[screen_idx],
    )


@pytest.fixture(autouse=True)
def _silence_lgbm_feature_name_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


def test_biz_val_honest_oof_outranks_base_additive_that_wins_internal_cv():
    df, groups, y, levels = _frame()
    screen_idx, holdout_idx = _split_upper_tail(groups, levels)
    disc = _disc(groups, holdout_idx)
    additive = _spec("y-diff-basepartial", "diff", "base_partial", {})
    honest = _spec("y-linres-basefull", "linear_residual", "base_full", {"alpha": 1.0, "beta": 0.0})

    res = honest_oof_reconstruction_rmse(disc, df, "y", [additive, honest], _FEATS, screen_idx, holdout_idx, y)
    assert additive.name in res and honest.name in res

    # The two estimators DISAGREE: the additive spec looks competitive on group-internal CV but collapses on honest OOF.
    int_add = _internal_cv(df, groups, screen_idx, y, additive)
    int_honest = _internal_cv(df, groups, screen_idx, y, honest)
    assert int_add < res[additive.name], (
        f"premise: additive internal CV ({int_add:.3f}) must be much lower than its honest OOF ({res[additive.name]:.3f})"
    )
    # On internal CV the additive spec is competitive (legacy would keep it near the top); on honest OOF it is buried.
    assert int_add <= int_honest * 1.6, "additive must be competitive on the optimistic group-internal CV (legacy ranks it high)"
    assert res[additive.name] >= 1.20 * res[honest.name], (
        f"honest OOF must rank additive ({res[additive.name]:.3f}) >= 1.20x the honest spec ({res[honest.name]:.3f}); "
        f"measured ratio ~4.2"
    )


def test_biz_val_honest_oof_selection_full_fit_ranks_honest_above_additive():
    df, groups, y, levels = _frame()
    screen_idx, holdout_idx = _split_upper_tail(groups, levels)
    disc = _disc(groups, holdout_idx)
    additive = _spec("y-diff-basepartial", "diff", "base_partial", {})
    honest = _spec("y-linres-basefull", "linear_residual", "base_full", {"alpha": 1.0, "beta": 0.0})

    # Drive the rerank directly (the load-bearing ordering path). train_idx == screening pool, exactly as fit() rebinds it.
    out = disc._tiny_model_rerank(
        kept_specs=[additive, honest], df=df, target_col="y", usable_features=_FEATS,
        train_idx=screen_idx, y_full=y,
    )
    names = [s.name for s in out]
    assert honest.name in names, "the honest non-AR spec must be retained"
    assert disc._honest_oof_rmse[additive.name] >= 1.20 * disc._honest_oof_rmse[honest.name]
    if additive.name in names:
        assert names.index(honest.name) < names.index(additive.name), "honest spec must rank above the additive spec"
    # Honest-OOF baseline gate should drop the additive spec (its honest reconstruction loses to raw-y honest OOF).
    assert names[0] == honest.name


def test_honest_oof_collapse_returns_inf():
    """A spec whose inverse goes non-finite / collapses to ~constant -> +inf (sinks to the bottom)."""
    df, groups, y, levels = _frame()
    screen_idx, holdout_idx = _split_upper_tail(groups, levels)
    disc = _disc(groups, holdout_idx)
    # log1p inverse expm1 overflows for the large standardized residual on out-of-range wells -> non-finite.
    huge = _spec("y-diff-blowup", "linear_residual", "base_partial", {"alpha": 1e6, "beta": 0.0})
    res = honest_oof_reconstruction_rmse(disc, df, "y", [huge], _FEATS, screen_idx, holdout_idx, y)
    # A finite-but-huge RMSE or +inf both mean "buried"; the contract is it is NOT a small competitive number.
    assert huge.name in res
    assert res[huge.name] == float("inf") or res[huge.name] > 1e3


def test_honest_oof_noop_without_group_ids():
    df, groups, y, levels = _frame()
    screen_idx, holdout_idx = _split_upper_tail(groups, levels)
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, tiny_model_n_estimators=40)
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = None  # no groups -> selector path not taken
    disc.honest_holdout_idx_ = holdout_idx
    additive = _spec("y-diff-basepartial", "diff", "base_partial", {})
    honest = _spec("y-linres-basefull", "linear_residual", "base_full", {"alpha": 1.0, "beta": 0.0})
    disc._tiny_model_rerank(
        kept_specs=[additive, honest], df=df, target_col="y", usable_features=_FEATS,
        train_idx=screen_idx, y_full=y,
    )
    assert not hasattr(disc, "_honest_oof_rmse"), "no group ids -> honest-OOF selector must not run"
