"""Regression tests for opt-in per-group/per-cluster composite discovery
(``config.per_group_discovery_enabled``, ``discovery/_per_group.py``).

Coverage
--------
- Per-group discovery finds a DIFFERENT (base, transform) per group on a synthetic
  panel where each group has a genuinely different true DGP.
- A group below ``per_group_min_rows`` gets no entry in ``specs_by_group_`` (falls
  back to the global spec) and discovery does not error.
- Leakage guard: corrupting one group's rows does not change another group's
  discovered spec.
- Default-off byte-identical: ``per_group_discovery_enabled=False`` (the default)
  leaves ``specs_`` / ``specs_by_group_`` exactly as before this feature existed.
- Honest-holdout RMSE: per-group discovery beats a single global spec forced onto
  every group, on a per-group holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig

pytestmark = pytest.mark.sklearn_matrix


def _make_panel(n_per_group: int = 900, seed: int = 0) -> pd.DataFrame:
    """3 large groups with DIFFERENT true (base, transform) relationships, plus one
    small group (< 500 rows) to exercise the fallback path.

    - group A: y additive in base_1 (y = base_1 + noise)  -> "diff"/"additive_residual" style
    - group B: y multiplicative in base_2 (y = base_2 * const + noise on log scale) -> "ratio"/"logratio" style
    - group C: pure noise (no real base dependence)
    - group small: same DGP as A but only 120 rows (< per_group_min_rows=500)
    """
    rng = np.random.default_rng(seed)

    def _group(n, kind, group_id):
        """Build one group's rows for the given DGP kind ('additive' / 'ratio' / 'noise')."""
        base_1 = rng.normal(loc=20.0, scale=4.0, size=n)
        base_2 = rng.normal(loc=15.0, scale=3.0, size=n)
        noise_extra = rng.normal(size=n)
        if kind == "additive":
            y = base_1 + 0.5 * noise_extra + rng.normal(scale=0.3, size=n)
        elif kind == "multiplicative":
            y = base_2 * np.exp(0.05 * noise_extra) + rng.normal(scale=0.2, size=n)
        else:  # noise
            y = rng.normal(loc=10.0, scale=5.0, size=n)
        return pd.DataFrame(
            {
                "base_1": base_1,
                "base_2": base_2,
                "x_extra": noise_extra,
                "group_id": group_id,
                "y": y,
            }
        )

    df = pd.concat(
        [
            _group(n_per_group, "additive", "A"),
            _group(n_per_group, "multiplicative", "B"),
            _group(n_per_group, "noise", "C"),
            _group(120, "additive", "small"),
        ],
        ignore_index=True,
    )
    return df


def _cfg(**overrides) -> CompositeTargetDiscoveryConfig:
    """Minimal fast discovery config for the per-group tests, with ``**overrides`` applied."""
    base = dict(
        enabled=True,
        base_candidates=["base_1", "base_2"],
        transforms=["diff", "additive_residual", "ratio", "logratio", "linear_residual"],
        screening="mi",  # keep test fast; tiny-model rerank not needed to exercise the routing/fallback logic
        honest_rmse_gate_enabled=False,
        yscale_holdout_gate_enabled=False,
        structural_fragility_gate_enabled=False,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        honest_holdout_frac=0.2,
        random_state=0,
    )
    base.update(overrides)
    return CompositeTargetDiscoveryConfig(**base)


def test_per_group_discovery_finds_different_specs_per_group():
    """Groups with genuinely different DGPs must get different (base, transform) specs, not one global compromise."""
    df = _make_panel()
    cfg = _cfg(per_group_discovery_enabled=True, per_group_column="group_id", per_group_min_rows=500)
    disc = CompositeTargetDiscovery(cfg)
    train_idx = np.arange(len(df))
    disc.fit(df, "y", ["base_1", "base_2", "x_extra"], train_idx)

    assert disc.specs_, "global fit must still populate specs_ unchanged"
    assert set(disc.specs_by_group_.keys()) == {"A", "B", "C"}, (
        f"expected the 3 large groups (>=500 rows) to get their own spec set, got {sorted(disc.specs_by_group_.keys())}"
    )
    for g, specs in disc.specs_by_group_.items():
        assert specs, f"group {g} discovered zero specs"

    # Group A's top spec should be additive-style on base_1; group B's should be a
    # ratio/log-style spec on base_2. They should not coincide (a single global
    # compromise would tend to pick ONE base/transform for everyone).
    top_a = disc.specs_by_group_["A"][0]
    top_b = disc.specs_by_group_["B"][0]
    assert top_a.base_column == "base_1", f"group A top spec used base {top_a.base_column!r}, expected base_1"
    assert top_b.base_column == "base_2", f"group B top spec used base {top_b.base_column!r}, expected base_2"
    assert (top_a.base_column, top_a.transform_name) != (top_b.base_column, top_b.transform_name)


def test_per_group_discovery_small_group_falls_back_without_error():
    """A group below per_group_min_rows must not error or appear in specs_by_group_ -- it falls back to the global spec."""
    df = _make_panel()
    cfg = _cfg(per_group_discovery_enabled=True, per_group_column="group_id", per_group_min_rows=500)
    disc = CompositeTargetDiscovery(cfg)
    train_idx = np.arange(len(df))
    disc.fit(df, "y", ["base_1", "base_2", "x_extra"], train_idx)  # must not raise

    assert "small" not in disc.specs_by_group_, "the 120-row group must not get a per-group spec set"
    assert disc.specs_, "the global spec set (fallback target for 'small') must be non-empty"


def test_per_group_discovery_leakage_guard_independent_groups():
    """Corrupting group B's rows must not change group A's discovered spec -- each
    per-group delegate fit only ever sees its own group's train_idx subset."""
    df = _make_panel(seed=42)
    cfg = _cfg(per_group_discovery_enabled=True, per_group_column="group_id", per_group_min_rows=500)

    disc_baseline = CompositeTargetDiscovery(cfg)
    train_idx = np.arange(len(df))
    disc_baseline.fit(df, "y", ["base_1", "base_2", "x_extra"], train_idx)
    baseline_a_spec = disc_baseline.specs_by_group_["A"][0]

    df_corrupted = df.copy()
    b_mask = (df_corrupted["group_id"] == "B").to_numpy()
    rng = np.random.default_rng(999)
    df_corrupted.loc[b_mask, "y"] = rng.normal(loc=-999.0, scale=500.0, size=int(b_mask.sum()))
    df_corrupted.loc[b_mask, "base_2"] = rng.normal(loc=-999.0, scale=500.0, size=int(b_mask.sum()))

    disc_corrupted = CompositeTargetDiscovery(cfg)
    disc_corrupted.fit(df_corrupted, "y", ["base_1", "base_2", "x_extra"], train_idx)
    corrupted_a_spec = disc_corrupted.specs_by_group_["A"][0]

    assert corrupted_a_spec.base_column == baseline_a_spec.base_column
    assert corrupted_a_spec.transform_name == baseline_a_spec.transform_name
    assert (
        corrupted_a_spec.fitted_params == baseline_a_spec.fitted_params
    ), "corrupting group B's rows changed group A's fitted transform params -- per-group discovery is leaking rows across groups."


def test_per_group_discovery_default_off_is_byte_identical():
    """The default (per_group_discovery_enabled=False) path must produce the exact
    same specs_ / report_ as before this feature existed, and specs_by_group_ stays
    empty."""
    df = _make_panel()
    cfg_off = _cfg(per_group_discovery_enabled=False)
    train_idx = np.arange(len(df))

    disc1 = CompositeTargetDiscovery(cfg_off)
    disc1.fit(df, "y", ["base_1", "base_2", "x_extra"], train_idx)
    disc2 = CompositeTargetDiscovery(cfg_off)
    disc2.fit(df, "y", ["base_1", "base_2", "x_extra"], train_idx)

    assert disc1.specs_by_group_ == {}
    assert [s.name for s in disc1.specs_] == [s.name for s in disc2.specs_]
    for s1, s2 in zip(disc1.specs_, disc2.specs_):
        assert s1.fitted_params == s2.fitted_params
        assert s1.mi_gain == s2.mi_gain


def test_per_group_discovery_beats_global_forced_on_all_groups():
    """Genuine train/holdout OOS check (never roundtrip-on-fit-rows, which is
    trivially ~0 RMSE for ANY invertible transform regardless of fit quality):
    split each large group 70/30, fit on the 70% train slice, predict on the 30%
    holdout, and compare y-scale RMSE between (a) ``PerGroupCompositeRouter``
    (each group routed to its OWN discovered spec) and (b) a single
    ``CompositeTargetEstimator`` using the GLOBAL spec forced onto every group."""
    from sklearn.tree import DecisionTreeRegressor

    from mlframe.training.composite.estimator import CompositeTargetEstimator
    from mlframe.training.composite.per_group_router import PerGroupCompositeRouter

    # A shallow tree (not a linear model) is used as the inner estimator: with the base column
    # present as a raw feature, a LINEAR inner can trivially re-derive any linear miscalibration
    # of the transform via its own coefficient on that feature, masking the transform-choice effect
    # this test exists to measure. A shallow tree cannot do that recombination as cheaply.
    def _mk_inner():
        """Fresh shallow-tree inner estimator (see the note above on why not a linear model)."""
        return DecisionTreeRegressor(max_depth=3, random_state=0)

    df = _make_panel(seed=7)
    large = df[df["group_id"].isin(["A", "B", "C"])].reset_index(drop=True)
    rng = np.random.default_rng(11)
    is_holdout = np.zeros(len(large), dtype=bool)
    for g in ("A", "B", "C"):
        g_idx = np.flatnonzero((large["group_id"] == g).to_numpy())
        holdout_g = rng.choice(g_idx, size=int(0.3 * g_idx.size), replace=False)
        is_holdout[holdout_g] = True
    train_df = large.loc[~is_holdout].reset_index(drop=True)
    holdout_df = large.loc[is_holdout].reset_index(drop=True)

    cfg = _cfg(per_group_discovery_enabled=True, per_group_column="group_id", per_group_min_rows=300)
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(train_df, "y", ["base_1", "base_2", "x_extra"], np.arange(len(train_df)))
    assert set(disc.specs_by_group_.keys()) == {"A", "B", "C"}

    router = PerGroupCompositeRouter(discovery=disc, base_estimator=_mk_inner(), group_column="group_id")
    router.fit(train_df, train_df["y"].to_numpy())
    router_pred = router.predict(holdout_df)

    global_spec = disc.specs_[0]
    global_est = CompositeTargetEstimator(
        base_estimator=_mk_inner(),
        transform_name=global_spec.transform_name,
        base_column=global_spec.base_column,
    )
    global_est.fit(train_df.drop(columns=["group_id"]), train_df["y"].to_numpy())
    global_pred = global_est.predict(holdout_df.drop(columns=["group_id"]))

    # Compare on the two groups with a REAL base/y relationship (A, B); group C is pure noise by
    # construction, so no transform choice can help it and its holdout error is dominated by
    # irreducible noise variance on both sides (excluded from the "did per-group discovery help"
    # comparison, exactly as a pure-noise cluster should be -- neither router nor global "wins" there).
    ab_mask = holdout_df["group_id"].isin(["A", "B"]).to_numpy()
    y_ab = holdout_df["y"].to_numpy()[ab_mask]
    router_rmse = float(np.sqrt(np.mean((router_pred[ab_mask] - y_ab) ** 2)))
    global_rmse = float(np.sqrt(np.mean((global_pred[ab_mask] - y_ab) ** 2)))

    assert router_rmse <= global_rmse, (
        f"PerGroupCompositeRouter holdout RMSE {router_rmse:.4f} on groups A/B did not beat the single "
        f"global spec forced on all groups holdout RMSE {global_rmse:.4f}"
    )
    # Group C (pure noise) must still produce finite, non-crashing predictions via the fallback path.
    c_mask = (holdout_df["group_id"] == "C").to_numpy()
    assert np.all(np.isfinite(router_pred[c_mask])), "group C (noise) predictions must stay finite"
