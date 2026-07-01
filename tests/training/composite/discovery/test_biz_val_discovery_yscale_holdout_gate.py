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


def _train_and_val_extrapolating(per=150, n_train_groups=8, n_val_groups=4, seed=3):
    """Separate TRAIN and VAL frames. Group levels are MONOTONE, and the VAL groups sit at the HIGH
    end -- their base values fall OUTSIDE the train range, the real unseen-well extrapolation that
    collapses a high-alpha inverse. Mirrors the discovery wiring: df is train-only; val is a separate
    frame + its own targets."""
    rng = np.random.default_rng(seed)
    n_groups = n_train_groups + n_val_groups
    levels = np.linspace(100.0, 1000.0, n_groups)  # monotone: high-id groups = high base

    def _frame(group_ids):
        g = np.repeat(group_ids, per)
        base = levels[g] + rng.normal(0.0, 5.0, g.size)
        x1 = rng.normal(size=g.size)
        y = base + 3.0 * x1 + rng.normal(0.0, 5.0, g.size)
        df = pd.DataFrame({"base": base, "x1": x1, "x2": rng.normal(size=g.size), "y": y.astype(np.float64)})
        return df, g.astype(np.int64), y.astype(np.float64)

    train_df, train_groups, train_y = _frame(np.arange(n_train_groups))            # low base levels
    val_df, _val_groups, val_y = _frame(np.arange(n_train_groups, n_groups))       # high base levels (OOD)
    return train_df, train_groups, train_y, val_df, val_y


def test_biz_val_yscale_gate_drops_collapsing_spec_on_val_split():
    train_df, train_groups, train_y, val_df, val_y = _train_and_val_extrapolating()
    disc = _make_gate_ctx(train_groups)
    bad = _spec("y-linres-base-BADalpha", alpha=50.0)
    good = _spec("y-linres-base-unit", alpha=1.0)
    train_idx = np.arange(len(train_df))
    survivors = apply_yscale_holdout_gate(
        disc, train_df, "y", [bad, good], ["base", "x1", "x2"], train_idx, train_y,
        val_df=val_df, val_y=val_y,
    )
    names = {s.name for s in survivors}
    assert "y-linres-base-BADalpha" not in names, (
        "high-alpha spec must be dropped: it collapses on the unseen-well VAL split"
    )
    assert "y-linres-base-unit" in names, "stable unit-alpha spec must survive the val-split gate"


# ----------------------------------------------------------------------
# Structural fragility gate: catch base-additive inverses on per-well-level bases from TRAIN alone
# ----------------------------------------------------------------------
from mlframe.training.composite.discovery._yscale_holdout_gate import apply_structural_fragility_gate


def _spec_t(name: str, transform_name: str, base_column: str, params: dict) -> CompositeSpec:
    return CompositeSpec(
        name=name, target_col="y", transform_name=transform_name, base_column=base_column,
        fitted_params=dict(params), mi_gain=1.0, mi_y=0.0, mi_t=1.0,
        valid_domain_frac=1.0, n_train_rows=100,
    )


def _per_well_base_frame(n_groups=12, per=200, seed=5):
    """``base`` is a PER-WELL LEVEL (variance dominated by between-well differences spanning ~std(y));
    ``x1`` is a row-level predictor (no between-well level). y = base + within-well signal."""
    rng = np.random.default_rng(seed)
    well_level = rng.uniform(10000.0, 13000.0, n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    base = well_level[groups] + rng.normal(0.0, 30.0, groups.size)   # ~all between-well level
    x1 = rng.normal(size=groups.size)                                # row-level, no well level
    y = base + 600.0 * x1 + rng.normal(0.0, 50.0, groups.size)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y.astype(np.float64)})
    return df, groups.astype(np.int64), y.astype(np.float64)


def test_biz_val_structural_gate_drops_additive_inverse_on_per_well_base():
    df, groups, y = _per_well_base_frame()
    disc = _make_gate_ctx(groups)
    diff_spec = _spec_t("y-diff-base", "diff", "base", {})
    addres_spec = _spec_t("y-addres-base", "additive_residual", "base", {"beta": 0.0})
    good = _spec_t("y-linres-x1", "linear_residual", "x1", {"alpha": 0.5, "beta": 0.0})
    survivors = apply_structural_fragility_gate(disc, df, [diff_spec, addres_spec, good], np.arange(len(df)), y)
    names = {s.name for s in survivors}
    assert "y-diff-base" not in names, "diff on a per-well-level base must be dropped (additive inverse extrapolates)"
    assert "y-addres-base" not in names, "additive_residual on a per-well-level base must be dropped"
    assert "y-linres-x1" in names, "row-level base (no between-well level) must survive the structural gate"


def test_rejection_ledger_records_structural_gate_drops():
    """The rejection ledger answers "why was MY spec rejected?": a structural-fragility drop must appear in
    disc.rejection_ledger with stage='structural_fragility' + the dropped spec name + numbers -- previously the
    gate's verdict lived only in a discarded local list."""
    from mlframe.training.composite.discovery._rejection_ledger import ledger_init
    df, groups, y = _per_well_base_frame()
    disc = _make_gate_ctx(groups)
    ledger_init(disc)  # fit() does this; the direct-gate test must initialise it
    diff_spec = _spec_t("y-diff-base", "diff", "base", {})
    good = _spec_t("y-linres-x1", "linear_residual", "x1", {"alpha": 0.5, "beta": 0.0})
    apply_structural_fragility_gate(disc, df, [diff_spec, good], np.arange(len(df)), y)
    led = disc.rejection_ledger
    rows = [r for r in led if r["spec_name"] == "y-diff-base"]
    assert rows, f"dropped spec must have a ledger row; ledger={led}"
    assert rows[0]["stage"] == "structural_fragility"
    assert "between_total_ratio" in rows[0]["numbers"]
    assert not any(r["spec_name"] == "y-linres-x1" for r in led), "a surviving spec must NOT be in the rejection ledger"


def test_biz_val_structural_gate_noop_without_group_ids():
    df, _g, y = _per_well_base_frame()
    disc = _make_gate_ctx(None)
    diff_spec = _spec_t("y-diff-base", "diff", "base", {})
    out = apply_structural_fragility_gate(disc, df, [diff_spec], np.arange(len(df)), y)
    assert [s.name for s in out] == ["y-diff-base"], "no group ids -> structural gate is a no-op"


def test_biz_val_structural_gate_drops_chain_linres_variants_on_per_well_base():
    """chain_linres_* (linresYj / linresCbrt) wrap a linear_residual bivariate under a unary tail; the inverse still
    re-injects alpha*base, so on a per-well-level base they must be dropped exactly like plain linear_residual. Prod
    TVT 2026-06: these slipped the gate (sensitivity returned None) and collapsed to R^2=-146 on unseen wells."""
    df, groups, y = _per_well_base_frame()
    disc = _make_gate_ctx(groups)
    # The chain nests the OLS-fitted linear_residual params under ``bivariate_params``.
    chain_params = {"bivariate_params": {"alpha": 1.0, "beta": 0.0}, "unary_stage_params": [{}]}
    yj = _spec_t("y-linresYj-base", "chain_linres_yj", "base", chain_params)
    cbrt = _spec_t("y-linresCbrt-base", "chain_linres_cbrt", "base", chain_params)
    alt = _spec_t("y-chainlinres-base", "chain_linear_residual_yj", "base", chain_params)
    good = _spec_t("y-linresYj-x1", "chain_linres_yj", "x1",
                   {"bivariate_params": {"alpha": 0.5, "beta": 0.0}, "unary_stage_params": [{}]})
    survivors = apply_structural_fragility_gate(disc, df, [yj, cbrt, alt, good], np.arange(len(df)), y)
    names = {s.name for s in survivors}
    assert "y-linresYj-base" not in names, "chain_linres_yj on a per-well base must be dropped (re-injects alpha*base)"
    assert "y-linresCbrt-base" not in names, "chain_linres_cbrt on a per-well base must be dropped"
    assert "y-chainlinres-base" not in names, "chain_linear_residual_yj on a per-well base must be dropped"
    assert "y-linresYj-x1" in names, "a row-level base (no between-well level) chain must still survive"


def test_biz_val_structural_gate_drops_chain_monres_on_per_well_base():
    """chain_monres_* (monotonic_residual + unary tail) also re-inject a per-group-level base at inverse; its bivariate
    has no ``alpha`` so sensitivity resolves to 1.0 (base-additive). Prod TVT 2026-07: chain_monotonic_residual_yj on
    pf_tvt_post_p90 was auto-chain-discovered AFTER the early structural pass, bypassed it, and collapsed R^2=-4.44 on
    test; the second structural pass (after opt-in steps) + this sensitivity must drop it on a per-well base."""
    df, groups, y = _per_well_base_frame()
    disc = _make_gate_ctx(groups)
    monres = _spec_t("y-monresYj-base", "chain_monotonic_residual_yj", "base",
                     {"bivariate_params": {"iso": 1}, "unary_stage_params": [{}]})
    good = _spec_t("y-monresYj-x1", "chain_monres_yj", "x1",
                   {"bivariate_params": {"iso": 1}, "unary_stage_params": [{}]})
    survivors = apply_structural_fragility_gate(disc, df, [monres, good], np.arange(len(df)), y)
    names = {s.name for s in survivors}
    assert "y-monresYj-base" not in names, "chain_monres on a per-well base must be dropped (re-injects base)"
    assert "y-monresYj-x1" in names, "a row-level base chain_monres must survive"
