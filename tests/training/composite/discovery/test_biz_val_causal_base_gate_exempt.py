"""Unit + biz_value: strictly-causal bases are exempt from the near-copy-of-y and structural-fragility gates.

Both gates drop a base whose additive inverse ``y = T_hat + s*base`` extrapolates on unseen groups: near-copy fires when
|corr(base,y)|~1, structural-fragility when the base variance is between-group-level dominated. On a strong-AR sequential
target the CAUSAL lag (``{y}_prev`` or an engineered ``{y}__gcausal_*`` base) trips BOTH -- yet it is the single best base
there, and its inverse re-injects a REAL per-row previous value so it stays in-range on unseen groups. These tests pin the
provenance-only exemption (a contemporaneous near-copy of y that is NOT a causal construct is still dropped).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery._causal_lag import is_causal_base_name
from mlframe.training.composite.discovery._yscale_holdout_gate import apply_structural_fragility_gate
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class TestIsCausalBaseName:
    def test_marker_matches(self):
        assert is_causal_base_name("TVT__gcausal_lag1")
        assert is_causal_base_name("TVT__gcausal_expmean", "TVT")

    def test_named_lag_matches_with_target(self):
        assert is_causal_base_name("TVT_prev", "TVT")
        assert is_causal_base_name("TVT_lag_1", "TVT")

    def test_named_lag_requires_target(self):
        # Without target_col, a stray *_prev is NOT silently exempted (only the unambiguous engineered marker matches).
        assert not is_causal_base_name("TVT_prev", None)
        assert not is_causal_base_name("something_prev", "TVT")

    def test_plain_near_copy_not_matched(self):
        assert not is_causal_base_name("y_shadow", "y")
        assert not is_causal_base_name("", "y")


def _spec(name, base_column, transform_name="diff"):
    return CompositeSpec(
        name=name,
        target_col="y",
        transform_name=transform_name,
        base_column=base_column,
        fitted_params={"alpha": 1.0, "beta": 0.0},
        mi_gain=1.0,
        mi_y=0.0,
        mi_t=1.0,
        valid_domain_frac=1.0,
        n_train_rows=100,
    )


def _level_frame(n_groups=30, per=60, seed=0):
    """Per-group LEVEL columns (between-group-var dominated -> structurally fragile). Both a causal-named column and a
    plain column carry the SAME level, so only provenance distinguishes them."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(0.0, 50.0, n_groups)
    g = np.repeat(np.arange(n_groups), per)
    lvl = level[g] + rng.normal(0.0, 0.1, g.size)
    df = pd.DataFrame(
        {
            "y": level[g] + rng.normal(0.0, 1.0, g.size),
            "y__gcausal_lag1": lvl.copy(),  # causal provenance (engineered marker)
            "level_base": lvl.copy(),  # identical values, NO causal provenance
        }
    )
    return df, g.astype(np.int64), df["y"].to_numpy()


def _disc(groups, exempt):
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, causal_base_gate_exempt=exempt)
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = groups
    disc._target_col = "y"
    return disc


@pytest.fixture(autouse=True)
def _silence():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        yield


class TestStructuralFragilityExemption:
    def test_biz_val_causal_base_survives_structural_gate_when_exempt(self):
        df, groups, y = _level_frame()
        train_idx = np.arange(groups.size)
        causal = _spec("y-diff-gcausal", "y__gcausal_lag1")
        plain = _spec("y-diff-level", "level_base")
        disc = _disc(groups, exempt=True)
        kept = apply_structural_fragility_gate(disc, df, [causal, plain], train_idx, y)
        names = [s.name for s in kept]
        assert "y-diff-gcausal" in names, "the causal base must be exempt from the structural-fragility gate"
        assert "y-diff-level" not in names, "the identical NON-causal per-group-level base must still be dropped as fragile"

    def test_biz_val_causal_base_dropped_when_exemption_off(self):
        df, groups, y = _level_frame()
        train_idx = np.arange(groups.size)
        causal = _spec("y-diff-gcausal", "y__gcausal_lag1")
        disc = _disc(groups, exempt=False)
        kept = apply_structural_fragility_gate(disc, df, [causal], train_idx, y)
        assert "y-diff-gcausal" not in [s.name for s in kept], "exemption OFF -> the causal base is gated like any fragile base"


def _strong_ar_frame(n_groups=25, per=80, seed=1, step=0.4):
    """Strong within-group AR: y is a slow random walk per group so lag-1 autocorr ~1 and corr(y_prev, y) > 0.9995
    (a near-copy). y_prev is the exact causal lag (named by the {target}_prev provenance)."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.0, 100.0, n_groups)
    g = np.repeat(np.arange(n_groups), per)
    y = np.empty(g.size, dtype=np.float64)
    y_prev = np.empty(g.size, dtype=np.float64)
    idx = 0
    for gi in range(n_groups):
        prev = float(start[gi])
        for _ in range(per):
            y_prev[idx] = prev
            cur = prev + rng.normal(0.0, step)
            y[idx] = cur
            prev = cur
            idx += 1
    df = pd.DataFrame({"y": y, "y_prev": y_prev, "x1": rng.normal(size=g.size)})
    return df, g.astype(np.int64), y


class TestNearCopyExemptionEndToEnd:
    def test_biz_val_strong_ar_lag_base_retained_when_exempt_excluded_when_off(self):
        df, groups, y = _strong_ar_frame()
        n = groups.size
        # Sanity: y_prev really is a near-copy of y (the gate would fire).
        corr = abs(float(np.corrcoef(df["y_prev"], df["y"])[0, 1]))
        assert corr > 0.9995, f"synthetic must make y_prev a near-copy (corr={corr:.6f})"

        def _run(exempt):
            cfg = CompositeTargetDiscoveryConfig(
                enabled=True,
                screening="mi",
                mi_estimator="bin",
                base_candidates="auto",
                transforms=("diff",),
                group_column="well",
                random_state=0,
                causal_base_gate_exempt=exempt,
                engineer_causal_bases=False,  # isolate the named-lag near-copy path, not the engineered bases
            )
            df2 = df.assign(well=groups)
            disc = CompositeTargetDiscovery(cfg)
            disc._group_ids_for_rerank = groups
            disc.fit(df2, "y", ["y_prev", "x1", "well"], train_idx=np.arange(n))
            return [s.base_column for s in disc.specs_]

        bases_exempt = _run(True)
        assert "y_prev" in bases_exempt, "causal lag base must be retained (not gated as near-copy) when exempt"
        bases_off = _run(False)
        assert "y_prev" not in bases_off, "with exemption off, the near-copy gate excludes the causal lag base"
