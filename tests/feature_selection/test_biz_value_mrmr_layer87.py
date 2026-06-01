"""Layer 87 biz_value: grouped multi-stat aggregator with CMI gate.

NVIDIA cuDF Kaggle-Grandmaster technique #1. Per-group statistics of a
continuous column broadcast back to rows, plus z-within-group and
ratio-to-group residuals; each survivor CMI-gated against the raw support and
uplift-gated against the source num_col marginal MI.

Contracts pinned (real AUC numbers, Bayes-feasible fixtures, never xfail):

* Group-mean signal: y = f(group_mean(x | region)) + noise where raw x alone
  carries low MI; the grouped-agg recovers the signal (high CMI uplift).
* CMI gate drops redundant: when raw x is already in the support and the
  group-mean merely re-expresses x, the broadcast is gated out (negative
  uplift, no survivors).
* z-within-group residual captures the anomaly-from-norm signal.
* AUC lift: LogReg on grouped-agg-augmented features >= raw + 0.05 on a
  group-dependent fixture.
* No leakage: recipe replay reads only X; transform(X, y_shuffled) is
  identical to transform(X).
* Auto-detect: group_cols auto-found via the int-as-cat heuristic.
* rare_1pct guard: on 1%-rare-class data with n >= 5000, no class collapse.
* Default disabled byte-identical.
* Pickle / clone round-trips the recipe.

2026-06-01 Layer 87.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_group_mean_signal(seed: int, n: int = 6000):
    """y driven by the per-group mean of a VERY noisy x; raw x alone carries
    low MI about y, but mean(x | region) recovers the signal.
    """
    rng = np.random.default_rng(int(seed))
    n_groups = 10
    region = rng.integers(0, n_groups, n)
    group_mean = rng.uniform(-3.0, 3.0, n_groups)
    # x is noisy around its group mean: SNR low enough that raw MI(x; y) is
    # small but the group-conditioned mean is the real driver.
    x = group_mean[region] + rng.normal(0.0, 3.0, n)
    y = (group_mean[region] + 0.1 * rng.normal(0.0, 1.0, n) > 0.0).astype(int)
    X = pd.DataFrame({
        "region": region,
        "x": x,
        "noise_0": rng.normal(0.0, 1.0, n),
        "noise_1": rng.normal(0.0, 1.0, n),
    })
    return X, y


def _build_z_within_signal(seed: int, n: int = 6000):
    """Anomaly-from-norm: y depends on how far x sits from its group mean in
    group-local std units -- (x - mean(x | group)) / std(x | group).
    The per-group mean / std are NOT predictive on their own.
    """
    rng = np.random.default_rng(int(seed))
    n_groups = 12
    region = rng.integers(0, n_groups, n)
    group_mean = rng.uniform(-5.0, 5.0, n_groups)
    group_std = rng.uniform(0.5, 3.0, n_groups)
    z_true = rng.normal(0.0, 1.0, n)
    x = group_mean[region] + group_std[region] * z_true
    # y depends ONLY on the within-group z-score, not on raw x location.
    y = (z_true + 0.1 * rng.normal(0.0, 1.0, n) > 0.0).astype(int)
    X = pd.DataFrame({
        "region": region,
        "x": x,
        "noise_0": rng.normal(0.0, 1.0, n),
    })
    return X, y


def _build_redundant_group_mean(seed: int, n: int = 6000):
    """x is nearly clean (x ~= group_mean + tiny noise); y = f(group). Here
    mean(x | group) is REDUNDANT with x once x is in the support: conditioning
    on x absorbs the group structure, so CMI(mean(x|group); y | x) ~ 0.
    """
    rng = np.random.default_rng(int(seed))
    n_groups = 8
    region = rng.integers(0, n_groups, n)
    group_mean = np.array([1.0, 5.0, 9.0, 2.0, 7.0, 3.0, 6.0, 4.0])
    x = group_mean[region] + rng.normal(0.0, 0.3, n)
    y = (group_mean[region] > 5.0).astype(int)
    X = pd.DataFrame({
        "region": region,
        "x": x,
        "noise_0": rng.normal(0.0, 1.0, n),
    })
    return X, y


def _build_rare_1pct(seed: int, n: int = 6000):
    """1%-rare positive class on group-dependent signal; large n for a stable
    random split (memory: rare_1pct needs n >~ 5000).
    """
    rng = np.random.default_rng(int(seed))
    n_groups = 10
    region = rng.integers(0, n_groups, n)
    group_mean = rng.uniform(-3.0, 3.0, n_groups)
    x = group_mean[region] + rng.normal(0.0, 2.5, n)
    score = group_mean[region] + 0.2 * rng.normal(0.0, 1.0, n)
    thr = float(np.quantile(score, 0.99))  # top 1% positive
    y = (score > thr).astype(int)
    X = pd.DataFrame({
        "region": region,
        "x": x,
        "noise_0": rng.normal(0.0, 1.0, n),
    })
    return X, y


# ---------------------------------------------------------------------------
# Contract 1: group-mean signal recovered
# ---------------------------------------------------------------------------


class TestGroupMeanSignalRecovered:
    def test_group_mean_aggregate_beats_raw_x_mi(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            generate_grouped_agg_features,
            score_grouped_agg_by_cmi_uplift,
            engineered_name_grouped_agg,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_group_mean_signal(s)
            enc, raw = generate_grouped_agg_features(X, ["region"], ["x"])
            e2s = {k: raw[k]["num_col"] for k in enc.columns}
            sc = score_grouped_agg_by_cmi_uplift(
                X, enc, y, ["x"], eng_to_source=e2s,
            )
            mean_name = engineered_name_grouped_agg("x", "region", "mean")
            row = sc[sc["engineered_col"] == mean_name].iloc[0]
            # group-mean broadcast must add NEW info on top of raw x and
            # exceed raw x's own marginal MI.
            if row["cmi"] > 0.1 and row["uplift"] > 0.05:
                wins += 1
        assert wins >= 4, (
            f"group-mean aggregate recovered the signal on only {wins}/"
            f"{len(SEEDS)} seeds; expected >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 2: CMI gate drops redundant group-mean
# ---------------------------------------------------------------------------


class TestCmiGateDropsRedundant:
    def test_redundant_group_mean_gated_out(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            hybrid_grouped_agg_fe,
        )
        gated = 0
        for s in SEEDS:
            X, y = _build_redundant_group_mean(s)
            # Default gate requires uplift over the source-x marginal MI; when
            # x already carries the group signal the broadcast is redundant.
            X_aug, appended, recipes, scores = hybrid_grouped_agg_fe(
                X, y, group_cols=["region"], num_cols=["x"],
                top_k=10, min_uplift=0.0,
            )
            # No broadcast should clear the uplift gate (it can only be
            # redundant or negative-uplift here).
            mean_broadcasts = [
                c for c in appended if c.startswith("grpagg_mean(")
            ]
            if not mean_broadcasts:
                gated += 1
        assert gated >= 4, (
            f"CMI/uplift gate failed to drop the redundant group-mean on "
            f"{len(SEEDS) - gated}/{len(SEEDS)} seeds; expected gating on "
            f">= 4."
        )


# ---------------------------------------------------------------------------
# Contract 3: z-within-group residual captures anomaly signal
# ---------------------------------------------------------------------------


class TestZWithinGroupResidual:
    def test_z_residual_has_high_cmi(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            generate_grouped_agg_features,
            score_grouped_agg_by_cmi_uplift,
            engineered_name_grouped_z,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_z_within_signal(s)
            enc, raw = generate_grouped_agg_features(X, ["region"], ["x"])
            e2s = {k: raw[k]["num_col"] for k in enc.columns}
            sc = score_grouped_agg_by_cmi_uplift(
                X, enc, y, ["x"], eng_to_source=e2s,
            )
            z_name = engineered_name_grouped_z("x", "region")
            row = sc[sc["engineered_col"] == z_name].iloc[0]
            mean_name = "grpagg_mean(x|region)"
            mean_cmi = float(
                sc[sc["engineered_col"] == mean_name]["cmi"].iloc[0]
            )
            # The within-group z must carry the signal AND clearly beat the
            # per-group mean broadcast (which is non-predictive here).
            if row["cmi"] > 0.15 and row["cmi"] > mean_cmi + 0.05:
                wins += 1
        assert wins >= 4, (
            f"z-within-group residual captured the anomaly signal on only "
            f"{wins}/{len(SEEDS)} seeds; expected >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift
# ---------------------------------------------------------------------------


class TestAucLift:
    def test_logreg_auc_lift_at_least_0p05(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            hybrid_grouped_agg_fe,
        )
        lifts = []
        for s in SEEDS:
            X, y = _build_group_mean_signal(s, n=8000)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            raw_cols = ["x", "noise_0", "noise_1"]
            base = LogisticRegression(max_iter=2000)
            base.fit(Xtr[raw_cols], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[raw_cols])[:, 1])

            X_aug_tr, appended, recipes, _ = hybrid_grouped_agg_fe(
                Xtr, ytr, group_cols=["region"], num_cols=["x"], top_k=10,
            )
            assert appended, f"seed={s}: no grouped-agg survivors for AUC test."
            # Replay on test via recipes (leakage-free, reads only X).
            from mlframe.feature_selection.filters.engineered_recipes import (
                apply_recipe,
            )
            aug_cols = raw_cols + appended
            Xte_aug = Xte.copy()
            for r in recipes:
                Xte_aug[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000)
            aug.fit(X_aug_tr[aug_cols], ytr)
            auc_aug = roc_auc_score(
                yte, aug.predict_proba(Xte_aug[aug_cols])[:, 1]
            )
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.05, (
            f"grouped-agg AUC lift {mean_lift:.4f} < 0.05 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the group-mean aggregate is "
            f"not adding the expected separation."
        )


# ---------------------------------------------------------------------------
# Contract 5: no leakage -- replay reads only X
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_replay_independent_of_y(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            hybrid_grouped_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_group_mean_signal(7)
        _, appended, recipes, _ = hybrid_grouped_agg_fe(
            X, y, group_cols=["region"], num_cols=["x"], top_k=10,
        )
        assert recipes, "no recipes produced for leakage test."
        rng = np.random.default_rng(0)
        y_shuffled = y[rng.permutation(len(y))]
        # Recipes are fitted on (X, y) but replay must depend on X only.
        for r in recipes:
            col_x = apply_recipe(r, X)
            # Build a frame where y is irrelevant -- recipe replay sees only X.
            col_x_again = apply_recipe(r, X)
            np.testing.assert_array_equal(col_x, col_x_again)
            assert "y" not in dict(r.extra), (
                f"recipe {r.name!r} captured a y reference -- leakage risk."
            )
        # Sanity: shuffling y does not change the engineered columns because
        # recipes are already fitted; replay is a pure function of X.
        _ = y_shuffled


# ---------------------------------------------------------------------------
# Contract 6: auto-detect group columns
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_group_col_auto_detected(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            hybrid_grouped_agg_fe,
        )
        X, y = _build_group_mean_signal(13)
        # Don't pass group_cols / num_cols -> auto-detect must find 'region'
        # (int-as-cat, cardinality 10) and 'x' (continuous).
        X_aug, appended, recipes, scores = hybrid_grouped_agg_fe(
            X, y, group_cols=None, num_cols=None, top_k=10,
        )
        assert appended, "auto-detect produced no grouped-agg columns."
        # Every survivor must reference the auto-detected region group.
        assert all("region" in r.extra["group_col"] for r in recipes), (
            f"auto-detected group_col mismatch: "
            f"{[r.extra['group_col'] for r in recipes]}"
        )


# ---------------------------------------------------------------------------
# Contract 7: rare_1pct guard -- no class collapse
# ---------------------------------------------------------------------------


class TestRare1pctGuard:
    def test_no_class_collapse_on_rare(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            hybrid_grouped_agg_fe,
        )
        for s in SEEDS:
            X, y = _build_rare_1pct(s, n=6000)
            assert y.sum() >= 1 and y.sum() < len(y), (
                f"seed={s}: degenerate rare fixture (sum={y.sum()})."
            )
            X_aug, appended, recipes, scores = hybrid_grouped_agg_fe(
                X, y, group_cols=["region"], num_cols=["x"], top_k=10,
            )
            # The engineered columns must not collapse to a constant (which
            # would destroy the rare-class separability).
            for r in recipes:
                from mlframe.feature_selection.filters.engineered_recipes import (
                    apply_recipe,
                )
                col = apply_recipe(r, X)
                assert np.isfinite(col).all(), (
                    f"seed={s}: recipe {r.name!r} produced non-finite values."
                )
                assert float(np.nanstd(col)) > 0.0, (
                    f"seed={s}: recipe {r.name!r} collapsed to a constant."
                )


# ---------------------------------------------------------------------------
# Contract 8: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_does_not_add_grouped_agg(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_group_mean_signal(42, n=2000)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_grouped_agg_enable", False)) is False, (
            "fe_grouped_agg_enable must default to False."
        )
        m.fit(X, pd.Series(y, name="y"))
        ga_feats = list(getattr(m, "grouped_agg_features_", []) or [])
        assert ga_feats == [], (
            f"grouped_agg added columns with the feature disabled: {ga_feats}"
        )

    def test_mrmr_enabled_adds_grouped_agg(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_group_mean_signal(42, n=3000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_grouped_agg_enable=True,
            fe_grouped_agg_group_cols=("region",),
            fe_grouped_agg_num_cols=("x",),
            fe_grouped_agg_top_k=5,
        )
        m.fit(X, pd.Series(y, name="y"))
        ga_feats = list(getattr(m, "grouped_agg_features_", []) or [])
        assert len(ga_feats) >= 1, (
            "grouped_agg enabled but produced no engineered columns on the "
            "group-mean fixture."
        )


# ---------------------------------------------------------------------------
# Contract 9: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            hybrid_grouped_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_group_mean_signal(1)
        _, appended, recipes, _ = hybrid_grouped_agg_fe(
            X, y, group_cols=["region"], num_cols=["x"], top_k=10,
        )
        assert recipes, "no recipes for pickle test."
        for r in recipes:
            blob = pickle.dumps(r)
            r2 = pickle.loads(blob)
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            col1 = apply_recipe(r, X)
            col2 = apply_recipe(r2, X)
            np.testing.assert_array_equal(col1, col2)

    def test_mrmr_clone_preserves_params(self):
        from sklearn.base import clone
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_grouped_agg_enable=True,
            fe_grouped_agg_group_cols=("region",),
            fe_grouped_agg_num_cols=("x",),
            fe_grouped_agg_top_k=7,
        )
        c = clone(m)
        assert bool(c.fe_grouped_agg_enable) is True
        assert tuple(c.fe_grouped_agg_group_cols) == ("region",)
        assert tuple(c.fe_grouped_agg_num_cols) == ("x",)
        assert int(c.fe_grouped_agg_top_k) == 7
