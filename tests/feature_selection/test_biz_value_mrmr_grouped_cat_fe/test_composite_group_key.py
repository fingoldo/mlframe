"""Layer 93 biz_value: COMPOSITE (multi-column) group-key aggregator.

Multi-column extension of Layer 87. Real-world aggregations key on more than
one column at once -- ``groupby([region, month])`` / ``groupby([store,
category])``. The interaction at the composite level frequently carries signal
that neither single-column group exposes.

Contracts pinned (real AUC numbers, Bayes-feasible fixtures, never xfail):

* Composite-group signal: y = f(mean(value | region, month)) where neither the
  per-region nor the per-month group_mean captures it (the cell mean is an
  interaction); the composite aggregate recovers it (high CMI uplift).
* Single-group can't capture composite: L87 single-group mean(value | region)
  has LOW MI while composite mean(value | region, month) has HIGH MI.
* Cardinality refusal: a composite key with cardinality > 0.5*n is refused
  (no explosion, no columns emitted).
* AUC lift: LogReg on composite-aug features >= raw + 0.05.
* CMI gate drops redundant composite: a composite that merely duplicates a
  single-group signal already in the support is gated out.
* No leakage: recipe replay reads only X; transform(X, y_shuffled) ==
  transform(X).
* Default disabled byte-identical.
* Pickle / clone round-trips the recipe.

2026-06-01 Layer 93.

Consolidated verbatim from test_biz_value_mrmr_layer93.py (per audit finding test_code_quality-16).
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


def _build_composite_signal(seed: int, n: int = 8000):
    """y driven by the per-(region, month) CELL mean of a noisy x, where the
    cell means are an INTERACTION: averaged over month each region is ~0, and
    averaged over region each month is ~0, so neither single-group mean carries
    the signal -- only the composite (region, month) mean does.
    """
    rng = np.random.default_rng(int(seed))
    n_region, n_month = 6, 5
    region = rng.integers(0, n_region, n)
    month = rng.integers(0, n_month, n)
    # Build an interaction cell-mean table with (approximately) ZERO row and
    # column marginals so single-group means are non-predictive.
    cell = rng.uniform(-3.0, 3.0, (n_region, n_month))
    cell = cell - cell.mean(axis=1, keepdims=True)  # each region: month-mean 0
    cell = cell - cell.mean(axis=0, keepdims=True)  # each month: region-mean ~0
    cell_mean = cell[region, month]
    # x is noisy around its cell mean; raw x alone is weak.
    x = cell_mean + rng.normal(0.0, 3.0, n)
    y = (cell_mean + 0.1 * rng.normal(0.0, 1.0, n) > 0.0).astype(int)
    X = pd.DataFrame({
        "region": region,
        "month": month,
        "x": x,
        "noise_0": rng.normal(0.0, 1.0, n),
    })
    return X, y


def _build_single_group_signal(seed: int, n: int = 6000):
    """y driven by the per-REGION mean only (no month interaction). Used for the
    redundancy gate: a composite (region, month) mean here just re-expresses the
    region mean already carried by the single-group support.
    """
    rng = np.random.default_rng(int(seed))
    n_region, n_month = 8, 4
    region = rng.integers(0, n_region, n)
    month = rng.integers(0, n_month, n)
    region_mean = rng.uniform(-3.0, 3.0, n_region)
    x = region_mean[region] + rng.normal(0.0, 2.5, n)
    y = (region_mean[region] + 0.1 * rng.normal(0.0, 1.0, n) > 0.0).astype(int)
    X = pd.DataFrame({
        "region": region,
        "month": month,
        "x": x,
        "noise_0": rng.normal(0.0, 1.0, n),
    })
    return X, y


def _build_high_card_composite(seed: int, n: int = 4000):
    """A composite key whose cardinality blows past 0.5*n: two near-unique id
    columns. The cross is ~n distinct cells -> must be refused.
    """
    rng = np.random.default_rng(int(seed))
    id_a = rng.integers(0, n, n)        # ~unique
    id_b = rng.integers(0, n, n)        # ~unique
    x = rng.normal(0.0, 1.0, n)
    y = (x + rng.normal(0.0, 0.5, n) > 0.0).astype(int)
    X = pd.DataFrame({"id_a": id_a, "id_b": id_b, "x": x})
    return X, y


# ---------------------------------------------------------------------------
# Contract 1: composite-group signal recovered
# ---------------------------------------------------------------------------


class TestCompositeSignalRecovered:
    def test_composite_mean_recovers_interaction_signal(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            generate_composite_group_agg_features,
            engineered_name_composite_agg,
        )
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            score_grouped_agg_by_cmi_uplift,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_composite_signal(s)
            enc, raw = generate_composite_group_agg_features(
                X, [("region", "month")], ["x"], stats=("mean", "std", "count"),
            )
            e2s = {k: raw[k]["num_col"] for k in enc.columns}
            sc = score_grouped_agg_by_cmi_uplift(
                X, enc, y, ["x"], eng_to_source=e2s,
            )
            mean_name = engineered_name_composite_agg("x", ("region", "month"), "mean")
            row = sc[sc["engineered_col"] == mean_name].iloc[0]
            if row["cmi"] > 0.1 and row["uplift"] > 0.05:
                wins += 1
        assert wins >= 4, (
            f"composite-group mean recovered the interaction signal on only "
            f"{wins}/{len(SEEDS)} seeds; expected >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 2: single-group can't capture the composite interaction
# ---------------------------------------------------------------------------


class TestSingleGroupCannotCaptureComposite:
    def test_single_group_low_mi_composite_high_mi(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            generate_composite_group_agg_features,
            engineered_name_composite_agg,
        )
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            generate_grouped_agg_features,
            score_grouped_agg_by_cmi_uplift,
            engineered_name_grouped_agg,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_composite_signal(s)
            # Single-group means (region alone, month alone).
            enc_sg, raw_sg = generate_grouped_agg_features(
                X, ["region", "month"], ["x"], stats=("mean",),
            )
            e2s_sg = {k: raw_sg[k]["num_col"] for k in enc_sg.columns}
            sc_sg = score_grouped_agg_by_cmi_uplift(
                X, enc_sg, y, ["x"], eng_to_source=e2s_sg,
            )
            region_mean = engineered_name_grouped_agg("x", "region", "mean")
            month_mean = engineered_name_grouped_agg("x", "month", "mean")
            cmi_region = float(
                sc_sg[sc_sg["engineered_col"] == region_mean]["cmi"].iloc[0]
            )
            cmi_month = float(
                sc_sg[sc_sg["engineered_col"] == month_mean]["cmi"].iloc[0]
            )
            single_best = max(cmi_region, cmi_month)

            # Composite (region, month) mean.
            enc_c, raw_c = generate_composite_group_agg_features(
                X, [("region", "month")], ["x"], stats=("mean",),
            )
            e2s_c = {k: raw_c[k]["num_col"] for k in enc_c.columns}
            sc_c = score_grouped_agg_by_cmi_uplift(
                X, enc_c, y, ["x"], eng_to_source=e2s_c,
            )
            comp_name = engineered_name_composite_agg("x", ("region", "month"), "mean")
            cmi_comp = float(
                sc_c[sc_c["engineered_col"] == comp_name]["cmi"].iloc[0]
            )
            # Composite clearly beats the best single-group mean.
            if cmi_comp > 0.1 and cmi_comp > single_best + 0.05:
                wins += 1
        assert wins >= 4, (
            f"composite mean beat the best single-group mean on only "
            f"{wins}/{len(SEEDS)} seeds; expected >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 3: cardinality refusal
# ---------------------------------------------------------------------------


class TestCardinalityRefusal:
    def test_high_card_composite_refused(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            generate_composite_group_agg_features,
            auto_detect_key_sets,
            composite_cardinality_ok,
        )
        X, y = _build_high_card_composite(7)
        # The explicit (id_a, id_b) key explodes -> no columns emitted.
        enc, raw = generate_composite_group_agg_features(
            X, [("id_a", "id_b")], ["x"], stats=("mean", "count"),
            max_card_frac=0.5,
        )
        assert enc.shape[1] == 0, (
            f"high-cardinality composite was NOT refused: emitted {list(enc.columns)}"
        )
        # Auto-detect must also refuse it.
        sets = auto_detect_key_sets(X, max_arity=2, max_card_frac=0.5)
        assert ("id_a", "id_b") not in sets, (
            f"auto-detect surfaced the exploded key: {sets}"
        )
        # Guard helper behaves at the boundary.
        assert composite_cardinality_ok(10, 100, 0.5) is True
        assert composite_cardinality_ok(60, 100, 0.5) is False


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on composite fixture
# ---------------------------------------------------------------------------


class TestAucLift:
    def test_logreg_auc_lift_at_least_0p05(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            hybrid_composite_group_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        lifts = []
        for s in SEEDS:
            X, y = _build_composite_signal(s, n=9000)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            raw_cols = ["x", "noise_0", "region", "month"]
            base = LogisticRegression(max_iter=2000)
            base.fit(Xtr[raw_cols], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[raw_cols])[:, 1])

            X_aug_tr, appended, recipes, _ = hybrid_composite_group_agg_fe(
                Xtr, ytr, group_col_sets=[("region", "month")],
                num_cols=["x"], stats=("mean", "std", "count"), top_k=10,
            )
            assert appended, f"seed={s}: no composite-agg survivors for AUC test."
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
            f"composite-agg AUC lift {mean_lift:.4f} < 0.05 (per-seed "
            f"{[round(x, 4) for x in lifts]})."
        )


# ---------------------------------------------------------------------------
# Contract 5: CMI gate drops a redundant composite
# ---------------------------------------------------------------------------


class TestCmiGateDropsRedundant:
    def test_redundant_composite_gated_when_single_group_in_support(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            generate_composite_group_agg_features,
            engineered_name_composite_agg,
        )
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            generate_grouped_agg_features,
            score_grouped_agg_by_cmi_uplift,
            engineered_name_grouped_agg,
        )
        gated = 0
        for s in SEEDS:
            X, y = _build_single_group_signal(s)
            # The single-group region mean is the true (and only) signal; put it
            # in the conditioning support, then ask whether the composite
            # (region, month) mean adds NEW info on top of it.
            enc_sg, raw_sg = generate_grouped_agg_features(
                X, ["region"], ["x"], stats=("mean",),
            )
            region_mean_name = engineered_name_grouped_agg("x", "region", "mean")
            X_with_sg = X.copy()
            X_with_sg[region_mean_name] = enc_sg[region_mean_name].to_numpy()

            enc_c, raw_c = generate_composite_group_agg_features(
                X, [("region", "month")], ["x"], stats=("mean",),
            )
            e2s_c = {k: raw_c[k]["num_col"] for k in enc_c.columns}
            # Condition on x AND the single-group region mean.
            sc_c = score_grouped_agg_by_cmi_uplift(
                X_with_sg, enc_c, y, ["x", region_mean_name],
                eng_to_source=e2s_c,
            )
            comp_name = engineered_name_composite_agg("x", ("region", "month"), "mean")
            cmi_comp = float(
                sc_c[sc_c["engineered_col"] == comp_name]["cmi"].iloc[0]
            )
            # Once the region mean is in the support, the composite mean adds
            # ~nothing (the month split is pure noise here).
            if cmi_comp < 0.05:
                gated += 1
        assert gated >= 4, (
            f"redundant composite mean was NOT gated on "
            f"{len(SEEDS) - gated}/{len(SEEDS)} seeds; expected gating on >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 6: no leakage -- replay reads only X
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_replay_independent_of_y(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            hybrid_composite_group_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_composite_signal(7)
        _, appended, recipes, _ = hybrid_composite_group_agg_fe(
            X, y, group_col_sets=[("region", "month")], num_cols=["x"], top_k=10,
        )
        assert recipes, "no recipes produced for leakage test."
        rng = np.random.default_rng(0)
        y_shuffled = y[rng.permutation(len(y))]
        for r in recipes:
            col_a = apply_recipe(r, X)
            col_b = apply_recipe(r, X)
            np.testing.assert_array_equal(col_a, col_b)
            assert "y" not in dict(r.extra), (
                f"recipe {r.name!r} captured a y reference -- leakage risk."
            )
        # Refit on shuffled y -> recipe lookups (functions of X only) replay to
        # the SAME columns for the SAME engineered names.
        _, _, recipes_shuf, _ = hybrid_composite_group_agg_fe(
            X, y_shuffled, group_col_sets=[("region", "month")],
            num_cols=["x"], top_k=10,
        )
        by_name = {r.name: r for r in recipes}
        for r2 in recipes_shuf:
            if r2.name in by_name:
                np.testing.assert_array_equal(
                    apply_recipe(r2, X), apply_recipe(by_name[r2.name], X),
                )


# ---------------------------------------------------------------------------
# Contract 7: default disabled byte-identical / enabled adds columns
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_does_not_add_composite_agg(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_composite_signal(42, n=2500)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_composite_group_agg_enable", False)) is False, (
            "fe_composite_group_agg_enable must default to False."
        )
        m.fit(X, pd.Series(y, name="y"))
        cga = list(getattr(m, "composite_group_agg_features_", []) or [])
        assert cga == [], (
            f"composite_group_agg added columns with the feature disabled: {cga}"
        )

    def test_mrmr_enabled_adds_composite_agg(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_composite_signal(42, n=4000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_composite_group_agg_enable=True,
            fe_composite_group_agg_key_sets=(("region", "month"),),
            fe_composite_group_agg_num_cols=("x",),
            fe_composite_group_agg_stats=("mean", "std", "count"),
            fe_composite_group_agg_top_k=5,
        )
        m.fit(X, pd.Series(y, name="y"))
        cga = list(getattr(m, "composite_group_agg_features_", []) or [])
        assert len(cga) >= 1, (
            "composite_group_agg enabled but produced no engineered columns on "
            "the composite-signal fixture."
        )


# ---------------------------------------------------------------------------
# Contract 8: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            hybrid_composite_group_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_composite_signal(1)
        _, appended, recipes, _ = hybrid_composite_group_agg_fe(
            X, y, group_col_sets=[("region", "month")], num_cols=["x"], top_k=10,
        )
        assert recipes, "no recipes for pickle test."
        for r in recipes:
            blob = pickle.dumps(r)
            r2 = pickle.loads(blob)
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_mrmr_clone_preserves_params(self):
        from sklearn.base import clone
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_composite_group_agg_enable=True,
            fe_composite_group_agg_key_sets=(("region", "month"),),
            fe_composite_group_agg_num_cols=("x",),
            fe_composite_group_agg_max_arity=3,
            fe_composite_group_agg_top_k=7,
        )
        c = clone(m)
        assert bool(c.fe_composite_group_agg_enable) is True
        assert tuple(c.fe_composite_group_agg_key_sets) == (("region", "month"),)
        assert tuple(c.fe_composite_group_agg_num_cols) == ("x",)
        assert int(c.fe_composite_group_agg_max_arity) == 3
        assert int(c.fe_composite_group_agg_top_k) == 7


class TestMeanStdReuseBitIdentical:
    """The mean/std stat broadcasts reuse the per-group series already materialised for the
    z / ratio residuals rather than re-running ``grouped.agg``. Pin that the reused values are
    bit-identical to an independent explicit ``grouped.agg("mean"|"std")`` reference, across
    adversarial magnitudes, so a future 'just recompute it' cannot silently drift the encoding."""

    def test_mean_std_broadcast_matches_explicit_groupby_agg(self):
        from mlframe.feature_selection.filters._composite_group_agg_fe import (
            build_composite_keys,
            engineered_name_composite_agg,
            generate_composite_group_agg_features,
        )
        rng = np.random.default_rng(11)
        for trial in range(5):
            n = int(rng.integers(2000, 8000))
            scale = float(rng.uniform(0.1, 50.0)) * (10.0 ** rng.integers(-6, 7))
            X = pd.DataFrame(
                {
                    "a": rng.integers(0, 15, n),
                    "b": rng.integers(0, 9, n),
                    "v": rng.normal(0.0, scale, n),
                }
            )
            gset = ("a", "b")
            enc, _ = generate_composite_group_agg_features(
                X, [gset], num_cols=["v"], stats=("mean", "std", "count"),
            )
            keys = build_composite_keys(X, gset)
            grouped = pd.DataFrame({"_g": keys, "_v": X["v"].to_numpy(dtype=float)}).groupby(
                "_g", observed=True, sort=False,
            )["_v"]
            ref_mean = {str(k): float(v) for k, v in grouped.agg("mean").items()}
            ref_std = {
                str(k): (float(v) if np.isfinite(v) else 0.0)
                for k, v in grouped.agg("std").items()
            }
            keys_str = np.array([str(k) for k in keys], dtype=object)
            exp_mean = np.array([ref_mean[k] for k in keys_str], dtype=np.float64)
            exp_std = np.array([ref_std[k] for k in keys_str], dtype=np.float64)
            got_mean = enc[engineered_name_composite_agg("v", gset, "mean")].to_numpy()
            got_std = enc[engineered_name_composite_agg("v", gset, "std")].to_numpy()
            assert np.max(np.abs(got_mean - exp_mean)) == 0.0, f"trial {trial}: mean drift"
            assert np.max(np.abs(got_std - exp_std)) == 0.0, f"trial {trial}: std drift"
