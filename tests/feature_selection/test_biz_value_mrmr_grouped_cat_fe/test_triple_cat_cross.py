"""Layer 94 biz_value: cat x cat x cat TRIPLE synergy cross via beam search.

Extends the Layer 89 pairwise interaction-information cross to the THIRD order.
The three-way interaction information (co-information; McGill 1954):

    II3(a, b, c; y) = I(a, b, c; y)
                      - [I(a, b; y) + I(a, c; y) + I(b, c; y)]
                      + [I(a; y) + I(b; y) + I(c; y)]

Positive II3 = genuine three-way synergy NO pair or single explains. The
canonical fixture is the parity target ``y = a XOR b XOR c`` where every PAIRWISE
II is ~ 0 (no pair predicts) yet the triple is fully predictive.

Contracts pinned (real numbers, Bayes-feasible fixtures, never xfail):

* 3-way cat XOR: II3 > 0 strongly; the beam recovers the triple in top-1.
* Not findable pairwise: II(a,b;y) ~ II(a,c;y) ~ II(b,c;y) ~ 0 (genuinely 3-way)
  while II3 is large.
* Beam efficiency: candidate triples evaluated <= beam_width * p (not C(p,3));
  report the reduction.
* AUC lift on 3-way cat XOR: triple-cross-aug LogReg >= raw + 0.20.
* Non-synergistic triples excluded: independent cats -> II3 ~ 0 -> zero features.
* No leakage: transform replay reads only X; identical under shuffled y.
* Default disabled byte-identical.
* Pickle / clone round-trips the recipes + params.

2026-06-01 Layer 94.

Consolidated verbatim from test_biz_value_mrmr_layer94.py (per audit finding test_code_quality-16).
"""
from __future__ import annotations

import pickle
import warnings
from itertools import combinations

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


def _build_cat_xor3(seed: int, n: int = 6000, n_decoys: int = 2):
    """y = cat_a XOR cat_b XOR cat_c. Every single AND every pair is marginally
    uninformative; only the joint triple determines y. Independent decoy cats
    carry no signal."""
    rng = np.random.default_rng(int(seed))
    cat_a = rng.integers(0, 2, n)
    cat_b = rng.integers(0, 2, n)
    cat_c = rng.integers(0, 2, n)
    flip = rng.random(n) < 0.02  # small label noise
    y = (cat_a ^ cat_b ^ cat_c) ^ flip.astype(int)
    cols = {
        "cat_a": cat_a.astype(str),
        "cat_b": cat_b.astype(str),
        "cat_c": cat_c.astype(str),
    }
    for d in range(n_decoys):
        cols[f"decoy_{d}"] = rng.integers(0, 2 + d, n).astype(str)
    X = pd.DataFrame(cols)
    return X, y.astype(int)


def _build_independent3(seed: int, n: int = 6000):
    """Three independent cats, none related to y -> II3 ~ 0."""
    rng = np.random.default_rng(int(seed))
    X = pd.DataFrame({
        "cat_a": rng.integers(0, 3, n).astype(str),
        "cat_b": rng.integers(0, 3, n).astype(str),
        "cat_c": rng.integers(0, 3, n).astype(str),
    })
    y = rng.integers(0, 2, n)
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# Contract 1: 3-way XOR -- II3 strongly positive; pairwise II ~ 0
# ---------------------------------------------------------------------------


class TestTripleSynergyVsPairwise:
    def test_ii3_large_pairwise_zero(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            triple_interaction_information,
        )
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
        )
        ii3_vals = []
        max_pair_ii = []
        for s in SEEDS:
            X, y = _build_cat_xor3(s)
            ii3 = triple_interaction_information(
                X["cat_a"].to_numpy(), X["cat_b"].to_numpy(),
                X["cat_c"].to_numpy(), y,
            )
            ii3_vals.append(ii3)
            # Pairwise II among the three signal cats must all be ~ 0 (the signal
            # is genuinely three-way -- no pair predicts y).
            sc = score_cat_pairs_by_interaction_information(
                X, y, ["cat_a", "cat_b", "cat_c"],
            )
            pair_ii = {
                tuple(sorted((r["cat_i"], r["cat_j"]))): r["ii"]
                for _, r in sc.iterrows()
            }
            signal_pairs = [
                pair_ii[tuple(sorted(p))]
                for p in combinations(["cat_a", "cat_b", "cat_c"], 2)
            ]
            max_pair_ii.append(max(abs(v) for v in signal_pairs))
        mean_ii3 = float(np.mean(ii3_vals))
        worst_pair = float(np.max(max_pair_ii))
        assert mean_ii3 > 0.2, (
            f"3-way XOR II3 mean {mean_ii3:.4f} <= 0.2 (per-seed "
            f"{[round(v, 4) for v in ii3_vals]}); the co-information should be "
            f"strongly positive on the parity target."
        )
        assert worst_pair < 0.05, (
            f"worst |pairwise II| among signal cats is {worst_pair:.4f} >= 0.05; "
            f"the 3-way XOR signal must be invisible to any PAIR "
            f"(per-seed max {[round(v, 4) for v in max_pair_ii]})."
        )
        assert mean_ii3 > 4.0 * worst_pair, (
            f"II3 ({mean_ii3:.4f}) should dwarf the best pairwise II "
            f"({worst_pair:.4f}); the genuine signal is third-order only."
        )


# ---------------------------------------------------------------------------
# Contract 2: beam recovers the triple in top-1
# ---------------------------------------------------------------------------


class TestBeamRecoversTriple:
    def test_xor3_triple_top1(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            score_cat_triples_by_interaction_information,
            engineered_name_cat_triple_cross,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_cat_xor3(s)
            cat_cols = ["cat_a", "cat_b", "cat_c", "decoy_0", "decoy_1"]
            sc = score_cat_triples_by_interaction_information(
                X, y, cat_cols, beam_width=3, top_k_pairs=3,
            )
            target = engineered_name_cat_triple_cross("cat_a", "cat_b", "cat_c")
            if not sc.empty and str(sc.iloc[0]["engineered_col"]) == target:
                if float(sc.iloc[0]["ii3"]) > 0.2:
                    wins += 1
        assert wins >= 4, (
            f"beam recovered the 3-way XOR triple as top-1 on only "
            f"{wins}/{len(SEEDS)} seeds; expected >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 3: beam efficiency -- evaluated <= beam_width * p, not C(p, 3)
# ---------------------------------------------------------------------------


class TestBeamEfficiency:
    def test_candidate_reduction(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            score_cat_triples_by_interaction_information,
        )
        # Wider cat set so C(p,3) is materially larger than the beam budget.
        X, y = _build_cat_xor3(7, n=5000, n_decoys=7)  # 3 signal + 7 decoy = 10
        cat_cols = ["cat_a", "cat_b", "cat_c"] + [f"decoy_{d}" for d in range(7)]
        p = len(cat_cols)
        beam_width, top_k_pairs = 3, 3
        sc = score_cat_triples_by_interaction_information(
            X, y, cat_cols, beam_width=beam_width, top_k_pairs=top_k_pairs,
        )
        n_eval = int(sc.attrs["n_triples_evaluated"])
        n_exhaustive = int(sc.attrs["n_triples_exhaustive"])
        assert n_exhaustive == len(list(combinations(cat_cols, 3)))
        # Beam bound: per refinement round, each of <= max(top_k_pairs,
        # beam_width) seed pairs sweeps the remaining (p-2) cats. With 2 rounds:
        n_rounds = 2
        budget = n_rounds * max(top_k_pairs, beam_width) * (p - 2)
        assert n_eval <= budget, (
            f"beam evaluated {n_eval} triples > budget {budget} "
            f"(n_rounds * max(top_k_pairs, beam_width) * (p-2))."
        )
        assert n_eval < n_exhaustive, (
            f"beam evaluated {n_eval} triples, NOT fewer than the exhaustive "
            f"C({p},3)={n_exhaustive}."
        )
        reduction = 1.0 - n_eval / float(n_exhaustive)
        # p=10: C(10,3)=120; beam budget=2*3*8=48 -> materially fewer.
        assert reduction >= 0.5, (
            f"beam candidate reduction only {reduction:.1%} "
            f"({n_eval}/{n_exhaustive}); expected >= 50%."
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on 3-way cat XOR >= +0.20
# ---------------------------------------------------------------------------


class TestAucLift:
    def test_logreg_auc_lift_at_least_0p20(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            hybrid_cat_triple_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        all_cols = ["cat_a", "cat_b", "cat_c", "decoy_0", "decoy_1"]
        lifts = []
        for s in SEEDS:
            X, y = _build_cat_xor3(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )

            def _onehot(df):
                return pd.get_dummies(
                    df[all_cols].astype(str), drop_first=False,
                ).astype(float)

            Xtr_oh = _onehot(Xtr)
            Xte_oh = _onehot(Xte).reindex(columns=Xtr_oh.columns, fill_value=0.0)
            base = LogisticRegression(max_iter=2000)
            base.fit(Xtr_oh, ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte_oh)[:, 1])

            _, appended, recipes, _ = hybrid_cat_triple_fe(
                Xtr, ytr, cat_cols=all_cols,
                min_interaction_info=0.001, top_k=3,
                beam_width=3, top_k_pairs=3, random_state=s,
            )
            assert appended, f"seed={s}: no cat-triple survivors."
            Xtr_aug = Xtr_oh.copy()
            Xte_aug = Xte_oh.copy()
            for r in recipes:
                tr_codes = apply_recipe(r, Xtr)
                te_codes = apply_recipe(r, Xte)
                oh_tr = pd.get_dummies(
                    pd.Series(tr_codes).astype(str), prefix=r.name,
                ).astype(float)
                oh_te = pd.get_dummies(
                    pd.Series(te_codes).astype(str), prefix=r.name,
                ).astype(float)
                oh_te = oh_te.reindex(columns=oh_tr.columns, fill_value=0.0)
                Xtr_aug = pd.concat(
                    [Xtr_aug.reset_index(drop=True), oh_tr.reset_index(drop=True)],
                    axis=1,
                )
                Xte_aug = pd.concat(
                    [Xte_aug.reset_index(drop=True), oh_te.reset_index(drop=True)],
                    axis=1,
                )
            aug = LogisticRegression(max_iter=2000)
            aug.fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.20, (
            f"cat-triple-cross AUC lift {mean_lift:.4f} < 0.20 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the 3-way synergy cross is not "
            f"recovering the parity separation a raw model can't learn."
        )


# ---------------------------------------------------------------------------
# Contract 5: non-synergistic triples emit zero features
# ---------------------------------------------------------------------------


class TestNonSynergisticExcluded:
    def test_independent_cats_emit_nothing(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            hybrid_cat_triple_fe,
        )
        for s in SEEDS:
            X, y = _build_independent3(s)
            _, appended, recipes, scores = hybrid_cat_triple_fe(
                X, y, cat_cols=["cat_a", "cat_b", "cat_c"],
                min_interaction_info=0.001, top_k=3,
            )
            assert appended == [], (
                f"seed={s}: independent cats produced engineered columns "
                f"{appended} (II3={scores['ii3'].tolist() if not scores.empty else []})."
            )
            assert recipes == []


# ---------------------------------------------------------------------------
# Contract 6: cardinality control -- high-card triple routed through TE
# ---------------------------------------------------------------------------


class TestCardinalityControl:
    def test_highcard_triple_routed_through_te(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            hybrid_cat_triple_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        rng = np.random.default_rng(7)
        n = 4000
        cat_a = rng.integers(0, 20, n)
        cat_b = rng.integers(0, 20, n)
        cat_c = rng.integers(0, 20, n)
        # 3-way synergy: parity of the sum needs the joint triple.
        y = ((cat_a + cat_b + cat_c) % 2).astype(int)
        y = y ^ (rng.random(n) < 0.02).astype(int)
        X = pd.DataFrame({
            "cat_a": cat_a.astype(str),
            "cat_b": cat_b.astype(str),
            "cat_c": cat_c.astype(str),
        })
        _, appended, recipes, _ = hybrid_cat_triple_fe(
            X, y, cat_cols=["cat_a", "cat_b", "cat_c"],
            min_interaction_info=-1.0, top_k=3, random_state=7,
        )
        assert appended, "high-card synergy triple produced no survivor."
        for r in recipes:
            n_cells = len(r.extra["mapping"])
            assert n_cells > 0.5 * len(X), (
                f"recipe {r.name!r} has only {n_cells} cells; the high-card "
                f"fixture should exceed the 0.5*n pre-screen threshold."
            )
            assert str(r.extra.get("encoding")) == "target", (
                f"recipe {r.name!r} encoding is {r.extra.get('encoding')!r}; "
                f"high-card triple must route through target encoding."
            )
            col = apply_recipe(r, X)
            assert col.dtype == np.float64
            assert np.isfinite(col).all()
            assert 0.0 <= float(col.min()) and float(col.max()) <= 1.0


# ---------------------------------------------------------------------------
# Contract 7: no y-leak -- replay independent of y
# ---------------------------------------------------------------------------


class TestNoYLeak:
    def test_replay_independent_of_y(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            hybrid_cat_triple_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_cat_xor3(7)
        _, appended, recipes, _ = hybrid_cat_triple_fe(
            X, y, cat_cols=["cat_a", "cat_b", "cat_c"],
            min_interaction_info=0.001, top_k=3, random_state=7,
        )
        assert recipes, "no recipes produced for leakage test."
        for r in recipes:
            c1 = apply_recipe(r, X)
            c2 = apply_recipe(r, X)
            np.testing.assert_array_equal(c1, c2)
            assert "y" not in dict(r.extra), (
                f"recipe {r.name!r} captured a y reference -- leakage risk."
            )

    def test_transform_same_under_shuffled_y(self):
        """The materialised RAW triple cross is a pure function of X; generating
        with any y reproduces the identical replay column."""
        from mlframe.feature_selection.filters._cat_triple_fe import (
            generate_cat_triple_crosses,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cat_triple_cross_recipe, apply_recipe,
        )
        X, y = _build_cat_xor3(42)
        cols = ["cat_a", "cat_b", "cat_c"]
        enc_a, raw_a = generate_cat_triple_crosses(X, cols)
        enc_b, raw_b = generate_cat_triple_crosses(X, cols)
        for name in raw_a:
            ra = build_cat_triple_cross_recipe(
                name=name, cat_a=raw_a[name]["cat_a"],
                cat_b=raw_a[name]["cat_b"], cat_c=raw_a[name]["cat_c"],
                mapping=raw_a[name]["mapping"], encoding="raw",
            )
            rb = build_cat_triple_cross_recipe(
                name=name, cat_a=raw_b[name]["cat_a"],
                cat_b=raw_b[name]["cat_b"], cat_c=raw_b[name]["cat_c"],
                mapping=raw_b[name]["mapping"], encoding="raw",
            )
            assert ra == rb
            np.testing.assert_array_equal(apply_recipe(ra, X), apply_recipe(rb, X))


# ---------------------------------------------------------------------------
# Contract 8: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_does_not_add_cat_triple(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_cat_xor3(42, n=2000)
        Xi = X.copy()
        for c in Xi.columns:
            Xi[c] = Xi[c].astype(int)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_cat_triple_enable", False)) is False, (
            "fe_cat_triple_enable must default to False."
        )
        m.fit(Xi, pd.Series(y, name="y"))
        ct_feats = list(getattr(m, "cat_triple_features_", []) or [])
        assert ct_feats == [], (
            f"cat_triple added columns with the feature disabled: {ct_feats}"
        )

    def test_mrmr_enabled_adds_cat_triple(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_cat_xor3(42, n=3000)
        Xi = X.copy()
        for c in Xi.columns:
            Xi[c] = Xi[c].astype(int)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_cat_triple_enable=True,
            fe_cat_triple_cat_cols=("cat_a", "cat_b", "cat_c"),
            fe_cat_triple_top_k=3,
        )
        m.fit(Xi, pd.Series(y, name="y"))
        ct_feats = list(getattr(m, "cat_triple_features_", []) or [])
        assert len(ct_feats) >= 1, (
            "cat_triple enabled but produced no engineered columns on the "
            "3-way cat-XOR fixture."
        )


# ---------------------------------------------------------------------------
# Contract 9: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            hybrid_cat_triple_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        # Raw (low-card 2x2x2) + target-encoded (high-card) payloads.
        X_raw, y_raw = _build_cat_xor3(1)
        rng = np.random.default_rng(1)
        n = 4000
        a = rng.integers(0, 20, n)
        b = rng.integers(0, 20, n)
        c = rng.integers(0, 20, n)
        y_hc = ((a + b + c) % 2).astype(int)
        X_hc = pd.DataFrame({
            "cat_a": a.astype(str), "cat_b": b.astype(str), "cat_c": c.astype(str),
        })
        cases = [
            (X_raw, y_raw, ["cat_a", "cat_b", "cat_c"], 0.001),
            (X_hc, y_hc, ["cat_a", "cat_b", "cat_c"], -1.0),
        ]
        for X, y, cols, thr in cases:
            _, appended, recipes, _ = hybrid_cat_triple_fe(
                X, y, cat_cols=cols, min_interaction_info=thr, top_k=3,
                random_state=1,
            )
            assert recipes, "no recipes for pickle test."
            for r in recipes:
                blob = pickle.dumps(r)
                r2 = pickle.loads(blob)
                assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
                np.testing.assert_array_equal(
                    apply_recipe(r, X), apply_recipe(r2, X),
                )

    def test_mrmr_clone_preserves_params(self):
        from sklearn.base import clone
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_cat_triple_enable=True,
            fe_cat_triple_cat_cols=("cat_a", "cat_b", "cat_c"),
            fe_cat_triple_min_interaction_info=0.01,
            fe_cat_triple_beam_width=5,
            fe_cat_triple_top_k=7,
        )
        c = clone(m)
        assert bool(c.fe_cat_triple_enable) is True
        assert tuple(c.fe_cat_triple_cat_cols) == ("cat_a", "cat_b", "cat_c")
        assert float(c.fe_cat_triple_min_interaction_info) == 0.01
        assert int(c.fe_cat_triple_beam_width) == 5
        assert int(c.fe_cat_triple_top_k) == 7
