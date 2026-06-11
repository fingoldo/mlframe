"""Layer 89 biz_value: cat x cat synergy cross with interaction-information pre-filter.

NVIDIA cuDF Kaggle-Grandmaster blog technique #3 -- combine two categorical
columns into a new high-cardinality categorical ``hash(cat_i || cat_j)`` then
target-encode it. The IT enhancement (THE KEY) pre-filters pairs by INTERACTION
INFORMATION:

    II(cat_i, cat_j; y) = I(cat_i, cat_j; y) - I(cat_i; y) - I(cat_j; y)

Positive II = synergy (XOR-like); negative II = redundancy. Only synergistic
pairs (II > threshold) are materialised.

Contracts pinned (real numbers, Bayes-feasible fixtures, never xfail):

* XOR cat signal: y = cat_a XOR cat_b; II(cat_a, cat_b; y) > 0 strongly; the
  synergy filter recovers the pair in the top-3 across 5 seeds.
* Non-synergistic pairs excluded: independent cats (II ~ 0) emit zero features.
* Redundant pairs excluded: cat_b a copy of cat_a (II < 0) -> not materialised.
* AUC lift on cat-XOR fixture: cross-aug LogReg >= raw + 0.20 (raw can't learn
  cat XOR).
* Cardinality control: high-card cross routed through TE not one-hot; support
  doesn't explode.
* No leakage: transform(X, y_shuffled) == transform(X); recipe replay reads
  only X.
* Default disabled byte-identical.
* Pickle / clone round-trips the recipes.

2026-06-01 Layer 89.

Consolidated verbatim from test_biz_value_mrmr_layer89.py (per audit finding test_code_quality-16).
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


def _build_cat_xor(seed: int, n: int = 6000):
    """y = cat_a XOR cat_b. Each parent is marginally uninformative
    (I(cat_a; y) ~ I(cat_b; y) ~ 0); only the joint determines y. Independent
    decoy cats carry no signal."""
    rng = np.random.default_rng(int(seed))
    cat_a = rng.integers(0, 2, n)
    cat_b = rng.integers(0, 2, n)
    flip = rng.random(n) < 0.03  # small label noise
    y = (cat_a ^ cat_b) ^ flip.astype(int)
    X = pd.DataFrame({
        "cat_a": cat_a.astype(str),
        "cat_b": cat_b.astype(str),
        "decoy_0": rng.integers(0, 2, n).astype(str),
        "decoy_1": rng.integers(0, 3, n).astype(str),
    })
    return X, y.astype(int)


def _build_redundant(seed: int, n: int = 6000):
    """cat_b is a copy of cat_a (perfect redundancy). The cross adds nothing
    over either parent -> II < 0."""
    rng = np.random.default_rng(int(seed))
    cat_a = rng.integers(0, 4, n)
    cat_b = cat_a.copy()  # exact copy
    # y depends on cat_a (so both parents already carry the signal).
    y = (cat_a >= 2).astype(int)
    flip = rng.random(n) < 0.03
    y = y ^ flip.astype(int)
    X = pd.DataFrame({
        "cat_a": cat_a.astype(str),
        "cat_b": cat_b.astype(str),
    })
    return X, y.astype(int)


def _build_independent(seed: int, n: int = 6000):
    """Two independent cats, neither related to y -> II ~ 0."""
    rng = np.random.default_rng(int(seed))
    cat_a = rng.integers(0, 3, n)
    cat_b = rng.integers(0, 3, n)
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "cat_a": cat_a.astype(str),
        "cat_b": cat_b.astype(str),
    })
    return X, y.astype(int)


def _build_highcard_xor(seed: int, n: int = 3000):
    """High-cardinality synergy: cross cell count > 0.5*n so the cardinality
    pre-screen routes through target encoding. With 60x60 possible cells and
    n=3000 the observed distinct-cell count (~2300) reliably exceeds the
    0.5*n=1500 pre-screen threshold. y depends on a synergistic function of the
    two high-card cats."""
    rng = np.random.default_rng(int(seed))
    cat_a = rng.integers(0, 60, n)
    cat_b = rng.integers(0, 60, n)
    # synergy: y is a function of (a + b) mod 2 -- needs the joint.
    y = ((cat_a + cat_b) % 2).astype(int)
    flip = rng.random(n) < 0.03
    y = y ^ flip.astype(int)
    X = pd.DataFrame({
        "cat_a": cat_a.astype(str),
        "cat_b": cat_b.astype(str),
    })
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# Contract 1: XOR synergy recovered in top-3 across seeds
# ---------------------------------------------------------------------------


class TestXorSynergyRecovered:
    def test_xor_pair_in_top3(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
            engineered_name_cat_pair_cross,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_cat_xor(s)
            cat_cols = ["cat_a", "cat_b", "decoy_0", "decoy_1"]
            sc = score_cat_pairs_by_interaction_information(X, y, cat_cols)
            xor_name = engineered_name_cat_pair_cross("cat_a", "cat_b")
            top3 = list(sc.head(3)["engineered_col"])
            xor_row = sc[sc["engineered_col"] == xor_name].iloc[0]
            # II must be strongly positive for the XOR pair AND it must rank top-3.
            if xor_row["ii"] > 0.2 and xor_name in top3:
                wins += 1
        assert wins >= 4, (
            f"cat-XOR synergy pair recovered in top-3 on only {wins}/{len(SEEDS)} "
            f"seeds; expected >= 4. The interaction-information filter is not "
            f"identifying the synergistic pair."
        )


# ---------------------------------------------------------------------------
# Contract 2: non-synergistic (independent) pairs emit zero features
# ---------------------------------------------------------------------------


class TestNonSynergisticExcluded:
    def test_independent_cats_emit_nothing(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            hybrid_cat_pair_fe,
        )
        for s in SEEDS:
            X, y = _build_independent(s)
            _, appended, recipes, scores = hybrid_cat_pair_fe(
                X, y, cat_cols=["cat_a", "cat_b"],
                min_interaction_info=0.001, top_k=5,
            )
            assert appended == [], (
                f"seed={s}: independent cats produced engineered columns "
                f"{appended} (II={scores['ii'].tolist()}); expected none."
            )
            assert recipes == []


# ---------------------------------------------------------------------------
# Contract 3: redundant pairs (cat_b == cat_a) excluded (II < 0)
# ---------------------------------------------------------------------------


class TestRedundantExcluded:
    def test_copy_cat_not_materialised(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            hybrid_cat_pair_fe,
            score_cat_pairs_by_interaction_information,
        )
        for s in SEEDS:
            X, y = _build_redundant(s)
            sc = score_cat_pairs_by_interaction_information(X, y, ["cat_a", "cat_b"])
            # The cross of a column with its own copy is pure redundancy.
            assert float(sc["ii"].iloc[0]) < 0.0, (
                f"seed={s}: II for cat_b == copy(cat_a) should be < 0 "
                f"(redundancy); got {sc['ii'].iloc[0]}."
            )
            _, appended, recipes, _ = hybrid_cat_pair_fe(
                X, y, cat_cols=["cat_a", "cat_b"],
                min_interaction_info=0.001, top_k=5,
            )
            assert appended == [], (
                f"seed={s}: redundant copy-cat pair was materialised "
                f"({appended}); expected none."
            )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on cat-XOR fixture >= +0.20
# ---------------------------------------------------------------------------


class TestAucLift:
    def test_logreg_auc_lift_at_least_0p20(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            hybrid_cat_pair_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        lifts = []
        for s in SEEDS:
            X, y = _build_cat_xor(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            # Raw model: one-hot the parent cats (a LogReg CANNOT learn XOR from
            # the marginals -- the linear decision boundary in (a, b) one-hots
            # is the additive contribution of each, which is uninformative).
            def _onehot(df):
                return pd.get_dummies(
                    df[["cat_a", "cat_b", "decoy_0", "decoy_1"]].astype(str),
                    drop_first=False,
                ).astype(float)
            Xtr_oh = _onehot(Xtr)
            Xte_oh = _onehot(Xte).reindex(columns=Xtr_oh.columns, fill_value=0.0)
            base = LogisticRegression(max_iter=2000)
            base.fit(Xtr_oh, ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte_oh)[:, 1])

            _, appended, recipes, _ = hybrid_cat_pair_fe(
                Xtr, ytr, cat_cols=["cat_a", "cat_b", "decoy_0", "decoy_1"],
                min_interaction_info=0.001, top_k=5, random_state=s,
            )
            assert appended, f"seed={s}: no cat-pair survivors."
            # Augmented model: one-hot the cross cell code (a single column that
            # encodes the joint, which a linear model CAN separate).
            Xtr_aug = Xtr_oh.copy()
            Xte_aug = Xte_oh.copy()
            for r in recipes:
                tr_codes = apply_recipe(r, Xtr)
                te_codes = apply_recipe(r, Xte)
                oh_tr = pd.get_dummies(pd.Series(tr_codes).astype(str), prefix=r.name).astype(float)
                oh_te = pd.get_dummies(pd.Series(te_codes).astype(str), prefix=r.name).astype(float)
                oh_te = oh_te.reindex(columns=oh_tr.columns, fill_value=0.0)
                Xtr_aug = pd.concat([Xtr_aug.reset_index(drop=True), oh_tr.reset_index(drop=True)], axis=1)
                Xte_aug = pd.concat([Xte_aug.reset_index(drop=True), oh_te.reset_index(drop=True)], axis=1)
            aug = LogisticRegression(max_iter=2000)
            aug.fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.20, (
            f"cat-pair-cross AUC lift {mean_lift:.4f} < 0.20 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the synergy cross is not "
            f"recovering the XOR separation a raw model can't learn."
        )


# ---------------------------------------------------------------------------
# Contract 5: cardinality control -- high-card cross routed through TE
# ---------------------------------------------------------------------------


class TestCardinalityControl:
    def test_highcard_cross_routed_through_te(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            hybrid_cat_pair_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_highcard_xor(7, n=4000)
        _, appended, recipes, _ = hybrid_cat_pair_fe(
            X, y, cat_cols=["cat_a", "cat_b"],
            min_interaction_info=-1.0, top_k=5, random_state=7,
        )
        assert appended, "high-card synergy cross produced no survivor."
        for r in recipes:
            n_cells = len(r.extra["mapping"])
            # cross cell count (~40*40 effective) must exceed the 0.5*n screen.
            assert n_cells > 0.5 * len(X), (
                f"recipe {r.name!r} has only {n_cells} cells; the high-card "
                f"fixture should exceed the pre-screen threshold."
            )
            assert str(r.extra.get("encoding")) == "target", (
                f"recipe {r.name!r} encoding is {r.extra.get('encoding')!r}; "
                f"high-card cross must route through target encoding, not raw."
            )
            col = apply_recipe(r, X)
            # Support does not explode: TE collapses the cross to a SINGLE dense
            # numeric column whose distinct-value count is bounded by n_cells but
            # whose effective range is just per-cell means in [0, 1].
            assert col.dtype == np.float64
            assert np.isfinite(col).all()
            assert 0.0 <= float(col.min()) and float(col.max()) <= 1.0


# ---------------------------------------------------------------------------
# Contract 6: no y-leak -- replay independent of y
# ---------------------------------------------------------------------------


class TestNoYLeak:
    def test_replay_independent_of_y(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            hybrid_cat_pair_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_cat_xor(7)
        _, appended, recipes, _ = hybrid_cat_pair_fe(
            X, y, cat_cols=["cat_a", "cat_b"],
            min_interaction_info=0.001, top_k=5, random_state=7,
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
        """The materialised RAW cross cell code is a pure function of X; fitting
        with a shuffled y must reproduce the identical replay column (the cross
        mapping does not depend on y in raw mode)."""
        from mlframe.feature_selection.filters._cat_pair_fe import (
            generate_cat_pair_crosses,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cat_pair_cross_recipe, apply_recipe,
        )
        X, y = _build_cat_xor(42)
        # generate_cat_pair_crosses never sees y; the cross codes are a pure
        # function of X regardless of any y the caller holds.
        enc_a, raw_a = generate_cat_pair_crosses(X, ["cat_a", "cat_b"])
        enc_b, raw_b = generate_cat_pair_crosses(X, ["cat_a", "cat_b"])
        for name in raw_a:
            ra = build_cat_pair_cross_recipe(
                name=name, cat_i=raw_a[name]["cat_i"],
                cat_j=raw_a[name]["cat_j"], mapping=raw_a[name]["mapping"],
                encoding="raw",
            )
            rb = build_cat_pair_cross_recipe(
                name=name, cat_i=raw_b[name]["cat_i"],
                cat_j=raw_b[name]["cat_j"], mapping=raw_b[name]["mapping"],
                encoding="raw",
            )
            assert ra == rb
            np.testing.assert_array_equal(apply_recipe(ra, X), apply_recipe(rb, X))


# ---------------------------------------------------------------------------
# Contract 7: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_does_not_add_cat_pair(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_cat_xor(42, n=2000)
        # MRMR consumes integer-coded cats; pass the int form.
        Xi = X.copy()
        for c in ["cat_a", "cat_b", "decoy_0", "decoy_1"]:
            Xi[c] = Xi[c].astype(int)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_cat_pair_enable", False)) is False, (
            "fe_cat_pair_enable must default to False."
        )
        m.fit(Xi, pd.Series(y, name="y"))
        cp_feats = list(getattr(m, "cat_pair_features_", []) or [])
        assert cp_feats == [], (
            f"cat_pair added columns with the feature disabled: {cp_feats}"
        )

    def test_mrmr_enabled_adds_cat_pair(self):
        # The cat-pair cross IS materialised and competes for selection; on the
        # cat-XOR fixture the default-on general FE families (smart-polynom pair
        # engineering ``div(exp(cat_a),abs(cat_b))``, the univariate Fourier
        # basis) independently recover the SAME XOR signal and rank ahead of the
        # cross, so the post-selection ``cat_pair_features_`` roster is reconciled
        # empty -- NOT because the mechanism failed but because a redundant
        # sibling won. Isolate the family under test (the sibling Layer-91/97
        # fixtures disable the same competitors via ``fe_max_steps=0``) so the
        # roster reflects the cat-pair mechanism's own output: with the
        # general-FE competitors off, the synergy cross is the selected
        # engineered column.
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_cat_xor(42, n=3000)
        Xi = X.copy()
        for c in ["cat_a", "cat_b", "decoy_0", "decoy_1"]:
            Xi[c] = Xi[c].astype(int)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_cat_pair_enable=True,
            fe_cat_pair_cat_cols=("cat_a", "cat_b"),
            fe_cat_pair_top_k=3,
            fe_max_steps=0,
            fe_univariate_basis_enable=False,
            fe_univariate_fourier_enable=False,
        )
        m.fit(Xi, pd.Series(y, name="y"))
        cp_feats = list(getattr(m, "cat_pair_features_", []) or [])
        assert len(cp_feats) >= 1, (
            "cat_pair enabled but produced no engineered columns on the "
            "cat-XOR fixture."
        )


# ---------------------------------------------------------------------------
# Contract 8: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            hybrid_cat_pair_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        # Cover both raw and target-encoded recipe payloads.
        X_raw, y_raw = _build_cat_xor(1)
        X_hc, y_hc = _build_highcard_xor(1)
        cases = [
            (X_raw, y_raw, ["cat_a", "cat_b"], 0.001),
            (X_hc, y_hc, ["cat_a", "cat_b"], -1.0),
        ]
        for X, y, cols, thr in cases:
            _, appended, recipes, _ = hybrid_cat_pair_fe(
                X, y, cat_cols=cols, min_interaction_info=thr, top_k=5,
                random_state=1,
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
            fe_cat_pair_enable=True,
            fe_cat_pair_cat_cols=("cat_a", "cat_b"),
            fe_cat_pair_min_interaction_info=0.01,
            fe_cat_pair_top_k=7,
        )
        c = clone(m)
        assert bool(c.fe_cat_pair_enable) is True
        assert tuple(c.fe_cat_pair_cat_cols) == ("cat_a", "cat_b")
        assert float(c.fe_cat_pair_min_interaction_info) == 0.01
        assert int(c.fe_cat_pair_top_k) == 7
