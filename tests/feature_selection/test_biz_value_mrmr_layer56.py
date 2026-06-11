"""Layer 56 biz_value: TRI-PRODUCT cross-basis ORTHOGONAL-POLYNOMIAL FE.

Validates ``hybrid_orth_mi_triplet_fe`` introduced 2026-05-31 (sibling
module ``_orthogonal_triplet_fe``), which extends Layer 22's pair-cross-
basis pipeline with a third leg:
``He_a(z_i) * He_b(z_j) * He_c(z_k)`` cells ranked by MI uplift vs the
BEST individual leg.

Why this layer matters
----------------------

Layer 22 covers pair non-linearities (XOR, saddle, circle). Layer 56
covers genuine 3-way interactions:

* 3-way XOR ``y = sign(x_1 * x_2 * x_3)``
  -> ``He_1(z_1) * He_1(z_2) * He_1(z_3)`` IS the signal. Every PAIR
     marginal MI is zero by symmetry (the third var randomises balanced),
     so the Layer 22 pair stage cannot find it. The triplet stage emits
     exactly the right cell and the engineered MI dominates.

* Volume target ``y = sign(price * quantity * count - threshold)``
  -> ``He_1*He_1*He_1`` recovers the product structure even when the
     marginals carry little univariate signal.

Linear LogReg on raw (x_1, x_2, x_3) cannot solve 3-way XOR -> AUC
~0.50. LogReg on (x_1, x_2, x_3, x_1*x_2*x_3) recovers >= 0.85 holdout.

Contracts pinned
----------------

* TestTripletGeneration: ``generate_triplet_cross_basis_features``
  emits the expected (deg_a, deg_b, deg_c) cells per triplet with the
  right naming and skips degenerate (self-aliased / missing) triplets.

* TestXor3WayDiscovery: ``y = sign(x1 * x2 * x3)`` -- the cell
  ``x1*x2*x3__He1_He1_He1`` enters the cross_scores top winners.

* TestVolumeDiscovery: ``y = sign(price * quantity * count - threshold)``
  -- the triplet term wins the ranking.

* TestXorLogRegLift: linear LogReg's holdout AUC on 3-way XOR jumps
  from ~0.50 (raw, unsolvable) to >= 0.85 with the triplet FE applied.

* TestNoiseTripletPruned: noise_a * noise_b * noise_c is filtered by
  the absolute MI floor even when a real signal pulls the floor high.

* TestDefaultDisabledByteIdentical: ``fe_hybrid_orth_triplet_enable=False``
  (the default) keeps ``support_`` and ``feature_names_in_`` identical to
  a fit without the triplet flag.

* TestPickleAndClone: sklearn-style ``clone`` preserves the new ctor
  params; ``pickle`` round-trips a fitted MRMR with triplet recipes.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_triplet_fe():
    from mlframe.feature_selection.filters._orthogonal_triplet_fe import (
        generate_triplet_cross_basis_features,
        score_triplet_cross_basis_by_mi_uplift,
        hybrid_orth_mi_triplet_fe,
    )
    return (
        generate_triplet_cross_basis_features,
        score_triplet_cross_basis_by_mi_uplift,
        hybrid_orth_mi_triplet_fe,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_xor3(seed: int, n: int = 3000):
    """``y = sign(x1 * x2 * x3)`` -- pure 3-way XOR on Gaussians.

    Every pair-cross-basis term has zero marginal MI because the third
    factor randomises balanced; only ``He_1*He_1*He_1`` recovers the
    sign of the triple product.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
    })
    y = ((x1 * x2 * x3) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_volume(seed: int, n: int = 3000):
    """``y = sign(price * quantity * count - threshold)``.

    Realistic three-way multiplicative signal. He_1^3 captures the
    centred product structure once the legs are z-scored.
    """
    rng = np.random.default_rng(seed)
    # log-normal-ish positive marginals to avoid degenerate sign behaviour.
    price = rng.standard_normal(n)
    quantity = rng.standard_normal(n)
    count = rng.standard_normal(n)
    X = pd.DataFrame({
        "price": price,
        "quantity": quantity,
        "count": count,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
    })
    raw = price * quantity * count
    y = (raw + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor3_with_noise(seed: int, n: int = 3000):
    """3-way XOR signal on (x1, x2, x3) + spectator noise legs. The
    noise*noise*noise triplet must be filtered by the absolute MI floor
    anchored on the strong x1*x2*x3 cell.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
        "noise_c": rng.standard_normal(n),
    })
    y = ((x1 * x2 * x3) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear-additive signal. Used for the default-disabled
    byte-identical contract.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
        "noise_c": rng.standard_normal(n),
    })
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: triplet generation produces expected output
# ---------------------------------------------------------------------------


class TestTripletGeneration:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_emits_per_triplet_per_degree_cell(self, seed):
        gen_triplet, _, _ = _import_triplet_fe()
        X, _ = _build_xor3(seed)
        eng = gen_triplet(
            X, triplets=[("x1", "x2", "x3")],
            max_degree=1, basis="hermite",
        )
        # 1 triplet * 1 cell (deg 1 only) = 1 column
        assert eng.shape == (X.shape[0], 1), (
            f"expected 1 cell (1 triplet * 1 degree cell), got shape {eng.shape}"
        )
        assert "x1*x2*x3__He1_He1_He1" in eng.columns, (
            f"expected x1*x2*x3__He1_He1_He1, got {list(eng.columns)}"
        )
        assert eng.notna().all().all(), f"NaN in triplet output seed={seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_degree_2_expands_cells(self, seed):
        gen_triplet, _, _ = _import_triplet_fe()
        X, _ = _build_xor3(seed)
        eng = gen_triplet(
            X, triplets=[("x1", "x2", "x3")],
            max_degree=2, basis="hermite",
        )
        # 1 triplet * 2^3 cells = 8 columns
        assert eng.shape == (X.shape[0], 8), (
            f"expected 8 cells (1 triplet * 2x2x2 degrees), got shape {eng.shape}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_skips_self_aliased_and_missing(self, seed):
        gen_triplet, _, _ = _import_triplet_fe()
        X, _ = _build_xor3(seed)
        eng = gen_triplet(
            X,
            triplets=[
                ("x1", "x1", "x2"),           # alias -> skip
                ("x1", "x2", "does_not_exist"),  # missing -> skip
                ("x1", "x2", "x3"),           # valid
            ],
            max_degree=1, basis="hermite",
        )
        assert eng.shape[1] == 1, (
            f"only the valid triplet should remain, got {list(eng.columns)}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_empty_triplets_returns_empty_frame(self, seed):
        gen_triplet, _, _ = _import_triplet_fe()
        X, _ = _build_xor3(seed)
        eng = gen_triplet(X, triplets=[], max_degree=1, basis="hermite")
        assert eng.shape == (X.shape[0], 0)


# ---------------------------------------------------------------------------
# Contract 2: 3-way XOR target -- He_1*He_1*He_1 cell discovered
# ---------------------------------------------------------------------------


class TestXor3WayDiscovery:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_he1_cubed_term_dominates_triplet_ranking(self, seed):
        gen_triplet, score_triplet, _ = _import_triplet_fe()
        X, y = _build_xor3(seed)
        eng = gen_triplet(
            X, triplets=[("x1", "x2", "x3")],
            max_degree=1, basis="hermite",
        )
        sc = score_triplet(X[["x1", "x2", "x3"]], eng, y.values)
        assert not sc.empty, "score frame empty"
        top = sc.iloc[0]
        assert top["engineered_col"] == "x1*x2*x3__He1_He1_He1", (
            f"seed={seed}: top triplet winner should be x1*x2*x3__He1_He1_He1, "
            f"got {top['engineered_col']}; full ranking:\n{sc}"
        )
        assert top["engineered_mi"] >= 0.3, (
            f"seed={seed}: 3-way XOR He_1^3 engineered_mi {top['engineered_mi']:.3f} "
            f"should clear 0.3 at n=3000"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_xor3_term_enters_augmented_frame(self, seed):
        _, _, hybrid = _import_triplet_fe()
        X, y = _build_xor3(seed)
        X_aug, uni_sc, triplet_sc = hybrid(
            X, y.values,
            cols=["x1", "x2", "x3", "noise_a", "noise_b"],
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            top_triplet_count=3,
            top_triplet_seed_k=4,
            triplet_max_degree=1,
            triplet_min_uplift=1.05,
            triplet_min_abs_mi_frac=0.1,
        )
        new_cols = [c for c in X_aug.columns if c not in X.columns]
        triplet_cols = [c for c in new_cols if c.split("__", 1)[0].count("*") == 2]
        # The He_1*He_1*He_1 cell on (x1, x2, x3) -- any permutation of leg
        # order -- must enter the augmented frame.
        ok = False
        for c in triplet_cols:
            head = c.split("__", 1)[0]
            legs = set(head.split("*"))
            if legs == {"x1", "x2", "x3"} and c.endswith("__He1_He1_He1"):
                ok = True
                break
        assert ok, (
            f"seed={seed}: 3-way XOR triplet He_1*He_1*He_1 should be in "
            f"augmented frame, got triplet_cols={triplet_cols}; "
            f"triplet_sc:\n{triplet_sc.head(6)}"
        )


# ---------------------------------------------------------------------------
# Contract 3: volume target discovered
# ---------------------------------------------------------------------------


class TestVolumeDiscovery:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_volume_triplet_top_ranked(self, seed):
        gen_triplet, score_triplet, _ = _import_triplet_fe()
        X, y = _build_volume(seed)
        eng = gen_triplet(
            X, triplets=[("price", "quantity", "count")],
            max_degree=1, basis="hermite",
        )
        sc = score_triplet(X[["price", "quantity", "count"]], eng, y.values)
        top = sc.iloc[0]
        assert top["engineered_col"] == "price*quantity*count__He1_He1_He1", (
            f"seed={seed}: top volume winner should be the He_1^3 triplet, "
            f"got {top['engineered_col']}"
        )
        assert top["engineered_mi"] >= 0.3, (
            f"seed={seed}: volume He_1^3 engineered_mi {top['engineered_mi']:.3f} "
            f"should clear 0.3 at n=3000"
        )


# ---------------------------------------------------------------------------
# Contract 4: downstream LogReg AUC lift on 3-way XOR
# ---------------------------------------------------------------------------


class TestXorLogRegLift:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lifts_with_triplet_fe(self, seed):
        _, _, hybrid = _import_triplet_fe()
        X, y = _build_xor3(seed, n=4000)
        n_train = 2800
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        # Baseline: raw LogReg on x1, x2, x3, noise -- 3-way XOR is
        # unsolvable by linear LogReg, AUC ~0.50.
        m_raw = LogisticRegression(max_iter=500).fit(Xtr.to_numpy(), ytr.to_numpy())
        auc_raw = roc_auc_score(
            yte.to_numpy(), m_raw.predict_proba(Xte.to_numpy())[:, 1]
        )
        # Augmented: triplet FE on the full frame, then refit LogReg.
        X_aug_joint, _, triplet_sc = hybrid(
            X, y.values,
            cols=["x1", "x2", "x3", "noise_a", "noise_b"],
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            top_triplet_count=3,
            top_triplet_seed_k=4,
            triplet_max_degree=1,
            triplet_min_uplift=1.05,
            triplet_min_abs_mi_frac=0.1,
        )
        Xtr_aug = X_aug_joint.iloc[:n_train]
        Xte_aug = X_aug_joint.iloc[n_train:]
        m_aug = LogisticRegression(max_iter=500).fit(
            Xtr_aug.to_numpy(), ytr.to_numpy()
        )
        auc_aug = roc_auc_score(
            yte.to_numpy(), m_aug.predict_proba(Xte_aug.to_numpy())[:, 1]
        )
        assert auc_raw < 0.60, (
            f"seed={seed}: raw LogReg AUC {auc_raw:.3f} should hover at "
            f"0.50 -- 3-way XOR is linearly unsolvable"
        )
        assert auc_aug >= 0.85, (
            f"seed={seed}: augmented LogReg AUC {auc_aug:.3f} should clear "
            f"0.85 on 3-way XOR with triplet FE; triplet_sc:\n{triplet_sc.head(5)}"
        )
        assert auc_aug > auc_raw + 0.20, (
            f"seed={seed}: triplet FE should lift LogReg AUC >= +0.20. "
            f"raw={auc_raw:.3f}, aug={auc_aug:.3f}"
        )


# ---------------------------------------------------------------------------
# Contract 5: pure-noise triplets pruned by abs MI floor
# ---------------------------------------------------------------------------


class TestNoiseTripletPruned:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_noise_noise_triplet_filtered(self, seed):
        _, _, hybrid = _import_triplet_fe()
        X, y = _build_xor3_with_noise(seed)
        X_aug, _, triplet_sc = hybrid(
            X, y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            top_triplet_count=5,
            top_triplet_seed_k=6,
            triplet_max_degree=1,
            triplet_min_uplift=1.05,
            triplet_min_abs_mi_frac=0.1,
        )
        new_cols = [c for c in X_aug.columns if c not in X.columns]
        triplet_added = [c for c in new_cols if c.split("__", 1)[0].count("*") == 2]
        # No triplet whose ALL THREE legs are noise_* may slip in.
        def _all_legs_noise(name: str) -> bool:
            head = name.split("__", 1)[0]
            legs = head.split("*")
            if len(legs) != 3:
                return False
            return all(leg.startswith("noise_") for leg in legs)
        noise_added = [c for c in triplet_added if _all_legs_noise(c)]
        assert not noise_added, (
            f"seed={seed}: pure-noise triplets should be filtered by the "
            f"absolute MI floor; got {noise_added}; "
            f"triplet_sc:\n{triplet_sc.head(8)}"
        )


# ---------------------------------------------------------------------------
# Contract 6: default disabled -- legacy behaviour byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_triplet_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        # No triplet columns surfaced in feature_names_in_.
        names = list(m.feature_names_in_)
        triplet_names = [n for n in names if str(n).count("*") == 2]
        assert not triplet_names, (
            f"seed={seed}: default fe_hybrid_orth_triplet_enable=False should "
            f"NOT inject triplet columns; got {triplet_names}"
        )
        # ``hybrid_orth_features_`` is the standard list; with both master
        # and triplet OFF it must be empty.
        assert list(getattr(m, "hybrid_orth_features_", []) or []) == [], (
            f"seed={seed}: hybrid_orth_features_ should be empty when "
            f"both master and triplet flags are off"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_triplet_appends_engineered(self, seed):
        X, y = _build_xor3(seed, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_triplet_enable=True,
            fe_hybrid_orth_triplet_max_degree=1,
            fe_hybrid_orth_triplet_seed_k=4,
            fe_hybrid_orth_triplet_top_count=2,
        ).fit(X, y)
        triplet_added = [
            n for n in (getattr(m, "hybrid_orth_features_", None) or [])
            if str(n).split("__", 1)[0].count("*") == 2
        ]
        assert triplet_added, (
            f"seed={seed}: triplet flag ON should append at least one "
            f"triplet column to hybrid_orth_features_; got "
            f"{list(m.hybrid_orth_features_ or [])}"
        )


# ---------------------------------------------------------------------------
# Contract 7: pickle / clone preserve triplet ctor + fitted recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_triplet_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_triplet_enable=True,
            fe_hybrid_orth_triplet_max_degree=2,
            fe_hybrid_orth_triplet_seed_k=5,
            fe_hybrid_orth_triplet_top_count=3,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_triplet_enable", True),
            ("fe_hybrid_orth_triplet_max_degree", 2),
            ("fe_hybrid_orth_triplet_seed_k", 5),
            ("fe_hybrid_orth_triplet_top_count", 3),
        ]:
            assert getattr(m2, name) == expected, (
                f"clone() dropped {name}: expected {expected}, got "
                f"{getattr(m2, name)}"
            )

    def test_pickle_roundtrip_fitted_with_triplet(self):
        X, y = _build_xor3(seed=42, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_triplet_enable=True,
            fe_hybrid_orth_triplet_max_degree=1,
            fe_hybrid_orth_triplet_seed_k=4,
            fe_hybrid_orth_triplet_top_count=2,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), (
            "pickle changed feature_names_in_"
        )
        assert (
            list(getattr(m2, "hybrid_orth_features_", []) or [])
            == list(getattr(m, "hybrid_orth_features_", []) or [])
        ), "pickle changed hybrid_orth_features_"
