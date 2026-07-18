"""Layer 77 biz_value: 4-WAY (QUADRUPLET) cross-basis ORTHOGONAL-POLY FE.

Consolidated verbatim from test_biz_value_mrmr_layer77.py (per audit finding test_code_quality-16).

Validates ``hybrid_orth_mi_quadruplet_fe`` introduced 2026-06-01 (sibling
module ``_orthogonal_quadruplet_fe``), extending Layer 22 (pair) and
Layer 56 (triplet) with a fourth leg:
``He_a(z_i) * He_b(z_j) * He_c(z_k) * He_d(z_l)`` cells ranked by MI
uplift vs the BEST individual leg.

Why this layer matters
----------------------

Layer 56 covers genuine 3-way interactions. Layer 77 covers genuine
4-way interactions:

* 4-way XOR ``y = sign(x_1 * x_2 * x_3 * x_4)``
  -> ``He_1(z_1) * He_1(z_2) * He_1(z_3) * He_1(z_4)`` IS the signal.
     Every TRIPLET marginal MI is zero by symmetry (the 4th factor
     randomises balanced), so the Layer 56 triplet stage cannot find
     it. The quadruplet stage emits exactly the right cell.

* Volume target ``y = sign(price * qty * count * discount - threshold)``
  -> ``He_1^4`` recovers the 4-way multiplicative structure.

Linear LogReg on raw (x1..x4) cannot solve 4-way XOR -> AUC ~0.50.
LogReg on (x1..x4, x1*x2*x3*x4) recovers >= 0.80 holdout.

Contracts pinned
----------------

* TestQuadrupletGeneration: emit the expected (deg_a, deg_b, deg_c, deg_d)
  cells per quadruplet with the right naming; skip degenerate (self-
  aliased / missing) quadruplets.
* TestXor4WayDiscovery: ``y = sign(x1*x2*x3*x4)`` -- the cell
  ``x1*x2*x3*x4__He1_He1_He1_He1`` is the top quadruplet winner.
* TestVolumeDiscovery: ``y = sign(price*qty*count*discount - thr)`` --
  the quadruplet term wins the ranking.
* TestXorLogRegLift: linear LogReg holdout AUC on 4-way XOR jumps from
  ~0.50 (raw, unsolvable) to >= 0.80 with the quadruplet FE applied.
* TestNoiseQuadrupletPruned: noise_a*noise_b*noise_c*noise_d filtered
  by the absolute MI floor.
* TestCombinatorialBound: seed_k=5 + max_degree=2 -> <= 80 candidates.
* TestDefaultDisabledByteIdentical: master switch OFF leaves
  ``support_`` / ``feature_names_in_`` clear of quadruplet columns.
* TestPickleAndClone: ``clone`` preserves new ctor params; ``pickle``
  round-trips fitted MRMR with quadruplet recipes.

NEVER xfail.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_quad_fe():
    """Lazily import the 4-way quadruplet cross-basis orthogonal-polynomial FE functions."""
    from mlframe.feature_selection.filters._orthogonal_quadruplet_fe import (
        generate_quadruplet_cross_basis_features,
        score_quadruplet_cross_basis_by_mi_uplift,
        hybrid_orth_mi_quadruplet_fe,
    )

    return (
        generate_quadruplet_cross_basis_features,
        score_quadruplet_cross_basis_by_mi_uplift,
        hybrid_orth_mi_quadruplet_fe,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr

# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_xor4(seed: int, n: int = 4000):
    """``y = sign(x1 * x2 * x3 * x4)`` -- pure 4-way XOR on Gaussians.

    Every triplet-cross-basis term has zero marginal MI because the
    fourth factor randomises balanced; only the He_1^4 cell recovers
    the sign of the 4-way product.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "noise_a": rng.standard_normal(n),
        }
    )
    y = ((x1 * x2 * x3 * x4) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_volume4(seed: int, n: int = 4000):
    """``y = sign(price * qty * count * discount - threshold)``.

    Realistic 4-way multiplicative signal. He_1^4 captures the
    centred product structure once the legs are z-scored.
    """
    rng = np.random.default_rng(seed)
    price = rng.standard_normal(n)
    qty = rng.standard_normal(n)
    count = rng.standard_normal(n)
    discount = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "price": price,
            "qty": qty,
            "count": count,
            "discount": discount,
            "noise_a": rng.standard_normal(n),
        }
    )
    raw = price * qty * count * discount
    y = (raw + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor4_with_noise(seed: int, n: int = 4000):
    """4-way XOR signal on (x1..x4) + four spectator noise legs. The
    noise^4 quadruplet must be filtered by the absolute MI floor.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
            "noise_d": rng.standard_normal(n),
        }
    )
    y = ((x1 * x2 * x3 * x4) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear-additive signal. Used for the default-disabled
    byte-identical contract.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
            "noise_d": rng.standard_normal(n),
        }
    )
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: quadruplet generation produces expected output
# ---------------------------------------------------------------------------


class TestQuadrupletGeneration:
    """generate_quadruplet_cross_basis_features emits the expected (deg_a..deg_d) cells per quadruplet."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_emits_per_quadruplet_per_degree_cell(self, seed):
        """One quadruplet at max_degree=1 emits exactly 1 correctly-named, NaN-free cell."""
        gen_quad, _, _ = _import_quad_fe()
        X, _ = _build_xor4(seed)
        eng = gen_quad(
            X,
            quadruplets=[("x1", "x2", "x3", "x4")],
            max_degree=1,
            basis="hermite",
        )
        # 1 quadruplet * 1 cell (deg 1 only) = 1 column
        assert eng.shape == (X.shape[0], 1), f"expected 1 cell (1 quadruplet * 1 degree cell), got shape {eng.shape}"
        assert "x1*x2*x3*x4__He1_He1_He1_He1" in eng.columns, f"expected x1*x2*x3*x4__He1_He1_He1_He1, got {list(eng.columns)}"
        assert eng.notna().all().all(), f"NaN in quadruplet output seed={seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_degree_2_expands_cells(self, seed):
        """One quadruplet at max_degree=2 emits exactly 16 (2^4) cells."""
        gen_quad, _, _ = _import_quad_fe()
        X, _ = _build_xor4(seed)
        eng = gen_quad(
            X,
            quadruplets=[("x1", "x2", "x3", "x4")],
            max_degree=2,
            basis="hermite",
        )
        # 1 quadruplet * 2^4 cells = 16 columns
        assert eng.shape == (X.shape[0], 16), f"expected 16 cells (1 quadruplet * 2^4 degrees), got shape {eng.shape}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_skips_self_aliased_and_missing(self, seed):
        """A self-aliased quadruplet and one referencing a missing column are skipped; only the valid one emits cells."""
        gen_quad, _, _ = _import_quad_fe()
        X, _ = _build_xor4(seed)
        eng = gen_quad(
            X,
            quadruplets=[
                ("x1", "x1", "x2", "x3"),  # alias -> skip
                ("x1", "x2", "x3", "does_not_exist"),  # missing -> skip
                ("x1", "x2", "x3", "x4"),  # valid
            ],
            max_degree=1,
            basis="hermite",
        )
        assert eng.shape[1] == 1, f"only the valid quadruplet should remain, got {list(eng.columns)}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_empty_quadruplets_returns_empty_frame(self, seed):
        """An empty quadruplets list returns a frame with zero columns."""
        gen_quad, _, _ = _import_quad_fe()
        X, _ = _build_xor4(seed)
        eng = gen_quad(X, quadruplets=[], max_degree=1, basis="hermite")
        assert eng.shape == (X.shape[0], 0)


# ---------------------------------------------------------------------------
# Contract 2: 4-way XOR target -- He_1^4 cell discovered
# ---------------------------------------------------------------------------


class TestXor4WayDiscovery:
    """``y = sign(x1*x2*x3*x4)`` (4-way XOR) is dominated by the He_1^4 quadruplet cell in ranking and augmented frame."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_he1_quad_term_tops_quadruplet_ranking(self, seed):
        """He_1^4 is the top-ranked quadruplet cell by MI uplift, with substantial engineered MI."""
        gen_quad, score_quad, _ = _import_quad_fe()
        X, y = _build_xor4(seed)
        eng = gen_quad(
            X,
            quadruplets=[("x1", "x2", "x3", "x4")],
            max_degree=1,
            basis="hermite",
        )
        sc = score_quad(X[["x1", "x2", "x3", "x4"]], eng, y.values)
        assert not sc.empty, "score frame empty"
        top = sc.iloc[0]
        assert (
            top["engineered_col"] == "x1*x2*x3*x4__He1_He1_He1_He1"
        ), f"seed={seed}: top quadruplet winner should be x1*x2*x3*x4__He1_He1_He1_He1, got {top['engineered_col']}; full ranking:\n{sc}"
        assert top["engineered_mi"] >= 0.2, f"seed={seed}: 4-way XOR He_1^4 engineered_mi {top['engineered_mi']:.3f} should clear 0.2 at n=4000"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_xor4_term_enters_augmented_frame(self, seed):
        """The He_1^4 4-way XOR quadruplet term enters the hybrid-augmented frame."""
        _, _, hybrid = _import_quad_fe()
        X, y = _build_xor4(seed)
        X_aug, _uni_sc, quad_sc = hybrid(
            X,
            y.values,
            cols=["x1", "x2", "x3", "x4", "noise_a"],
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            top_quadruplet_count=3,
            top_quadruplet_seed_k=5,
            quadruplet_max_degree=1,
            quadruplet_min_uplift=1.05,
            quadruplet_min_abs_mi_frac=0.1,
        )
        new_cols = [c for c in X_aug.columns if c not in X.columns]
        quad_cols = [c for c in new_cols if c.split("__", 1)[0].count("*") == 3]
        ok = False
        for c in quad_cols:
            head = c.split("__", 1)[0]
            legs = set(head.split("*"))
            if legs == {"x1", "x2", "x3", "x4"} and c.endswith("__He1_He1_He1_He1"):
                ok = True
                break
        assert ok, f"seed={seed}: 4-way XOR quadruplet He_1^4 should be in augmented frame, got quad_cols={quad_cols}; quad_sc:\n{quad_sc.head(6)}"


# ---------------------------------------------------------------------------
# Contract 3: volume target discovered
# ---------------------------------------------------------------------------


class TestVolumeDiscovery:
    """``y = sign(price*qty*count*discount - thr)`` is dominated by the He_1^4 quadruplet cell in the ranking."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_volume_quadruplet_top_ranked(self, seed):
        """He_1^4 on (price, qty, count, discount) is the top-ranked quadruplet cell, with substantial engineered MI."""
        gen_quad, score_quad, _ = _import_quad_fe()
        X, y = _build_volume4(seed)
        eng = gen_quad(
            X,
            quadruplets=[("price", "qty", "count", "discount")],
            max_degree=1,
            basis="hermite",
        )
        sc = score_quad(
            X[["price", "qty", "count", "discount"]],
            eng,
            y.values,
        )
        top = sc.iloc[0]
        assert top["engineered_col"] == (
            "price*qty*count*discount__He1_He1_He1_He1"
        ), f"seed={seed}: top volume winner should be the He_1^4 quadruplet, got {top['engineered_col']}"
        assert top["engineered_mi"] >= 0.2, f"seed={seed}: volume He_1^4 engineered_mi {top['engineered_mi']:.3f} should clear 0.2 at n=4000"


# ---------------------------------------------------------------------------
# Contract 4: downstream LogReg AUC lift on 4-way XOR
# ---------------------------------------------------------------------------


class TestXorLogRegLift:
    """Quadruplet-augmented LogReg measurably lifts holdout AUC over raw LogReg on the unsolvable 4-way XOR target."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lifts_with_quadruplet_fe(self, seed):
        """Quadruplet-augmented LogReg clears 0.80 AUC and beats raw LogReg by >= +0.15 on 4-way XOR."""
        _, _, hybrid = _import_quad_fe()
        X, y = _build_xor4(seed, n=5000)
        n_train = 3500
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        m_raw = LogisticRegression(max_iter=500).fit(Xtr.to_numpy(), ytr.to_numpy())
        auc_raw = roc_auc_score(yte.to_numpy(), m_raw.predict_proba(Xte.to_numpy())[:, 1])
        X_aug_joint, _, quad_sc = hybrid(
            X,
            y.values,
            cols=["x1", "x2", "x3", "x4", "noise_a"],
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            top_quadruplet_count=3,
            top_quadruplet_seed_k=5,
            quadruplet_max_degree=1,
            quadruplet_min_uplift=1.05,
            quadruplet_min_abs_mi_frac=0.1,
        )
        Xtr_aug = X_aug_joint.iloc[:n_train]
        Xte_aug = X_aug_joint.iloc[n_train:]
        m_aug = LogisticRegression(max_iter=500).fit(Xtr_aug.to_numpy(), ytr.to_numpy())
        auc_aug = roc_auc_score(yte.to_numpy(), m_aug.predict_proba(Xte_aug.to_numpy())[:, 1])
        assert auc_raw < 0.60, f"seed={seed}: raw LogReg AUC {auc_raw:.3f} should hover at 0.50 -- 4-way XOR is linearly unsolvable"
        assert (
            auc_aug >= 0.80
        ), f"seed={seed}: augmented LogReg AUC {auc_aug:.3f} should clear 0.80 on 4-way XOR with quadruplet FE; quad_sc:\n{quad_sc.head(5)}"
        assert auc_aug > auc_raw + 0.15, f"seed={seed}: quadruplet FE should lift LogReg AUC >= +0.15. raw={auc_raw:.3f}, aug={auc_aug:.3f}"


# ---------------------------------------------------------------------------
# Contract 5: pure-noise quadruplets pruned by abs MI floor
# ---------------------------------------------------------------------------


class TestNoiseQuadrupletPruned:
    """Pure-noise quadruplet cells are pruned by the absolute MI floor."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_quad_filtered(self, seed):
        """No quadruplet column whose four legs are all noise sources enters the augmented frame."""
        _, _, hybrid = _import_quad_fe()
        X, y = _build_xor4_with_noise(seed)
        X_aug, _, quad_sc = hybrid(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            top_quadruplet_count=8,
            top_quadruplet_seed_k=8,
            quadruplet_max_degree=1,
            quadruplet_min_uplift=1.05,
            quadruplet_min_abs_mi_frac=0.1,
        )
        new_cols = [c for c in X_aug.columns if c not in X.columns]
        quad_added = [c for c in new_cols if c.split("__", 1)[0].count("*") == 3]

        def _all_legs_noise(name: str) -> bool:
            """Check whether a quadruplet column name's four source legs are all noise columns."""
            head = name.split("__", 1)[0]
            legs = head.split("*")
            if len(legs) != 4:
                return False
            return all(leg.startswith("noise_") for leg in legs)

        noise_added = [c for c in quad_added if _all_legs_noise(c)]
        assert (
            not noise_added
        ), f"seed={seed}: pure-noise quadruplets should be filtered by the absolute MI floor; got {noise_added}; quad_sc:\n{quad_sc.head(8)}"


# ---------------------------------------------------------------------------
# Contract 6: combinatorial bound
# ---------------------------------------------------------------------------


class TestCombinatorialBound:
    """The quadruplet cell count matches the combinatorial bound: C(seed_k,4) quadruplets times 2^4 degree cells."""

    def test_seed_k_5_max_degree_2_at_most_80_cells(self):
        """5 columns (C(5,4)=5 quadruplets) at max_degree=2 produces exactly 80 cells."""
        gen_quad, _, _ = _import_quad_fe()
        # 5 columns: C(5,4) = 5 quadruplets; per quadruplet 2^4 = 16 cells.
        # Total: 5 * 16 = 80 candidates.
        rng = np.random.default_rng(0)
        n = 800
        X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(5)})
        names = list(X.columns)
        quadruplets = [
            (names[i], names[j], names[k], names[m])
            for i in range(len(names))
            for j in range(i + 1, len(names))
            for k in range(j + 1, len(names))
            for m in range(k + 1, len(names))
        ]
        assert len(quadruplets) == 5, f"C(5,4) should be 5, got {len(quadruplets)}"
        eng = gen_quad(
            X,
            quadruplets=quadruplets,
            max_degree=2,
            basis="hermite",
        )
        assert eng.shape[1] <= 80, f"seed_k=5 + max_degree=2 should produce <= 80 quadruplet cells, got {eng.shape[1]}"
        assert eng.shape[1] == 80, f"seed_k=5 + max_degree=2 should produce exactly 80 quadruplet cells (5 quadruplets * 2^4), got {eng.shape[1]}"


# ---------------------------------------------------------------------------
# Contract 7: default disabled -- legacy behaviour byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_quadruplet_enable=False (the default) leaves feature_names_in_ and hybrid_orth_features_ clear."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_quadruplet_columns(self, seed):
        """Default fe_hybrid_orth_quadruplet_enable=False injects no quadruplet columns."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        names = list(m.feature_names_in_)
        # No quadruplet columns (3 stars) surfaced in feature_names_in_.
        quad_names = [n for n in names if str(n).count("*") == 3]
        assert not quad_names, f"seed={seed}: default fe_hybrid_orth_quadruplet_enable=False should NOT inject quadruplet columns; got {quad_names}"
        assert list(getattr(m, "hybrid_orth_features_", []) or []) == [], f"seed={seed}: hybrid_orth_features_ should be empty when all hybrid flags are off"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_quadruplet_appends_engineered(self, seed):
        """fe_hybrid_orth_quadruplet_enable=True appends at least one quadruplet column to hybrid_orth_features_."""
        X, y = _build_xor4(seed, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_quadruplet_enable=True,
            fe_hybrid_orth_quadruplet_max_degree=1,
            fe_hybrid_orth_quadruplet_seed_k=5,
            fe_hybrid_orth_quadruplet_top_count=2,
        ).fit(X, y)
        quad_added = [n for n in (getattr(m, "hybrid_orth_features_", None) or []) if str(n).split("__", 1)[0].count("*") == 3]
        assert (
            quad_added
        ), f"seed={seed}: quadruplet flag ON should append at least one quadruplet column to hybrid_orth_features_; got {list(m.hybrid_orth_features_ or [])}"


# ---------------------------------------------------------------------------
# Contract 8: pickle / clone preserve quadruplet ctor + fitted recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """clone() preserves the quadruplet ctor params; pickle round-trips a fitted MRMR with quadruplet recipes."""

    def test_clone_preserves_quadruplet_params(self):
        """clone() copies all fe_hybrid_orth_quadruplet_* params without carrying over fitted state."""
        m = _make_mrmr(
            fe_hybrid_orth_quadruplet_enable=True,
            fe_hybrid_orth_quadruplet_max_degree=2,
            fe_hybrid_orth_quadruplet_seed_k=5,
            fe_hybrid_orth_quadruplet_top_count=3,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_quadruplet_enable", True),
            ("fe_hybrid_orth_quadruplet_max_degree", 2),
            ("fe_hybrid_orth_quadruplet_seed_k", 5),
            ("fe_hybrid_orth_quadruplet_top_count", 3),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_fitted_with_quadruplet(self):
        """pickle.dumps/loads preserves feature_names_in_ and hybrid_orth_features_ for a quadruplet-enabled fit."""
        X, y = _build_xor4(seed=42, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_quadruplet_enable=True,
            fe_hybrid_orth_quadruplet_max_degree=1,
            fe_hybrid_orth_quadruplet_seed_k=5,
            fe_hybrid_orth_quadruplet_top_count=2,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        assert list(getattr(m2, "hybrid_orth_features_", []) or []) == list(
            getattr(m, "hybrid_orth_features_", []) or []
        ), "pickle changed hybrid_orth_features_"
