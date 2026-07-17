"""Consolidated from test_biz_value_mrmr_layer22.py.

Layer 22 biz_value: CROSS-BASIS PAIR ORTHOGONAL-POLYNOMIAL FE.

Validates ``hybrid_orth_mi_pair_fe`` introduced 2026-05-31, which extends
Layer 21's univariate-only pipeline with a pair-cross-basis stage:
``He_a(x_i) * He_b(x_j)`` cells ranked by MI uplift vs the better
individual leg.

Why this layer matters
----------------------

Layer 21 covers single-feature non-linearities (``y = f(x_i)``). Layer 22
covers PAIR non-linearities that are ALSO not learnable by a CMA-ES pair
search at the cells we enumerate -- and far cheaper. The classic targets:

* XOR    ``y = sign(x_i * x_j)``           -> ``He_1(z_i) * He_1(z_j)`` = z_i*z_j
* Saddle ``y = sign((x_i^2 - 1)(x_j^2 - 1))`` -> ``He_2(z_i) * He_2(z_j)``

Each of these is a SINGLE cell in the pair-cross-basis grid. The MI of
the right cell against y is >>> the MI of either source column alone --
linear LogReg on x_i, x_j cannot solve them at all (random AUC), but
LogReg on x_i, x_j, x_i*x_j (the He_1*He_1 augmentation) scores ~1.0.

What the contract classes pin
-----------------------------

* TestCrossBasisGeneration: ``generate_pair_cross_basis_features`` emits
  the expected (deg_a, deg_b) cells per pair with the right naming.

* TestXorDiscovery: ``y = sign(x_i * x_j)`` -- ``x_i*x_j__He1_He1`` enters
  the cross_scores top winners.

* TestSaddleDiscovery: ``y = sign((x_i^2 - 1)(x_j^2 - 1))`` -- the
  He_2 * He_2 cross-basis cell ranks among the top winners.

* TestMixedSignalsBothSurfaced: a dataset with only univariate signal in
  one column and only cross-basis signal in another pair -- the
  univariate winner enters X_aug FROM the univariate stage AND the
  cross-basis winner enters X_aug FROM the pair stage.

* TestXorLogRegLift: linear LogReg's holdout AUC on XOR jumps from
  ~0.50 (raw, unsolvable) to >= 0.85 with the cross-basis pair FE
  applied. The end-to-end biz_value claim.

* TestNoisePairPruned: pure-noise pairs (no signal anywhere) do NOT
  reach the augmented frame -- the absolute MI floor catches them even
  when relative uplifts look attractive on tiny baselines.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_fe():
    """Lazily import the pair cross-basis orthogonal-polynomial FE functions."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_pair_cross_basis_features,
        score_pair_cross_basis_by_mi_uplift,
        hybrid_orth_mi_pair_fe,
    )

    return (
        generate_pair_cross_basis_features,
        score_pair_cross_basis_by_mi_uplift,
        hybrid_orth_mi_pair_fe,
    )


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_xor(seed: int, n: int = 2500):
    """y = sign(x1 * x2) -- pure XOR on Gaussians. Linear LogReg cannot
    solve this. He_1(z1)*He_1(z2) = z1*z2 IS the signal, MI is large.
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
        }
    )
    y = (x1 * x2 + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y)


def _build_he2_pair_saddle(seed: int, n: int = 2500):
    """y = sign((x1^2 - 1) * (x2^2 - 1)). The exact He_2(z1) * He_2(z2)
    target (z = z-scored Gaussian). Individual He_2(x1) alone has noise-
    floor MI; the cross-basis He_2 * He_2 cell carries the signal.
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
        }
    )
    raw = (x1**2 - 1.0) * (x2**2 - 1.0)
    y = (raw + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y)


def _build_mixed_signals(seed: int, n: int = 3000):
    """Univariate signal in x_solo (y component: He_2(x_solo)) AND
    cross-basis signal in (x_pa, x_pb) (y component: He_1*He_1 = XOR).
    Combined target: y = sign(He_2(x_solo) + (x_pa * x_pb) * scale).
    Both the univariate stage and the cross-basis stage should surface
    their respective winners.
    """
    rng = np.random.default_rng(seed)
    x_solo = rng.standard_normal(n)
    x_pa = rng.standard_normal(n)
    x_pb = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x_solo": x_solo,
            "x_pa": x_pa,
            "x_pb": x_pb,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
        }
    )
    # Weight the two signals comparably so both reach above the noise floor.
    raw = (x_solo**2 - 1.0) + 1.5 * (x_pa * x_pb)
    y = (raw + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y)


def _build_xor_with_noise_pair(seed: int, n: int = 2500):
    """Real XOR signal on (x1, x2) PLUS spectator noise columns. The
    noise*noise cross-basis terms should be filtered out by the absolute
    MI floor while the x1*x2__He1_He1 winner enters.
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
    y = (x1 * x2 + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y)


# ---------------------------------------------------------------------------
# Contract 1: cross-basis generation produces expected output
# ---------------------------------------------------------------------------


class TestCrossBasisGeneration:
    """generate_pair_cross_basis_features emits the expected (deg_a, deg_b) cells per pair."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_emits_per_pair_per_degree_cell(self, seed):
        """One pair times max_degree=2 emits exactly 4 correctly-named, NaN-free cross-basis cells."""
        gen_pair, _, _ = _import_fe()
        X, _ = _build_xor(seed)
        eng = gen_pair(X, pairs=[("x1", "x2")], max_degree=2, basis="hermite")
        # 1 pair * (2*2) cells = 4 columns (min_degree=1, max_degree=2 by default)
        assert eng.shape == (X.shape[0], 4), f"expected 4 cells (1 pair * 2x2 degrees), got shape {eng.shape}"
        expected = {
            "x1*x2__He1_He1",
            "x1*x2__He1_He2",
            "x1*x2__He2_He1",
            "x1*x2__He2_He2",
        }
        assert set(eng.columns) == expected, f"col set mismatch: got {list(eng.columns)}, expected {expected}"
        assert eng.notna().all().all(), f"NaN in cross-basis output seed={seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_skips_self_pair_and_missing(self, seed):
        """A self-pair and a pair referencing a missing column are silently skipped; only the valid pair emits cells."""
        gen_pair, _, _ = _import_fe()
        X, _ = _build_xor(seed)
        eng = gen_pair(
            X,
            pairs=[("x1", "x1"), ("x1", "does_not_exist"), ("x1", "x2")],
            max_degree=2,
            basis="hermite",
        )
        # Only ("x1","x2") remains -> 4 cells.
        assert eng.shape[1] == 4
        for col in eng.columns:
            head = col.split("__", 1)[0]
            assert head == "x1*x2"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_empty_pairs_returns_empty_frame(self, seed):
        """An empty pairs list returns a frame with zero columns."""
        gen_pair, _, _ = _import_fe()
        X, _ = _build_xor(seed)
        eng = gen_pair(X, pairs=[], max_degree=2, basis="hermite")
        assert eng.shape == (X.shape[0], 0)


# ---------------------------------------------------------------------------
# Contract 2: XOR target -- He_1 * He_1 cross-basis discovered
# ---------------------------------------------------------------------------


class TestXorDiscovery:
    """``y = sign(x1*x2)`` (XOR) is dominated by the He_1*He_1 cross-basis cell in the ranking and augmented frame."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_he1_he1_term_dominates_cross_ranking(self, seed):
        """He_1*He_1 is the top-ranked cross-basis cell by MI uplift, with substantial engineered MI."""
        _, score_pair, _ = _import_fe()
        gen_pair, _, _ = _import_fe()
        X, y = _build_xor(seed)
        eng = gen_pair(X, pairs=[("x1", "x2")], max_degree=2, basis="hermite")
        sc = score_pair(X[["x1", "x2"]], eng, y.values)
        # Top winner by uplift must be the He_1 * He_1 cell.
        top = sc.iloc[0]
        assert top["engineered_col"] == "x1*x2__He1_He1", (
            f"seed={seed}: top cross-basis winner should be x1*x2__He1_He1, " f"got {top['engineered_col']}; full ranking:\n{sc}"
        )
        # MI should be substantial (>= 0.4 nats on a clean XOR with n=2500).
        assert top["engineered_mi"] >= 0.4, f"seed={seed}: XOR He_1*He_1 engineered_mi {top['engineered_mi']:.3f} " f"should clear 0.4 on n=2500"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_xor_term_enters_augmented_frame(self, seed):
        """The He_1*He_1 XOR cross-basis term enters the hybrid-augmented frame."""
        _, _, hybrid = _import_fe()
        X, y = _build_xor(seed)
        X_aug, _uni_sc, cross_sc = hybrid(
            X,
            y.values,
            cols=["x1", "x2", "noise_a", "noise_b"],
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            top_pair_count=3,
            pair_max_degree=2,
            pair_min_uplift=1.05,
            pair_min_abs_mi_frac=0.1,
        )
        pair_cols = [c for c in X_aug.columns if "*" in c and "__" in c]
        # The HE_1*HE_1 XOR term must be among the appended pair columns.
        # Order of legs (x1*x2 vs x2*x1) depends on seed pool order; allow both.
        ok = any((("x1*x2__He1_He1" == c) or ("x2*x1__He1_He1" == c)) for c in pair_cols)
        assert ok, f"seed={seed}: XOR cross-basis He1*He1 should be in augmented " f"frame, got pair cols {pair_cols}; cross_sc:\n{cross_sc.head(6)}"


# ---------------------------------------------------------------------------
# Contract 3: He_2 product saddle -- He_2 * He_2 cross-basis discovered
# ---------------------------------------------------------------------------


class TestSaddleDiscovery:
    """``y = sign((x1^2-1)(x2^2-1))`` (saddle) is dominated by the He_2*He_2 (or adjacent) cross-basis cell."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_he2_he2_term_top_ranked(self, seed):
        """He_2*He_2 or an adjacent cell is the top-ranked cross-basis cell, with substantial engineered MI."""
        gen_pair, score_pair, _ = _import_fe()
        X, y = _build_he2_pair_saddle(seed)
        eng = gen_pair(X, pairs=[("x1", "x2")], max_degree=2, basis="hermite")
        sc = score_pair(X[["x1", "x2"]], eng, y.values)
        top = sc.iloc[0]
        # Either He_2*He_2 (the exact target form) or He_2*He_1/He_1*He_2
        # may rank top depending on quantile binning noise; both reflect
        # the saddle signal in the basis.
        assert top["engineered_col"] in {
            "x1*x2__He2_He2",
            "x1*x2__He2_He1",
            "x1*x2__He1_He2",
        }, (
            f"seed={seed}: top He_2 saddle winner should be He_2*He_2 or " f"adjacent cell, got {top['engineered_col']}; full ranking:\n{sc}"
        )
        assert top["engineered_mi"] >= 0.20, f"seed={seed}: saddle top cross engineered_mi " f"{top['engineered_mi']:.3f} should clear 0.20 at n=2500"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_saddle_cross_enters_augmented(self, seed):
        """At least one (x1, x2) He_? cross-basis cell enters the hybrid-augmented frame for the saddle signal."""
        _, _, hybrid = _import_fe()
        X, y = _build_he2_pair_saddle(seed)
        X_aug, _, cross_sc = hybrid(
            X,
            y.values,
            cols=["x1", "x2", "noise_a", "noise_b"],
            degrees=(2, 3),
            basis="hermite",
            top_k=4,
            top_pair_count=3,
            pair_max_degree=2,
            pair_min_uplift=1.05,
            pair_min_abs_mi_frac=0.1,
        )
        pair_cols = [c for c in X_aug.columns if "*" in c and "__" in c]
        # At least one (x1, x2) cross-basis He_? cell entered.
        ok = any((("x1*x2__" in c) or ("x2*x1__" in c)) for c in pair_cols)
        assert ok, f"seed={seed}: saddle cross-basis should be in augmented frame, " f"got pair cols {pair_cols}; cross_sc:\n{cross_sc.head(6)}"


# ---------------------------------------------------------------------------
# Contract 4: mixed univariate + cross-basis signals both surface
# ---------------------------------------------------------------------------


class TestMixedSignalsBothSurfaced:
    """A dataset with both a univariate and a cross-basis signal surfaces winners from both stages."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_both_univariate_and_cross_winners_appear(self, seed):
        """x_solo__He2 (univariate) and x_pa*x_pb__He1_He1 (cross-basis) both enter the augmented frame."""
        _, _, hybrid = _import_fe()
        X, y = _build_mixed_signals(seed)
        X_aug, uni_sc, cross_sc = hybrid(
            X,
            y.values,
            cols=["x_solo", "x_pa", "x_pb", "noise_a", "noise_b"],
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            top_pair_count=3,
            top_pair_seed_k=5,
            pair_max_degree=2,
            pair_min_uplift=1.05,
            pair_min_abs_mi_frac=0.1,
        )
        added_cols = [c for c in X_aug.columns if "__" in c]
        # Univariate-side: x_solo__He2 (the univariate He_2 signal) must
        # have entered.
        uni_added = [c for c in added_cols if "*" not in c]
        assert any("x_solo__He2" == c for c in uni_added), (
            f"seed={seed}: x_solo__He2 should be in augmented frame as the " f"univariate He_2 winner; uni_added={uni_added}; uni_sc:\n" f"{uni_sc.head(6)}"
        )
        # Pair-side: x_pa*x_pb__He1_He1 (or x_pb*x_pa__He1_He1) must be there.
        pair_added = [c for c in added_cols if "*" in c]
        ok_pair = any((("x_pa*x_pb__He1_He1" == c) or ("x_pb*x_pa__He1_He1" == c)) for c in pair_added)
        assert ok_pair, (
            f"seed={seed}: x_pa*x_pb__He1_He1 should be in augmented frame "
            f"as the cross-basis XOR winner; pair_added={pair_added}; "
            f"cross_sc:\n{cross_sc.head(6)}"
        )


# ---------------------------------------------------------------------------
# Contract 5: downstream LogReg AUC lift on XOR
# ---------------------------------------------------------------------------


class TestXorLogRegLift:
    """Cross-basis-augmented LogReg measurably lifts holdout AUC over raw LogReg on the unsolvable XOR target."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lifts_with_cross_basis_fe(self, seed):
        """Cross-basis-augmented LogReg clears 0.85 AUC and beats raw LogReg by >= +0.20 on XOR."""
        _, _, hybrid = _import_fe()
        X, y = _build_xor(seed, n=3000)
        n_train = 2000
        # Baseline: raw LogReg on x1, x2, noise_a, noise_b.
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        m_raw = LogisticRegression(max_iter=500).fit(Xtr.to_numpy(), ytr.to_numpy())
        auc_raw = roc_auc_score(yte.to_numpy(), m_raw.predict_proba(Xte.to_numpy())[:, 1])
        # Hybrid FE on the full frame, then refit LogReg on the augmented
        # support. The augmented frame must contain x1*x2__He1_He1.
        X_aug_joint, _, cross_sc = hybrid(
            X,
            y.values,
            cols=["x1", "x2", "noise_a", "noise_b"],
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            top_pair_count=3,
            pair_max_degree=2,
            pair_min_uplift=1.05,
            pair_min_abs_mi_frac=0.1,
        )
        Xtr_aug = X_aug_joint.iloc[:n_train]
        Xte_aug = X_aug_joint.iloc[n_train:]
        m_aug = LogisticRegression(max_iter=500).fit(Xtr_aug.to_numpy(), ytr.to_numpy())
        auc_aug = roc_auc_score(yte.to_numpy(), m_aug.predict_proba(Xte_aug.to_numpy())[:, 1])
        # XOR is unsolvable by linear LogReg on raw -- AUC ~ 0.50.
        # With He_1(x1) * He_1(x2) feature added it should jump to >= 0.85.
        assert auc_aug >= 0.85, (
            f"seed={seed}: augmented LogReg AUC {auc_aug:.3f} should clear " f"0.85 on XOR with cross-basis FE; cross_sc:\n{cross_sc.head(5)}"
        )
        assert auc_aug > auc_raw + 0.20, f"seed={seed}: cross-basis FE should lift LogReg holdout AUC " f">= +0.20 on XOR. raw={auc_raw:.3f}, aug={auc_aug:.3f}"


# ---------------------------------------------------------------------------
# Contract 6: pure-noise pairs are pruned by the abs MI floor
# ---------------------------------------------------------------------------


class TestNoisePairPruned:
    """Pure-noise cross-basis pairs are pruned by the absolute MI floor even when relative uplift looks attractive."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_noise_pair_filtered_by_abs_floor(self, seed):
        """Real XOR signal on (x1,x2) + spectator noise cols. The genuine
        x1*x2__He1_He1 winner must enter X_aug while noise_a*noise_b and
        every other noise*noise cross-basis term is filtered by the
        absolute MI floor (which is anchored on the strong x1*x2 term).
        """
        _, _, hybrid = _import_fe()
        X, y = _build_xor_with_noise_pair(seed)
        X_aug, _uni_sc, cross_sc = hybrid(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            top_pair_count=5,
            top_pair_seed_k=6,
            pair_max_degree=2,
            pair_min_uplift=1.05,
            pair_min_abs_mi_frac=0.1,
        )
        new_cols = [c for c in X_aug.columns if c not in X.columns]
        # Real XOR pair winner must enter.
        pair_added = [c for c in new_cols if "*" in c]
        ok_real = any((("x1*x2__He1_He1" == c) or ("x2*x1__He1_He1" == c)) for c in pair_added)
        assert ok_real, f"seed={seed}: x1*x2 He1*He1 (the genuine signal) should enter " f"augmented frame; pair_added={pair_added}"

        # NO noise*noise pair term may slip in: every term whose BOTH legs
        # start with 'noise_' must be filtered.
        def _both_legs_noise(name: str) -> bool:
            """Check whether a cross-basis column name's two source legs both start with 'noise_'."""
            head = name.split("__", 1)[0]
            if "*" not in head:
                return False
            a, b = head.split("*", 1)
            return a.startswith("noise_") and b.startswith("noise_")

        noise_pair_added = [c for c in pair_added if _both_legs_noise(c)]
        assert not noise_pair_added, (
            f"seed={seed}: noise-noise cross-basis terms should be filtered "
            f"by the absolute MI floor; got {noise_pair_added}; "
            f"cross_sc:\n{cross_sc.head(8)}"
        )


class TestMixedBasisPairCrossReplay:
    """Pair-cross-basis legs are routed INDEPENDENTLY (Gaussian leg -> Hermite,
    bounded leg -> Chebyshev), so a single engineered product can mix families:
    ``He_a(x_i) * T_b(x_j)``. The column NAME suffix is cosmetically lossy (it
    reuses leg-i's basis code for both legs), so the recipe MUST carry the true
    per-leg bases in ``extra['basis_i']`` / ``extra['basis_j']`` and replay from
    THERE -- never re-derive the basis from the name. This pins that contract: a
    future refactor that parses the basis from the (lossy) name would silently
    replay He(x_j) where fit used T(x_j) -- a transform-time correctness bug.
    Benched value: mixed-basis recovers a mixed-domain product (He2*T2) at OOS
    |corr| ~0.98 vs ~0.85 if both legs were forced onto one family.
    """

    def test_mixed_basis_pair_recipe_replays_exactly(self):
        """A mixed-domain He_2(Gaussian)*T_2(bounded) product replays from the recipe's per-leg extra bases, not the lossy name."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            hybrid_orth_mi_pair_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        rng = np.random.default_rng(0)
        n = 3000
        x_i = rng.standard_normal(n)  # Gaussian domain -> Hermite-natural
        x_j = rng.uniform(-1.0, 1.0, n)  # bounded domain  -> Chebyshev-natural
        # He_2(x_i) * T_2(x_j): a genuinely mixed-domain product target.
        y = (((x_i * x_i - 1.0) * (2.0 * x_j * x_j - 1.0) + 0.2 * rng.standard_normal(n)) > 0).astype(int)
        X = pd.DataFrame({"x_i": x_i, "x_j": x_j})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_aug, _uni, _cross, recipes = hybrid_orth_mi_pair_fe_with_recipes(
                X,
                y,
                degrees=(2,),
                pair_max_degree=2,
                pair_min_uplift=1.0,
                top_pair_count=3,
            )

        pair_recipes = [r for r in recipes if getattr(r, "kind", None) == "orth_pair_cross"]
        assert pair_recipes, "the mixed-domain product target should engineer at least one " "orth_pair_cross feature"
        saw_mixed = False
        for r in pair_recipes:
            basis_i = r.extra.get("basis_i")
            basis_j = r.extra.get("basis_j")
            assert basis_i is not None and basis_j is not None, (
                f"orth_pair_cross recipe {r.name!r} must carry both per-leg bases " f"in extra (not derive from the lossy name); extra={dict(r.extra)}"
            )
            if basis_i != basis_j:
                saw_mixed = True
            # Replay must reproduce the fit-time engineered column, using the per-leg bases from
            # extra (not the name). Tolerance 5e-6 (2026-07-13), not bit-identical: under the default
            # MLFRAME_CRIT_DTYPE_RELAXED=1 both fit (generate_pair_cross_basis_features) and replay
            # (_apply_orth_pair_cross) now operate on the operand at the SAME relaxed (f32) dtype, but
            # _eval_orth_basis_column's own internal `np.asarray(x, dtype=np.float64)` still runs the
            # polynomial recurrence in float64 arithmetic on those f32-rounded values regardless of
            # caller dtype -- a smaller second-order version of the host/device arithmetic-precision
            # gap documented in _gpu_resident_cross_basis.py's build_leg_product_matrix_gpu docstring.
            # Measured worst case on this fixture: 7.91e-7 absolute, near a zero-crossing where rtol
            # alone contributes ~0 -- atol must independently cover that (2e-6 gives ~2.5x margin);
            # 5e-6 rtol matches the tolerance already established for the same bug class in
            # test_device_born_cross_basis_parity.py.
            replay = np.asarray(apply_recipe(r, X), dtype=float)
            fit_vals = np.asarray(X_aug[r.name], dtype=float)
            assert np.allclose(replay, fit_vals, rtol=5e-6, atol=2e-6), (
                f"mixed-basis pair recipe {r.name!r} (basis_i={basis_i}, "
                f"basis_j={basis_j}) replay drifted from fit: "
                f"max|d|={float(np.max(np.abs(replay - fit_vals)))}; "
                f"extra={dict(r.extra)} -- the recipe must replay from the "
                f"per-leg bases in extra, NOT the cosmetically-lossy name suffix."
            )
        assert saw_mixed, (
            "on a Gaussian x bounded mixed-domain product the two legs should "
            "route to DIFFERENT bases (Hermite x Chebyshev); got only same-basis "
            f"recipes: {[(r.extra.get('basis_i'), r.extra.get('basis_j')) for r in pair_recipes]}"
        )
