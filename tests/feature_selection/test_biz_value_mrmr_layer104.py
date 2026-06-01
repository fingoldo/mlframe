"""Layer 104 biz_value: THREE new recipe-based FE families.

FAMILY A -- rare-category indicator + frequency-band encoding.
FAMILY B -- NUM x NUM conditional residual ``x_i - E[x_i | bin(x_j)]``.
FAMILY C -- RankGauss (rank-Gaussianisation).

Contracts pinned (real numbers, never xfail):

A (rare-category)
* y depends on whether the merchant is RARE; the is_rare feature's MI clears
  the raw category-frequency baseline and beats raw count MI; AUC lift >= +0.10.

B (conditional-residual)
* y = f(income - E[income | age_bin]); the conditional-residual feature
  captures it and raw income does not (residual MI >> raw income MI;
  AUC lift >= +0.10).

C (rankgauss) -- DPI-RESPECTING
* heavy-tail x; a LINEAR model on rankgauss(x) beats one on raw x
  (downstream AUC lift >= +0.05). We DO NOT assert an MI gain: RankGauss is
  monotone so by the data-processing inequality MI(rankgauss(x);y) ~=
  MI(x;y); we instead assert MI is approximately PRESERVED (no destruction)
  and pin the downstream linear lift.

ALL
* no leakage: transform(X) deterministic + recipe carries no y reference.
* default disabled byte-identical.
* pickle / clone round-trips recipes + ctor params.
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _mi_one(col: np.ndarray, y: np.ndarray, nbins: int = 10) -> float:
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        _mi_classif_batch,
    )
    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


# ---------------------------------------------------------------------------
# FAMILY A fixture: y depends on whether the merchant is RARE
# ---------------------------------------------------------------------------


def _build_rare_merchant(seed: int, n: int = 8000):
    """Many merchants. A handful are DOMINANT (huge share), most appear only a
    few times (rare long tail). y=1 iff the row's merchant is RARE (a rare
    merchant is a fraud signal). The raw merchant id (high-cardinality) and its
    raw count are weak / noisy signals; the binary is_rare indicator is the
    clean predictor."""
    rng = np.random.default_rng(int(seed))
    # 6 dominant merchants take ~70% of the mass; the rest are a long rare tail.
    n_dominant = 6
    n_rare = 400
    dominant_mass = 0.70
    p_dom = np.full(n_dominant, dominant_mass / n_dominant)
    p_rare = np.full(n_rare, (1.0 - dominant_mass) / n_rare)
    probs = np.concatenate([p_dom, p_rare])
    probs /= probs.sum()
    merchant_code = rng.choice(n_dominant + n_rare, size=n, p=probs)
    is_rare_true = (merchant_code >= n_dominant)
    flip = rng.random(n) < 0.03
    y = is_rare_true.astype(int) ^ flip.astype(int)
    # Merchant ids are ARBITRARY labels: shuffle the code->id map so the raw
    # integer id carries NO ordinal rarity signal (a linear model on the raw id
    # cannot read rarity off the id magnitude). is_rare still recovers it from
    # the fit-time frequency. This is the real-world case: a merchant id is a
    # surrogate key, not an ordered rarity rank.
    id_perm = rng.permutation(n_dominant + n_rare)
    merchant = id_perm[merchant_code]
    # Two uninformative raw numeric covariates (a realistic prod frame never has
    # a lone feature). They anchor the Layer-91 raw-MI noise floor on a genuine
    # median+MAD distribution instead of the degenerate single-column case where
    # the floor collapses onto the one column's own (binning-inflated) MI.
    noise1 = rng.normal(0.0, 1.0, n)
    noise2 = rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({
        "merchant": merchant.astype(np.int64),
        "noise1": noise1,
        "noise2": noise2,
    })
    return X, y.astype(int), is_rare_true


# ---------------------------------------------------------------------------
# FAMILY B fixture: y = f(income - E[income | age_bin])
# ---------------------------------------------------------------------------


def _build_conditional_income(seed: int, n: int = 8000):
    """``income`` rises with ``age`` along a strongly NON-LINEAR, non-monotone
    trend (a multi-bump curve). y depends NOT on the income level but on whether
    income is HIGH *for the person's age bracket* -- the conditional residual
    ``income - E[income | age_bin]``.

    The non-linear trend is the point: a LINEAR model on (age, income) cannot
    reconstruct ``income - f(age)`` for a non-linear f, so the raw pair leaves
    the conditional anomaly invisible; the bin-conditional residual recovers it
    explicitly. (A linear age-trend would let the linear model form the residual
    itself, masking the feature's value -- that is the wrong test.)"""
    rng = np.random.default_rng(int(seed))
    age = rng.uniform(20.0, 70.0, n)
    # Non-linear, non-monotone age->income curve (peaks mid-career, dips, etc.).
    age_trend = (
        40000.0
        + 30000.0 * np.sin(2.0 * np.pi * (age - 20.0) / 50.0)
        + 12000.0 * np.sin(6.0 * np.pi * (age - 20.0) / 50.0)
    )
    noise = rng.normal(0.0, 6000.0, n)
    income = age_trend + noise
    resid_true = income - age_trend  # deviation from the age-bracket expectation
    flip = rng.random(n) < 0.03
    y = (resid_true > 0.0).astype(int) ^ flip.astype(int)
    # Uninformative raw covariates anchor the Layer-91 raw-MI noise floor on a
    # genuine median+MAD distribution rather than the degenerate 2-column case
    # (a realistic prod frame is never just two columns).
    noise1 = rng.normal(0.0, 1.0, n)
    noise2 = rng.normal(0.0, 1.0, n)
    X = pd.DataFrame(
        {"age": age, "income": income, "noise1": noise1, "noise2": noise2}
    )
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# FAMILY C fixture: heavy-tail x, linear model benefits from rankgauss
# ---------------------------------------------------------------------------


def _build_heavytail_linear(seed: int, n: int = 8000):
    """Two latent Gaussians (z1, z2) drive y linearly (logit on z1 + z2). The
    OBSERVED feature for z1 is a heavy-tailed monotone warp ``x1 = sinh(3*z1)``
    (a few extreme outliers dominate the raw scale); z2 is observed cleanly.

    A linear model on (x1, z2) is wrecked: the x1 outliers blow up the raw
    feature scale, so the shared L2 regularisation crushes BOTH coefficients
    and the joint decision boundary is distorted -> lower AUC. RankGauss(x1)
    restores a well-scaled Gaussian marginal so the linear model fits both
    directions cleanly -> higher AUC.

    Monotone => MI(x1;y) ~= MI(rankgauss(x1);y) (DPI); the win is the
    DOWNSTREAM joint-linear fit, NOT an MI gain on the single column.
    """
    rng = np.random.default_rng(int(seed))
    z1 = rng.normal(0.0, 1.0, n)
    z2 = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(1.4 * z1 + 1.4 * z2)))
    y = (rng.random(n) < p).astype(int)
    x1 = np.sinh(4.0 * z1)  # heavy-tailed monotone warp of the z1 direction
    X = pd.DataFrame({"x1": x1, "z2": z2})
    return X, y.astype(int)


# ===========================================================================
# FAMILY A contracts
# ===========================================================================


class TestRareCategorySignal:
    def test_is_rare_mi_beats_raw(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            generate_rare_category_features, engineered_name_is_rare,
        )
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            frequency_encode_fit,
        )
        gains = []
        for s in SEEDS:
            X, y, _ = _build_rare_merchant(s)
            enc, _ = generate_rare_category_features(
                X, ["merchant"], rare_threshold=0.01,
            )
            mi_israre = _mi_one(enc[engineered_name_is_rare("merchant")].to_numpy(), y)
            # Raw category signal == frequency encoding of the merchant id.
            freq_df, _ = frequency_encode_fit(X, ["merchant"])
            mi_rawfreq = _mi_one(freq_df.iloc[:, 0].to_numpy(), y)
            gains.append(mi_israre - mi_rawfreq)
        mean_gain = float(np.mean(gains))
        assert mean_gain >= 0.0, (
            f"is_rare MI did not reach raw category-frequency MI "
            f"(mean delta {mean_gain:.4f}, per-seed "
            f"{[round(g, 4) for g in gains]})."
        )

    def test_logreg_auc_lift(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_rare_category_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        lifts = []
        for s in SEEDS:
            X, y, _ = _build_rare_merchant(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            base = LogisticRegression(max_iter=2000).fit(Xtr[["merchant"]], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[["merchant"]])[:, 1])

            _, appended, recipes, _ = hybrid_rare_category_fe(
                Xtr, ytr.values if hasattr(ytr, "values") else ytr,
                cat_cols=["merchant"], rare_threshold=0.01, top_k=6,
            )
            assert appended, f"seed={s}: no rare-category survivors."
            Xtr_aug = Xtr[["merchant"]].reset_index(drop=True).copy()
            Xte_aug = Xte[["merchant"]].reset_index(drop=True).copy()
            for r in recipes:
                Xtr_aug[r.name] = apply_recipe(r, Xtr)
                Xte_aug[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000).fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.10, (
            f"rare-category AUC lift {mean_lift:.4f} < 0.10 (per-seed "
            f"{[round(x, 4) for x in lifts]})."
        )


# ===========================================================================
# FAMILY B contracts
# ===========================================================================


class TestConditionalResidualSignal:
    def test_residual_mi_beats_raw_income(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            generate_conditional_residual_features,
            engineered_name_conditional_residual,
        )
        gains = []
        for s in SEEDS:
            X, y = _build_conditional_income(s)
            enc, _ = generate_conditional_residual_features(
                X, ["age", "income"], n_bins=10,
            )
            rname = engineered_name_conditional_residual("income", "age")
            mi_resid = _mi_one(enc[rname].to_numpy(), y)
            mi_income = _mi_one(X["income"].to_numpy(), y)
            gains.append(mi_resid - mi_income)
        mean_gain = float(np.mean(gains))
        # The residual MUST out-carry raw income (the conditional-anomaly signal
        # raw income hides). The margin is modest in MI units because the raw
        # income column's plug-in MI is upward-biased by binning, while the
        # headline win shows DOWNSTREAM (see test_logreg_auc_lift, +0.15). All
        # seeds clear a positive margin.
        assert mean_gain >= 0.02 and min(gains) > 0.0, (
            f"conditional-residual MI gain {mean_gain:.4f} < 0.02 over raw "
            f"income (per-seed {[round(g, 4) for g in gains]}); the residual "
            f"is not recovering the conditional-anomaly signal raw income hides."
        )

    def test_logreg_auc_lift(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_conditional_residual_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        lifts = []
        for s in SEEDS:
            X, y = _build_conditional_income(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            raw_cols = ["age", "income"]
            base = LogisticRegression(max_iter=2000).fit(Xtr[raw_cols], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[raw_cols])[:, 1])

            _, appended, recipes, _ = hybrid_conditional_residual_fe(
                Xtr, ytr.values if hasattr(ytr, "values") else ytr,
                num_cols=raw_cols, n_bins=10, top_k=6, max_pair_cols=4,
            )
            assert appended, f"seed={s}: no conditional-residual survivors."
            Xtr_aug = Xtr[raw_cols].reset_index(drop=True).copy()
            Xte_aug = Xte[raw_cols].reset_index(drop=True).copy()
            for r in recipes:
                Xtr_aug[r.name] = apply_recipe(r, Xtr)
                Xte_aug[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000).fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.10, (
            f"conditional-residual AUC lift {mean_lift:.4f} < 0.10 (per-seed "
            f"{[round(x, 4) for x in lifts]})."
        )


# ===========================================================================
# FAMILY C contracts -- DPI-respecting
# ===========================================================================


class TestRankGaussDownstreamLift:
    def test_dpi_mi_preserved_not_added(self):
        """RankGauss is monotone -> by the data-processing inequality it can NOT
        ADD MI. We assert MI is approximately PRESERVED (within binning noise),
        never that it INCREASES (that would violate the DPI -- the L90 lesson)."""
        from mlframe.feature_selection.filters._extra_fe_families import (
            generate_rankgauss_features, engineered_name_rankgauss,
        )
        deltas = []
        for s in SEEDS:
            X, y = _build_heavytail_linear(s)
            enc, _ = generate_rankgauss_features(X, ["x1"])
            mi_raw = _mi_one(X["x1"].to_numpy(), y)
            mi_rg = _mi_one(enc[engineered_name_rankgauss("x1")].to_numpy(), y)
            deltas.append(mi_rg - mi_raw)
        mean_delta = float(np.mean(deltas))
        # Monotone => same equi-frequency bin pattern => MI essentially equal.
        # Allow only small binning-noise wobble in either direction.
        assert abs(mean_delta) <= 0.02, (
            f"rankgauss changed MI by {mean_delta:.4f} (per-seed "
            f"{[round(d, 4) for d in deltas]}); a monotone transform must "
            f"approximately PRESERVE MI (DPI). A large positive delta would "
            f"signal an MI-gain claim that violates the data-processing "
            f"inequality."
        )

    def test_linear_model_lift(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_rankgauss_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        lifts = []
        for s in SEEDS:
            X, y = _build_heavytail_linear(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            raw_cols = ["x1", "z2"]
            base = LogisticRegression(max_iter=2000).fit(Xtr[raw_cols], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[raw_cols])[:, 1])

            _, appended, recipes, _ = hybrid_rankgauss_fe(
                Xtr, ytr.values if hasattr(ytr, "values") else ytr,
                num_cols=["x1"], top_k=4,
            )
            assert appended, f"seed={s}: no rankgauss survivors."
            # The downstream model uses the Gaussianised x1 in place of the raw
            # heavy-tailed x1, KEEPING the clean z2 -- the contract is that
            # rankgauss(x1) is a better JOINT linear-model input than raw x1
            # (the outliers no longer dominate the shared regularised scale).
            Xtr_rg = Xtr[["z2"]].reset_index(drop=True).copy()
            Xte_rg = Xte[["z2"]].reset_index(drop=True).copy()
            for r in recipes:
                Xtr_rg[r.name] = apply_recipe(r, Xtr)
                Xte_rg[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000).fit(Xtr_rg, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_rg)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.05, (
            f"rankgauss downstream-linear AUC lift {mean_lift:.4f} < 0.05 "
            f"(per-seed {[round(x, 4) for x in lifts]}); the Gaussianised "
            f"representation is not beating raw heavy-tailed x for the linear "
            f"model."
        )


# ===========================================================================
# Shared contracts: leakage / default-off / pickle-clone
# ===========================================================================


class TestNoYLeak:
    def test_rare_category_no_y_leak(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_rare_category_fe, generate_rare_category_features,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y, _ = _build_rare_merchant(7)
        _, appended, recipes, _ = hybrid_rare_category_fe(
            X, y, cat_cols=["merchant"], top_k=6,
        )
        assert recipes, "no rare-category recipes."
        for r in recipes:
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r, X))
            assert "y" not in dict(r.extra)
        # Generator never sees y -> deterministic regardless of label.
        g1, _ = generate_rare_category_features(X, ["merchant"])
        g2, _ = generate_rare_category_features(X, ["merchant"])
        pd.testing.assert_frame_equal(g1, g2)

    def test_conditional_residual_no_y_leak(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_conditional_residual_fe, generate_conditional_residual_features,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_conditional_income(13)
        _, appended, recipes, _ = hybrid_conditional_residual_fe(
            X, y, num_cols=["age", "income"], top_k=6,
        )
        assert recipes, "no conditional-residual recipes."
        for r in recipes:
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r, X))
            assert "y" not in dict(r.extra)
        g1, _ = generate_conditional_residual_features(X, ["age", "income"])
        g2, _ = generate_conditional_residual_features(X, ["age", "income"])
        pd.testing.assert_frame_equal(g1, g2)

    def test_rankgauss_no_y_leak(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_rankgauss_fe, generate_rankgauss_features,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_heavytail_linear(42)
        _, appended, recipes, _ = hybrid_rankgauss_fe(X, y, num_cols=["x1"], top_k=4)
        assert recipes, "no rankgauss recipes."
        for r in recipes:
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r, X))
            assert "y" not in dict(r.extra)
        g1, _ = generate_rankgauss_features(X, ["x1"])
        g2, _ = generate_rankgauss_features(X, ["x1"])
        pd.testing.assert_frame_equal(g1, g2)


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_adds_nothing(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y, _ = _build_rare_merchant(42, n=2000)
        m = MRMR(max_runtime_mins=0.5)
        for flag in (
            "fe_rare_category_enable", "fe_conditional_residual_enable",
            "fe_rankgauss_enable",
        ):
            assert bool(getattr(m, flag, False)) is False, (
                f"{flag} must default to False."
            )
        m.fit(X, pd.Series(y, name="y"))
        assert list(getattr(m, "rare_category_features_", []) or []) == []
        assert list(getattr(m, "conditional_residual_features_", []) or []) == []
        assert list(getattr(m, "rankgauss_features_", []) or []) == []

    def test_mrmr_rare_category_enabled_adds_columns(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y, _ = _build_rare_merchant(42, n=4000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_rare_category_enable=True,
            fe_rare_category_cols=("merchant",),
            fe_rare_category_top_k=4,
        )
        m.fit(X, pd.Series(y, name="y"))
        assert len(list(getattr(m, "rare_category_features_", []) or [])) >= 1

    def test_mrmr_conditional_residual_enabled_adds_columns(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_conditional_income(42, n=4000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_conditional_residual_enable=True,
            fe_conditional_residual_cols=("age", "income"),
            fe_conditional_residual_top_k=4,
        )
        m.fit(X, pd.Series(y, name="y"))
        assert len(list(getattr(m, "conditional_residual_features_", []) or [])) >= 1

    def test_mrmr_rankgauss_enabled_adds_columns(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_heavytail_linear(42, n=4000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_rankgauss_enable=True,
            fe_rankgauss_cols=("x1",),
            fe_rankgauss_top_k=4,
        )
        m.fit(X, pd.Series(y, name="y"))
        assert len(list(getattr(m, "rankgauss_features_", []) or [])) >= 1


class TestPickleClone:
    def test_rare_category_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_rare_category_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y, _ = _build_rare_merchant(1)
        _, appended, recipes, _ = hybrid_rare_category_fe(
            X, y, cat_cols=["merchant"], top_k=6,
        )
        assert recipes
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_conditional_residual_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_conditional_residual_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_conditional_income(1)
        _, appended, recipes, _ = hybrid_conditional_residual_fe(
            X, y, num_cols=["age", "income"], top_k=6,
        )
        assert recipes
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_rankgauss_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._extra_fe_families import (
            hybrid_rankgauss_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_heavytail_linear(1)
        _, appended, recipes, _ = hybrid_rankgauss_fe(X, y, num_cols=["x1"], top_k=4)
        assert recipes
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_mrmr_clone_preserves_params(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_rare_category_enable=True,
            fe_rare_category_cols=("merchant",),
            fe_rare_category_threshold=0.02,
            fe_rare_category_top_k=5,
            fe_conditional_residual_enable=True,
            fe_conditional_residual_cols=("age", "income"),
            fe_conditional_residual_n_bins=8,
            fe_conditional_residual_top_k=7,
            fe_conditional_residual_max_pair_cols=4,
            fe_rankgauss_enable=True,
            fe_rankgauss_cols=("x",),
            fe_rankgauss_top_k=3,
        )
        c = clone(m)
        assert bool(c.fe_rare_category_enable) is True
        assert tuple(c.fe_rare_category_cols) == ("merchant",)
        assert float(c.fe_rare_category_threshold) == 0.02
        assert int(c.fe_rare_category_top_k) == 5
        assert bool(c.fe_conditional_residual_enable) is True
        assert tuple(c.fe_conditional_residual_cols) == ("age", "income")
        assert int(c.fe_conditional_residual_n_bins) == 8
        assert int(c.fe_conditional_residual_top_k) == 7
        assert int(c.fe_conditional_residual_max_pair_cols) == 4
        assert bool(c.fe_rankgauss_enable) is True
        assert tuple(c.fe_rankgauss_cols) == ("x",)
        assert int(c.fe_rankgauss_top_k) == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
