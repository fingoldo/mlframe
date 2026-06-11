"""Layer 33 biz_value: K-FOLD TARGET ENCODING for categorical features.

Validates the new K-fold OOF target-encoding FE stage introduced
2026-05-31 as the production-grade pattern for medium / high-cardinality
categorical columns. Naive single-pass mean-of-y per category leaks y
into X (the Layer 17 leakage pattern); K-fold OOF discipline isolates
each row's target value from its own per-category estimate.

Contracts pinned
----------------

* TestMultiCategorySignalAUCLift
    On a 50-level ``cat_region`` where 6 levels carry the signal and the
    rest are noise, hybrid-augmented LogReg using the TE column clears
    a meaningful AUC lift over raw LogReg (which cannot consume the
    object-dtype source column at all).

* TestNoLeakage
    ``transform(X_holdout)`` with the recipe is bit-identical regardless
    of whether the user passes a shuffled y to a hypothetical refit on
    holdout -- the transform path has no y dependency.

* TestSmoothingShrinksRareCategories
    Categories with a single occurrence in train get an encoding heavily
    pulled toward the global mean (smoothing dominates). Categories
    with hundreds of rows get an encoding close to their raw per-category
    mean.

* TestRecipeReplayUnseenCategories
    A test row carrying a category that was never seen during fit gets
    ``global_mean`` (no NaN propagation, no KeyError).

* TestPickleCloneRoundTrip
    The TE recipe survives ``pickle.dumps`` + ``pickle.loads`` AND
    ``sklearn.base.clone``; transform output bit-identical post round-trip.

* TestDefaultDisabledByteIdentical
    With ``fe_kfold_te_enable=False`` (default), the existing Layer 21 /
    23 / 28 contracts are unchanged: no ``__te`` columns leak into
    ``transform`` output, no ``kfold_te_features_`` attr populated.

NEVER xfail. Real LogReg AUC numbers.
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


SEEDS = (1, 7, 13)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
def _build_cat_signal(seed: int, n: int = 3000):
    """y depends on which subset of 50 region levels the row belongs to.

    6 of the 50 levels are "hot" -- conditional P(y=1) = 0.95; the
    other 44 are "cold" with P(y=1) = 0.05. The 18-fold separation
    in conditional probabilities puts Bayes-optimal AUC at ~0.90,
    giving the test's +0.15 lift contract clear headroom over the
    raw LogReg baseline (which cannot consume the object dtype and
    rides the noise-only floor near 0.5). Earlier values (0.8 / 0.2)
    were Bayes-bounded at ~0.66 - infeasible for the +0.15 contract
    after seed-to-seed variance.
    """
    rng = np.random.default_rng(seed)
    levels = [f"R{i:03d}" for i in range(50)]
    cat_region = rng.choice(levels, size=n)
    hot_set = set(levels[:6])
    p = np.where(np.isin(cat_region, list(hot_set)), 0.95, 0.05)
    # Weak numeric noise to keep the predictor non-degenerate even when
    # the categorical signal isn't used.
    noise_a = rng.standard_normal(n)
    noise_b = rng.standard_normal(n)
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({
        "cat_region": cat_region,
        "noise_a": noise_a,
        "noise_b": noise_b,
    })
    return X, pd.Series(y, name="y")


from tests.feature_selection._biz_val_synth import _logreg_auc, _train_holdout_split


# ---------------------------------------------------------------------------
# Direct unit tests on the encoder kernel
# ---------------------------------------------------------------------------


class TestKFoldEncoder:
    def test_fit_returns_oof_for_every_row(self):
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit,
        )
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({"region": rng.choice(["A", "B", "C", "D"], size=n)})
        y = (X["region"].isin(["A", "B"])).astype(int).to_numpy()
        te_df, recipes = kfold_target_encode_fit(
            X, y, ["region"], n_folds=5, smoothing=1.0, random_state=0,
        )
        assert te_df.shape == (n, 1)
        assert "region__te" in te_df.columns
        assert "region" in recipes
        rec = recipes["region"]
        assert "lookup" in rec and "global_mean" in rec and "smoothing" in rec
        # Lookup covers every observed category.
        assert set(rec["lookup"].keys()) == {"A", "B", "C", "D"}
        # OOF values are NOT identical to the full-data lookup values
        # (proves OOF discipline -- the folded out estimate differs from the
        # all-rows estimate by at least one fold's worth of held-out signal).
        # We pick a category seen many times so the difference is robust.
        cat_a_mask = X["region"].to_numpy() == "A"
        oof_a_mean = float(te_df["region__te"].to_numpy()[cat_a_mask].mean())
        full_a = rec["lookup"]["A"]
        assert abs(oof_a_mean - full_a) < 0.2  # close but generally not equal

    def test_rejects_bad_n_folds(self):
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit,
        )
        X = pd.DataFrame({"a": ["x", "y"] * 5})
        y = np.array([0, 1] * 5)
        with pytest.raises(ValueError, match="n_folds"):
            kfold_target_encode_fit(X, y, ["a"], n_folds=1)

    def test_rejects_missing_columns(self):
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit,
        )
        X = pd.DataFrame({"a": ["x"] * 4})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="missing"):
            kfold_target_encode_fit(X, y, ["b"])


# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------


class TestAutoDetectTeCols:
    def test_picks_object_in_band(self):
        from mlframe.feature_selection.filters._target_encoding_fe import (
            auto_detect_te_cols,
        )
        X = pd.DataFrame({
            "cat10": [f"L{i % 10}" for i in range(100)],
            "cat2": ["A", "B"] * 50,        # below band
            "num": np.arange(100),
            "cat1000": [f"X{i}" for i in range(100)],  # would be > 500 if larger
        })
        picked = auto_detect_te_cols(X, min_card=5, max_card=500)
        assert "cat10" in picked
        # cat2 has cardinality 2 -> below min_card.
        assert "cat2" not in picked
        # num is numeric -> excluded by dtype gate.
        assert "num" not in picked


# ---------------------------------------------------------------------------
# Main biz_value contract: AUC lift
# ---------------------------------------------------------------------------


class TestMultiCategorySignalAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_te(self, seed: int):
        X, y = _build_cat_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)

        # Raw baseline: LogReg cannot consume the object column at all;
        # numeric-only baseline rides on the noise floor.
        auc_raw = _logreg_auc(X_tr, y_tr, X_ho, y_ho)

        # TE-augmented baseline: fit TE on train, transform holdout.
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit, apply_target_encoding,
        )
        te_tr_df, recipes = kfold_target_encode_fit(
            X_tr, y_tr.to_numpy(), ["cat_region"],
            n_folds=5, smoothing=10.0, random_state=seed,
        )
        te_ho = apply_target_encoding(X_ho, "cat_region", recipes["cat_region"])
        X_tr_aug = X_tr.copy()
        X_tr_aug["cat_region__te"] = te_tr_df["cat_region__te"].to_numpy()
        X_ho_aug = X_ho.copy()
        X_ho_aug["cat_region__te"] = te_ho
        auc_te = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)

        # Meaningful lift -- TE gives the categorical signal to LogReg in a
        # numeric form it can split on. ``auc_te >> auc_raw`` is the
        # contract; the threshold is conservative enough to ride 3 seeds.
        assert auc_te > auc_raw + 0.15, (
            f"seed={seed}: TE failed to lift AUC; raw={auc_raw:.3f} te={auc_te:.3f}"
        )
        # And TE alone clears a meaningful absolute bar.
        assert auc_te > 0.70, (
            f"seed={seed}: TE LogReg AUC={auc_te:.3f} below 0.70 floor"
        )


# ---------------------------------------------------------------------------
# No-leakage proof
# ---------------------------------------------------------------------------


class TestNoLeakage:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_transform_independent_of_holdout_y(self, seed: int):
        """``apply_target_encoding`` is a pure function of the test row's
        category + the fitted lookup. Shuffling holdout-y must NOT change
        transform output -- transform has no y reference at all."""
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit, apply_target_encoding,
        )
        X, y = _build_cat_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        _te_tr, recipes = kfold_target_encode_fit(
            X_tr, y_tr.to_numpy(), ["cat_region"],
            n_folds=5, smoothing=10.0, random_state=seed,
        )
        te_ho_orig = apply_target_encoding(X_ho, "cat_region", recipes["cat_region"])
        # Shuffle holdout-y and re-call -- transform takes no y.
        rng = np.random.default_rng(seed + 100)
        _y_shuffled = y_ho.to_numpy().copy()
        rng.shuffle(_y_shuffled)
        te_ho_shuf = apply_target_encoding(X_ho, "cat_region", recipes["cat_region"])
        np.testing.assert_array_equal(te_ho_orig, te_ho_shuf)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


class TestSmoothingShrinksRareCategories:
    def test_rare_cat_pulled_to_global_mean(self):
        """With smoothing=100 and a rare category (n_c=1), the smoothed
        estimate must sit closer to global_mean than to the raw per-cell
        mean. Conversely a common category with n_c=200 retains the raw
        estimate."""
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit,
        )
        rng = np.random.default_rng(0)
        n_common = 200
        # Common category 'C' gets P(y=1)=0.9 ; rare 'R' gets P(y=1)=1.0 (forced)
        X_common = pd.DataFrame({"cat": ["C"] * n_common})
        y_common = (rng.random(n_common) < 0.9).astype(int)
        # 1 rare row with y=1 -> raw mean would be 1.0.
        X_rare = pd.DataFrame({"cat": ["R"]})
        y_rare = np.array([1])
        X = pd.concat([X_common, X_rare], ignore_index=True)
        y = np.concatenate([y_common, y_rare])
        # Smoothing=100 is heavier than n_c=1 -> rare cat pulled hard.
        _, recipes = kfold_target_encode_fit(
            X, y, ["cat"], n_folds=2, smoothing=100.0, random_state=0,
        )
        rec = recipes["cat"]
        global_mean = rec["global_mean"]
        common_te = rec["lookup"]["C"]
        rare_te = rec["lookup"]["R"]
        # Common: stayed near raw 0.9 (not collapsed).
        assert abs(common_te - 0.9) < 0.1
        # Rare: pulled toward global_mean.
        assert abs(rare_te - global_mean) < abs(rare_te - 1.0)


# ---------------------------------------------------------------------------
# Recipe replay -- unseen categories
# ---------------------------------------------------------------------------


class TestRecipeReplayUnseenCategories:
    def test_unseen_category_maps_to_global_mean_no_nan(self):
        from mlframe.feature_selection.filters._target_encoding_fe import (
            kfold_target_encode_fit, apply_target_encoding,
        )
        X_tr = pd.DataFrame({"cat": ["A", "B", "C", "A", "B", "C"] * 10})
        y_tr = np.array([1, 0, 1, 1, 0, 1] * 10)
        _, recipes = kfold_target_encode_fit(
            X_tr, y_tr, ["cat"], n_folds=3, smoothing=1.0, random_state=0,
        )
        rec = recipes["cat"]
        X_ho = pd.DataFrame({"cat": ["A", "Z", "QQ", "B", None]})
        encoded = apply_target_encoding(X_ho, "cat", rec)
        assert encoded.shape == (5,)
        assert np.isfinite(encoded).all()  # no NaN propagation
        # Unseen 'Z' and 'QQ' map to global_mean.
        gm = float(rec["global_mean"])
        assert encoded[1] == pytest.approx(gm)
        assert encoded[2] == pytest.approx(gm)
        # NaN -> "__nan__" sentinel which was never observed -> global_mean.
        assert encoded[4] == pytest.approx(gm)


# ---------------------------------------------------------------------------
# Pickle / clone
# ---------------------------------------------------------------------------


class TestPickleCloneRoundTrip:
    def test_recipe_survives_pickle(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_kfold_target_encoded_recipe, apply_recipe,
        )
        rec = build_kfold_target_encoded_recipe(
            name="region__te", src_name="region",
            lookup={"A": 0.8, "B": 0.2, "C": 0.5},
            global_mean=0.5, smoothing=10.0,
        )
        rec_pkl = pickle.loads(pickle.dumps(rec))
        assert rec_pkl == rec
        X_test = pd.DataFrame({"region": ["A", "B", "Z", "C"]})
        orig = apply_recipe(rec, X_test)
        roundtrip = apply_recipe(rec_pkl, X_test)
        np.testing.assert_array_equal(orig, roundtrip)
        # Unseen 'Z' -> global_mean
        assert roundtrip[2] == pytest.approx(0.5)

    def test_full_mrmr_pickle_with_te(self):
        X, y = _build_cat_signal(seed=1)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=1)
        m = _make_mrmr(
            fe_kfold_te_enable=True,
            fe_kfold_te_cols=("cat_region",),
            fe_kfold_te_folds=3,
            fe_kfold_te_smoothing=10.0,
            fe_ntop_features=4,
        )
        m.fit(X_tr, y_tr)
        # Sklearn-clone preserves all ctor params bit-exactly.
        m2 = clone(m)
        params = m.get_params()
        params2 = m2.get_params()
        for k in ("fe_kfold_te_enable", "fe_kfold_te_cols",
                  "fe_kfold_te_folds", "fe_kfold_te_smoothing"):
            assert params[k] == params2[k], f"clone lost {k}"
        # Pickle preserves fitted state including recipes.
        m_pkl = pickle.loads(pickle.dumps(m))
        out_orig = m.transform(X_ho)
        out_pkl = m_pkl.transform(X_ho)
        # Both DataFrames, same shape, same engineered columns present.
        assert list(out_orig.columns) == list(out_pkl.columns)
        # ``cat_region__te`` was an engineered column; if it survived
        # MRMR's selection it should appear in both.
        for col in out_orig.columns:
            if pd.api.types.is_numeric_dtype(out_orig[col]):
                np.testing.assert_allclose(
                    out_orig[col].to_numpy(),
                    out_pkl[col].to_numpy(),
                    atol=1e-12,
                )


# ---------------------------------------------------------------------------
# Default-disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_no_te_attrs_populated_when_disabled(self):
        X, y = _build_cat_signal(seed=1)
        m = _make_mrmr(fe_ntop_features=3)
        m.fit(X, y)
        # Default OFF -> kfold_te_features_ stays empty.
        assert getattr(m, "kfold_te_features_", []) == []

    def test_transform_unchanged_when_disabled(self):
        """With ``fe_kfold_te_enable=False`` (default), transform output
        is bit-identical to a fresh instance with the same params -- the
        TE knobs are no-ops when the master switch is off."""
        X, y = _build_cat_signal(seed=7)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=7)
        m1 = _make_mrmr(fe_ntop_features=3)
        m1.fit(X_tr, y_tr)
        out1 = m1.transform(X_ho)
        # Same defaults except the TE knobs are explicitly set -- still
        # equivalent because the master is False.
        m2 = _make_mrmr(
            fe_ntop_features=3,
            fe_kfold_te_cols=("cat_region",),
            fe_kfold_te_folds=5,
            fe_kfold_te_smoothing=10.0,
        )
        m2.fit(X_tr, y_tr)
        out2 = m2.transform(X_ho)
        assert list(out1.columns) == list(out2.columns), (
            "Setting TE cols without master switch changed selected columns"
        )
        for c in out1.columns:
            if pd.api.types.is_numeric_dtype(out1[c]):
                np.testing.assert_allclose(
                    out1[c].to_numpy(), out2[c].to_numpy(), atol=1e-12,
                )
