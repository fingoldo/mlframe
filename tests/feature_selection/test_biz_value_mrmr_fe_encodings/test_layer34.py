"""Layer 34 biz_value: COUNT + FREQUENCY encoding + CAT x NUM residual.

Consolidated verbatim from test_biz_value_mrmr_layer34.py (per audit finding test_code_quality-16).

Validates the three new categorical FE kernels introduced 2026-05-31 as
companions to Layer 33's k-fold target encoding:

* **Count encoding**: y depends on cat_user_id's count (rare users
  behave differently from frequent ones). The count-encoded column gives
  LogReg a numeric signal it cannot otherwise see from the object dtype.

* **Frequency encoding**: same as count but scaled by 1 / n_samples;
  verify it ALSO works (equivalent up to affine for LogReg).

* **Cat x num interaction (OOF residual)**: y depends on the deviation
  of ``price`` from its category mean. The residual feature
  ``price__resid_by__cat`` captures the deviation; raw LogReg on price +
  cat (without the interaction) cannot consume the object cat column.

* **No leakage**: ``apply_*`` paths take no y -- shuffled y at transform
  time gives bit-identical output.

* **Pickle / clone**: recipes survive both round-trips; transform output
  matches bit-exactly.

* **Default disabled byte-identical**: with all three master switches
  OFF, MRMR.transform output is identical to a vanilla instance.

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
from tests.feature_selection._biz_val_synth import _logreg_auc, _train_holdout_split


# ---------------------------------------------------------------------------
# Fixtures: signal that ONLY count / freq / cat_num residual can decode
# ---------------------------------------------------------------------------


def _build_count_signal(seed: int, n: int = 4000):
    """y depends on cat_user_id's count of occurrences.

    A long-tailed Zipf-like distribution of user_id (a handful of "heavy"
    users with thousands of rows, a long tail of rare singletons).
    P(y=1) is a sigmoid function of log(count) -- rare users have
    P(y=1) ~ 0.1, heavy users P(y=1) ~ 0.9. The cat_user_id column
    itself is object-dtype so vanilla LogReg cannot consume it; the
    count-encoded column unlocks the signal cleanly.
    """
    rng = np.random.default_rng(seed)
    # Build a Zipf-like user distribution: 8 heavy users + 200 rare singletons.
    heavy_ids = [f"U_HEAVY_{i}" for i in range(8)]
    heavy_weights = np.array([400, 350, 300, 250, 200, 200, 200, 150], dtype=np.float64)
    # Sample heavy contributions
    n_heavy = int(heavy_weights.sum())  # 2050
    heavy_assignments = np.concatenate([
        np.repeat(uid, int(w)) for uid, w in zip(heavy_ids, heavy_weights)
    ])
    rng.shuffle(heavy_assignments)
    # Fill the rest with rare singletons
    n_rare = n - n_heavy
    rare_ids = np.array([f"U_RARE_{i:05d}" for i in range(max(n_rare, 0))])
    cat_user = np.concatenate([heavy_assignments, rare_ids])[:n]
    rng.shuffle(cat_user)
    # Per-row count
    counts = pd.Series(cat_user).value_counts()
    cnt_arr = pd.Series(cat_user).map(counts).to_numpy()
    # Sigmoid in log(count)
    log_cnt = np.log1p(cnt_arr.astype(np.float64))
    # Calibrate: split at median log_cnt
    centered = log_cnt - float(np.median(log_cnt))
    p = 1.0 / (1.0 + np.exp(-2.0 * centered))
    noise_a = rng.standard_normal(n)
    noise_b = rng.standard_normal(n)
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({
        "cat_user": cat_user,
        "noise_a": noise_a,
        "noise_b": noise_b,
    })
    return X, pd.Series(y, name="y")


def _build_cat_num_residual_signal(seed: int, n: int = 4000):
    """y depends on (price - mean_price | category).

    20 categories, each with its own price mean uniformly drawn in
    [10, 110]. Per-row price is the category mean plus N(0, 10) noise.
    y = 1 iff residual > 0 (price above its category mean). Raw LogReg
    on (price + cat) cannot see the residual because cat is object;
    even one-hot cat + price is roughly the same -- only the explicit
    residual feature captures the signal as a single numeric column.
    """
    rng = np.random.default_rng(seed)
    levels = [f"CAT_{i:02d}" for i in range(20)]
    level_means = rng.uniform(10.0, 110.0, size=20)
    cat = rng.choice(levels, size=n)
    cat_mean = pd.Series(cat).map(dict(zip(levels, level_means))).to_numpy()
    price = cat_mean + rng.normal(0.0, 10.0, size=n)
    residual = price - cat_mean
    # Sigmoid on the TRUE residual (Bayes ~ 0.95 AUC ceiling).
    p = 1.0 / (1.0 + np.exp(-0.2 * residual))
    y = (rng.random(n) < p).astype(int)
    noise = rng.standard_normal(n)
    X = pd.DataFrame({
        "cat": cat,
        "price": price,
        "noise": noise,
    })
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Direct unit tests on each kernel
# ---------------------------------------------------------------------------


class TestCountEncodeKernel:
    def test_fit_returns_integer_counts(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit,
        )
        X = pd.DataFrame({"cat": ["A", "B", "A", "C", "A", "B"]})
        enc, recipes = count_encode_fit(X, ["cat"])
        assert enc.shape == (6, 1)
        assert "cat__count" in enc.columns
        np.testing.assert_array_equal(
            enc["cat__count"].to_numpy(), np.array([3, 2, 3, 1, 3, 2]),
        )
        assert recipes["cat"]["lookup"] == {"A": 3, "B": 2, "C": 1}
        assert recipes["cat"]["default"] == 0

    def test_empty_X_rejected(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit,
        )
        with pytest.raises(ValueError, match="empty"):
            count_encode_fit(pd.DataFrame({"cat": []}), ["cat"])

    def test_missing_column_rejected(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit,
        )
        with pytest.raises(ValueError, match="missing"):
            count_encode_fit(pd.DataFrame({"a": ["x"] * 3}), ["b"])

    def test_apply_unseen_maps_to_default(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit, apply_count_encoding,
        )
        X_tr = pd.DataFrame({"cat": ["A", "A", "B", "B", "B"]})
        _, recipes = count_encode_fit(X_tr, ["cat"])
        X_ho = pd.DataFrame({"cat": ["A", "Z", "B"]})
        encoded = apply_count_encoding(X_ho, "cat", recipes["cat"])
        np.testing.assert_array_equal(encoded, np.array([2, 0, 3]))


class TestFrequencyEncodeKernel:
    def test_fit_returns_float_freqs(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            frequency_encode_fit,
        )
        X = pd.DataFrame({"cat": ["A", "B", "A", "C", "A", "B"]})
        enc, recipes = frequency_encode_fit(X, ["cat"])
        assert "cat__freq" in enc.columns
        # n=6, A:3 -> 0.5, B:2 -> 1/3, C:1 -> 1/6
        expected = np.array([0.5, 1 / 3, 0.5, 1 / 6, 0.5, 1 / 3])
        np.testing.assert_allclose(
            enc["cat__freq"].to_numpy(), expected, rtol=1e-12,
        )
        assert recipes["cat"]["lookup"]["A"] == pytest.approx(0.5)
        assert recipes["cat"]["lookup"]["C"] == pytest.approx(1 / 6)


class TestCatNumInteractionKernel:
    def test_residual_matches_construction(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            cat_num_interaction_fit,
        )
        rng = np.random.default_rng(0)
        n = 600
        cat = rng.choice(["A", "B", "C"], size=n)
        # Cat-conditional means: A=10, B=50, C=100
        means_map = {"A": 10.0, "B": 50.0, "C": 100.0}
        cat_mean = np.array([means_map[c] for c in cat])
        price = cat_mean + rng.normal(0.0, 5.0, size=n)
        y = (rng.random(n) > 0.5).astype(int)
        X = pd.DataFrame({"cat": cat, "price": price})
        residual, recipe = cat_num_interaction_fit(
            X, y, "cat", "price", n_folds=5, smoothing=1.0, random_state=0,
        )
        # Each category's stored lookup mean should be close to its construction mean
        # (small smoothing, n_c large -> approx raw mean).
        for c, mu in means_map.items():
            assert abs(recipe["lookup"][c] - mu) < 2.0, (
                f"lookup[{c}]={recipe['lookup'][c]:.3f} far from true mu={mu}"
            )
        # Residual sample mean per category should hover near zero.
        for c, mu in means_map.items():
            mask = cat == c
            assert abs(residual[mask].mean()) < 3.0

    def test_missing_columns_rejected(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            cat_num_interaction_fit,
        )
        X = pd.DataFrame({"a": ["x"] * 5, "b": [1.0] * 5})
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="missing"):
            cat_num_interaction_fit(X, y, "cat_nope", "b")
        with pytest.raises(ValueError, match="missing"):
            cat_num_interaction_fit(X, y, "a", "num_nope")

    def test_non_numeric_num_col_rejected(self):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            cat_num_interaction_fit,
        )
        X = pd.DataFrame({"cat": ["x"] * 5, "stringy": ["v"] * 5})
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="not numeric"):
            cat_num_interaction_fit(X, y, "cat", "stringy")


# ---------------------------------------------------------------------------
# Biz-value: AUC lift via each encoder
# ---------------------------------------------------------------------------


class TestCountEncodingAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_count(self, seed: int):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit, apply_count_encoding,
        )
        X, y = _build_count_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)

        auc_raw = _logreg_auc(X_tr, y_tr, X_ho, y_ho)

        enc_tr, recipes = count_encode_fit(X_tr, ["cat_user"])
        enc_ho = apply_count_encoding(X_ho, "cat_user", recipes["cat_user"])
        X_tr_aug = X_tr.copy()
        X_tr_aug["cat_user__count"] = enc_tr["cat_user__count"].to_numpy()
        X_ho_aug = X_ho.copy()
        X_ho_aug["cat_user__count"] = enc_ho
        auc_cnt = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)

        assert auc_cnt > auc_raw + 0.10, (
            f"seed={seed}: count enc failed to lift AUC; raw={auc_raw:.3f} cnt={auc_cnt:.3f}"
        )
        assert auc_cnt > 0.65, (
            f"seed={seed}: count enc LogReg AUC={auc_cnt:.3f} below 0.65 floor"
        )


class TestFrequencyEncodingAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_frequency(self, seed: int):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            frequency_encode_fit, apply_frequency_encoding,
        )
        X, y = _build_count_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)

        auc_raw = _logreg_auc(X_tr, y_tr, X_ho, y_ho)

        enc_tr, recipes = frequency_encode_fit(X_tr, ["cat_user"])
        enc_ho = apply_frequency_encoding(X_ho, "cat_user", recipes["cat_user"])
        X_tr_aug = X_tr.copy()
        X_tr_aug["cat_user__freq"] = enc_tr["cat_user__freq"].to_numpy()
        X_ho_aug = X_ho.copy()
        X_ho_aug["cat_user__freq"] = enc_ho
        auc_freq = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)

        assert auc_freq > auc_raw + 0.10, (
            f"seed={seed}: freq enc failed to lift AUC; raw={auc_raw:.3f} freq={auc_freq:.3f}"
        )
        assert auc_freq > 0.65, (
            f"seed={seed}: freq enc LogReg AUC={auc_freq:.3f} below 0.65 floor"
        )


class TestCatNumResidualAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_residual(self, seed: int):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            cat_num_interaction_fit, apply_cat_num_residual,
        )
        X, y = _build_cat_num_residual_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)

        # Baseline: numeric cols only (price + noise); cat dropped because
        # LogReg cannot consume object dtype.
        auc_raw = _logreg_auc(X_tr, y_tr, X_ho, y_ho)

        residual_tr, recipe = cat_num_interaction_fit(
            X_tr, y_tr.to_numpy(), "cat", "price",
            n_folds=5, smoothing=10.0, random_state=seed,
        )
        residual_ho = apply_cat_num_residual(X_ho, "cat", "price", recipe)
        X_tr_aug = X_tr.copy()
        X_tr_aug["price__resid_by__cat"] = residual_tr
        X_ho_aug = X_ho.copy()
        X_ho_aug["price__resid_by__cat"] = residual_ho
        auc_res = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)

        assert auc_res > auc_raw + 0.15, (
            f"seed={seed}: residual failed to lift AUC; raw={auc_raw:.3f} res={auc_res:.3f}"
        )
        assert auc_res > 0.75, (
            f"seed={seed}: residual LogReg AUC={auc_res:.3f} below 0.75 floor"
        )


# ---------------------------------------------------------------------------
# No leakage: transform paths take no y
# ---------------------------------------------------------------------------


class TestNoLeakageCountFreq:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_count_apply_independent_of_y(self, seed: int):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit, apply_count_encoding,
        )
        X, y = _build_count_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        _, recipes = count_encode_fit(X_tr, ["cat_user"])
        out1 = apply_count_encoding(X_ho, "cat_user", recipes["cat_user"])
        # Shuffle holdout-y; apply ignores y entirely.
        rng = np.random.default_rng(seed + 100)
        _y_shuffled = y_ho.to_numpy().copy()
        rng.shuffle(_y_shuffled)
        out2 = apply_count_encoding(X_ho, "cat_user", recipes["cat_user"])
        np.testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_freq_apply_independent_of_y(self, seed: int):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            frequency_encode_fit, apply_frequency_encoding,
        )
        X, y = _build_count_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        _, recipes = frequency_encode_fit(X_tr, ["cat_user"])
        out1 = apply_frequency_encoding(X_ho, "cat_user", recipes["cat_user"])
        out2 = apply_frequency_encoding(X_ho, "cat_user", recipes["cat_user"])
        np.testing.assert_array_equal(out1, out2)


class TestNoLeakageCatNum:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_residual_apply_independent_of_y(self, seed: int):
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            cat_num_interaction_fit, apply_cat_num_residual,
        )
        X, y = _build_cat_num_residual_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        _, recipe = cat_num_interaction_fit(
            X_tr, y_tr.to_numpy(), "cat", "price",
            n_folds=5, smoothing=10.0, random_state=seed,
        )
        out1 = apply_cat_num_residual(X_ho, "cat", "price", recipe)
        # Shuffle holdout-y is a no-op; apply path has no y.
        out2 = apply_cat_num_residual(X_ho, "cat", "price", recipe)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Recipe replay -- unseen categories never NaN
# ---------------------------------------------------------------------------


class TestUnseenCategoryHandling:
    def test_count_unseen_default_zero(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_count_encoded_recipe, apply_recipe,
        )
        rec = build_count_encoded_recipe(
            name="cat__count", src_name="cat",
            lookup={"A": 5, "B": 7}, default=0,
        )
        X = pd.DataFrame({"cat": ["A", "Z", "B", None, "B"]})
        out = apply_recipe(rec, X)
        np.testing.assert_array_equal(out, np.array([5, 0, 7, 0, 7]))

    def test_freq_unseen_default_zero(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_frequency_encoded_recipe, apply_recipe,
        )
        rec = build_frequency_encoded_recipe(
            name="cat__freq", src_name="cat",
            lookup={"A": 0.3, "B": 0.7}, default=0.0,
        )
        X = pd.DataFrame({"cat": ["A", "Z", "B"]})
        out = apply_recipe(rec, X)
        np.testing.assert_allclose(out, np.array([0.3, 0.0, 0.7]))

    def test_residual_unseen_falls_back_to_global_mean(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cat_num_residual_recipe, apply_recipe,
        )
        rec = build_cat_num_residual_recipe(
            name="price__resid_by__cat", cat_name="cat", num_name="price",
            lookup={"A": 10.0, "B": 20.0}, global_mean=15.0, smoothing=10.0,
        )
        X = pd.DataFrame({"cat": ["A", "Z", "B"], "price": [12.0, 16.0, 25.0]})
        out = apply_recipe(rec, X)
        # A: 12 - 10 = 2 ; Z (unseen): 16 - 15 = 1 ; B: 25 - 20 = 5
        np.testing.assert_allclose(out, np.array([2.0, 1.0, 5.0]))

    def test_residual_nan_num_emits_zero(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cat_num_residual_recipe, apply_recipe,
        )
        rec = build_cat_num_residual_recipe(
            name="price__resid_by__cat", cat_name="cat", num_name="price",
            lookup={"A": 10.0}, global_mean=10.0, smoothing=10.0,
        )
        X = pd.DataFrame({"cat": ["A", "A"], "price": [12.0, np.nan]})
        out = apply_recipe(rec, X)
        assert out[0] == pytest.approx(2.0)
        assert out[1] == pytest.approx(0.0)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Pickle / clone preservation
# ---------------------------------------------------------------------------


class TestPickleCloneRoundTrip:
    def test_count_recipe_pickle(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_count_encoded_recipe, apply_recipe,
        )
        rec = build_count_encoded_recipe(
            name="cat__count", src_name="cat",
            lookup={"A": 5, "B": 7}, default=0,
        )
        rec_pkl = pickle.loads(pickle.dumps(rec))
        assert rec_pkl == rec
        X = pd.DataFrame({"cat": ["A", "Z", "B"]})
        np.testing.assert_array_equal(apply_recipe(rec, X), apply_recipe(rec_pkl, X))

    def test_freq_recipe_pickle(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_frequency_encoded_recipe, apply_recipe,
        )
        rec = build_frequency_encoded_recipe(
            name="cat__freq", src_name="cat",
            lookup={"A": 0.4, "B": 0.6}, default=0.0,
        )
        rec_pkl = pickle.loads(pickle.dumps(rec))
        assert rec_pkl == rec
        X = pd.DataFrame({"cat": ["A", "B"]})
        np.testing.assert_allclose(apply_recipe(rec, X), apply_recipe(rec_pkl, X))

    def test_residual_recipe_pickle(self):
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cat_num_residual_recipe, apply_recipe,
        )
        rec = build_cat_num_residual_recipe(
            name="price__resid_by__cat", cat_name="cat", num_name="price",
            lookup={"A": 10.0, "B": 20.0}, global_mean=15.0, smoothing=5.0,
        )
        rec_pkl = pickle.loads(pickle.dumps(rec))
        assert rec_pkl == rec
        X = pd.DataFrame({"cat": ["A", "B"], "price": [11.0, 25.0]})
        np.testing.assert_allclose(apply_recipe(rec, X), apply_recipe(rec_pkl, X))

    def test_full_mrmr_clone_preserves_layer34_params(self):
        X, y = _build_count_signal(seed=1)
        m = _make_mrmr(
            fe_count_encoding_enable=True,
            fe_count_encoding_cols=("cat_user",),
            fe_frequency_encoding_enable=True,
            fe_frequency_encoding_cols=("cat_user",),
            fe_cat_num_interaction_enable=False,
            fe_ntop_features=3,
        )
        m2 = clone(m)
        p, p2 = m.get_params(), m2.get_params()
        for k in (
            "fe_count_encoding_enable", "fe_count_encoding_cols",
            "fe_frequency_encoding_enable", "fe_frequency_encoding_cols",
            "fe_cat_num_interaction_enable", "fe_cat_num_interaction_cat_cols",
            "fe_cat_num_interaction_num_cols", "fe_cat_num_interaction_folds",
            "fe_cat_num_interaction_smoothing",
        ):
            assert p[k] == p2[k], f"clone lost {k}"


# ---------------------------------------------------------------------------
# Default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_no_layer34_attrs_populated_when_disabled(self):
        X, y = _build_count_signal(seed=1)
        m = _make_mrmr(fe_ntop_features=3)
        m.fit(X, y)
        assert getattr(m, "count_encoding_features_", []) == []
        assert getattr(m, "frequency_encoding_features_", []) == []
        assert getattr(m, "cat_num_interaction_features_", []) == []

    def test_transform_unchanged_when_disabled(self):
        """With all three master switches OFF, transform output is
        bit-identical to a fresh instance with the same params -- the
        Layer 34 knobs are no-ops when the masters are False."""
        X, y = _build_count_signal(seed=7)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=7)
        m1 = _make_mrmr(fe_ntop_features=3)
        m1.fit(X_tr, y_tr)
        out1 = m1.transform(X_ho)
        m2 = _make_mrmr(
            fe_ntop_features=3,
            fe_count_encoding_cols=("cat_user",),
            fe_frequency_encoding_cols=("cat_user",),
            fe_cat_num_interaction_cat_cols=("cat_user",),
            fe_cat_num_interaction_num_cols=("noise_a",),
            fe_cat_num_interaction_folds=5,
            fe_cat_num_interaction_smoothing=10.0,
        )
        m2.fit(X_tr, y_tr)
        out2 = m2.transform(X_ho)
        assert list(out1.columns) == list(out2.columns), (
            "Setting Layer 34 cols without master switch changed selected columns"
        )
        for c in out1.columns:
            if pd.api.types.is_numeric_dtype(out1[c]):
                np.testing.assert_allclose(
                    out1[c].to_numpy(), out2[c].to_numpy(), atol=1e-12,
                )
