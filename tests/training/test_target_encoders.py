"""
Tests for the phase-N :class:`LeakageSafeEncoder`.

Round-3 architecture A17 + tests T14: target encoding inside a fit/
transform pipeline LEAKS the target unless OOF-fitted. The
hard-to-detect symptom is "train AUC near 1.0, val AUC drops out".
This pack pins the contract:

  * ``fit_transform`` returns OOF-computed encodings on train rows;
  * ``transform`` on held-out rows uses the full-train statistic;
  * the leakage probe (high-cardinality cat + random target):
    naive encoder -> train ~ memorised; OOF encoder -> random.

Coverage:
  * Each method (target_mean / target_m_estimate / target_james_stein
    / target_loo / woe) produces correct shape + finite values.
  * Smoothing pulls rare-category encodings toward the prior.
  * Unseen categories at transform time -> global prior, no crash.
  * None / NaN values map to a single sentinel category.
  * Reproducibility: same random_state -> identical encodings
    across runs (round-3 R3-09).
  * WoE rejects non-binary targets.
  * Leakage probe (the headline test).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling import LeakageSafeEncoder


# =====================================================================
# 1. Headline leakage probe (round-3 T14)
# =====================================================================


class TestLeakageProbe:
    """The probe: high-cardinality cat (each value unique) + random
    target. Naive encoding memorises the row-level noise; OOF
    encoding doesn't.

    With cardinality == n_rows, the naive encoded value for each row
    IS that row's target (each "category" has one observation).
    OOF leaves that row out, so its encoding is the prior + smoothing
    -- gets nothing useful out of the encoding.
    """

    def test_oof_breaks_naive_memorisation(self):
        """Oof breaks naive memorisation."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(0)
        n = 1000
        # Each row in its own category (perfect-cardinality).
        cats = np.array([f"cat_{i}" for i in range(n)])
        # Random binary target.
        y = rng.randint(0, 2, size=n).astype(float)

        # Naive (re-implement: per-cat mean, no OOF)
        naive = y.copy()  # since each cat has one row, mean = y[i]
        naive_auc = roc_auc_score(y, naive)
        # Memorisation -> near-perfect train AUC.
        assert naive_auc >= 0.99

        # OOF
        enc = LeakageSafeEncoder(
            method="target_mean",
            smoothing=10.0,
            cv=5,
            random_state=0,
        )
        oof = enc.fit_transform(cats, y)
        oof_auc = roc_auc_score(y, oof)
        # OOF on random target should be near random (around 0.5).
        # Don't pin a tight bound -- random fluctuation can make this
        # 0.4-0.6; the lock is "much less than naive".
        assert oof_auc < 0.7, f"OOF AUC {oof_auc:.3f} too high; leak may be present"

    def test_oof_train_then_transform_held_out(self):
        """After fit_transform on train, the fitted encoder should
        transform held-out rows using the full-train statistic.
        Held-out rows weren't in any fold's train set, so no leak
        risk from the held-out side."""
        rng = np.random.RandomState(0)
        train_cats = np.array(["A", "B", "C"]).repeat(100)
        rng.shuffle(train_cats)
        train_y = (train_cats == "A").astype(float)

        enc = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=5)
        enc.fit_transform(train_cats, train_y)

        # Held-out rows (use new categories + seen ones).
        held_cats = np.array(["A", "B", "C", "Z_unseen"])
        out = enc.transform(held_cats)
        assert out.shape == (4,)
        # "Z_unseen" -> global prior.
        prior = train_y.mean()
        np.testing.assert_allclose(out[3], prior, atol=1e-3)


# =====================================================================
# 2. Method shape coverage
# =====================================================================


@pytest.fixture
def synthetic_train():
    """Synthetic train."""
    rng = np.random.RandomState(0)
    n = 200
    cats = rng.choice(["A", "B", "C", "D", "E"], size=n)
    y = rng.randint(0, 2, size=n).astype(float)
    return cats, y


class TestMethodShape:
    """Groups tests covering method shape."""
    @pytest.mark.parametrize(
        "method",
        [
            "target_mean",
            "target_m_estimate",
            "target_james_stein",
            "target_loo",
            "woe",
        ],
    )
    def test_fit_transform_returns_correct_shape(self, method, synthetic_train):
        """Fit transform returns correct shape."""
        cats, y = synthetic_train
        enc = LeakageSafeEncoder(method=method, smoothing=5.0, cv=5, random_state=0)
        out = enc.fit_transform(cats, y)
        assert out.shape == (len(cats),)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize(
        "method",
        [
            "target_mean",
            "target_m_estimate",
            "target_james_stein",
            "target_loo",
        ],
    )
    def test_transform_returns_correct_shape(self, method, synthetic_train):
        """Transform returns correct shape."""
        cats, y = synthetic_train
        enc = LeakageSafeEncoder(method=method, smoothing=5.0, cv=5, random_state=0)
        enc.fit(cats, y)
        out = enc.transform(["A", "B", "C", "X_unseen"])
        assert out.shape == (4,)
        assert np.isfinite(out).all()


# =====================================================================
# 3. Smoothing semantics
# =====================================================================


class TestSmoothing:
    """Groups tests covering smoothing."""
    def test_high_smoothing_pulls_toward_prior(self):
        """High smoothing pulls toward prior."""
        cats = np.array(["A"] * 5 + ["B"] * 100)
        y = np.array([1.0] * 5 + [0.0] * 100)  # A has 100% pos, B has 0%
        prior = y.mean()  # 5/105 ~ 0.048

        # Low smoothing: rare cat A keeps its 1.0.
        enc_low = LeakageSafeEncoder(method="target_mean", smoothing=0.1, cv=2)
        enc_low.fit(cats, y)
        a_low = enc_low.transform(["A"])[0]
        assert a_low > 0.9, f"low-smoothing A encoding should be near 1.0, got {a_low}"

        # High smoothing: rare cat A shrinks toward prior.
        enc_high = LeakageSafeEncoder(method="target_mean", smoothing=100.0, cv=2)
        enc_high.fit(cats, y)
        a_high = enc_high.transform(["A"])[0]
        assert a_high < 0.5, f"high-smoothing A should pull toward prior {prior:.3f}, got {a_high}"


# =====================================================================
# 4. Unseen categories
# =====================================================================


class TestUnseenCategories:
    """Groups tests covering unseen categories."""
    def test_unseen_at_transform_returns_prior(self, synthetic_train):
        """Unseen at transform returns prior."""
        cats, y = synthetic_train
        enc = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=3)
        enc.fit(cats, y)
        out = enc.transform(["never_seen", "also_never_seen"])
        prior = y.mean()
        np.testing.assert_allclose(out, [prior, prior], atol=1e-9)

    def test_none_and_nan_share_a_category(self, synthetic_train):
        """None and nan share a category."""
        cats, y = synthetic_train
        # Augment with None / NaN
        cats_with_nulls = np.concatenate([cats, np.array([None, float("nan"), None], dtype=object)])
        y_with_nulls = np.concatenate([y, np.array([1.0, 0.0, 1.0])])
        enc = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=3)
        enc.fit_transform(cats_with_nulls, y_with_nulls)
        # None and NaN both map to "__NULL__" sentinel; encoder should
        # treat them as one category.
        out = enc.transform([None, float("nan")])
        np.testing.assert_allclose(out[0], out[1], atol=1e-9)


# =====================================================================
# 5. Reproducibility
# =====================================================================


class TestReproducibility:
    """Groups tests covering reproducibility."""
    def test_same_random_state_yields_identical_encodings(self, synthetic_train):
        """Same random state yields identical encodings."""
        cats, y = synthetic_train
        enc1 = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=5, random_state=42)
        enc2 = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=5, random_state=42)
        out1 = enc1.fit_transform(cats, y)
        out2 = enc2.fit_transform(cats, y)
        np.testing.assert_array_equal(out1, out2)

    def test_different_random_state_different_encodings(self, synthetic_train):
        """Different random state different encodings."""
        cats, y = synthetic_train
        enc1 = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=5, random_state=0)
        enc2 = LeakageSafeEncoder(method="target_mean", smoothing=5.0, cv=5, random_state=999)
        out1 = enc1.fit_transform(cats, y)
        out2 = enc2.fit_transform(cats, y)
        # Different folds -> different OOF encodings (very unlikely to coincide).
        assert not np.array_equal(out1, out2)


# =====================================================================
# 6. WoE binary-target requirement
# =====================================================================


class TestWoE:
    """Groups tests covering wo e."""
    def test_woe_rejects_non_binary(self):
        """Woe rejects non binary."""
        cats = np.array(["A", "B"] * 50)
        y_continuous = np.linspace(0, 10, 100)
        enc = LeakageSafeEncoder(method="woe", smoothing=1.0, cv=2)
        with pytest.raises(ValueError, match="binary"):
            enc.fit(cats, y_continuous)

    def test_woe_binary_works(self):
        """Woe binary works."""
        rng = np.random.RandomState(0)
        cats = np.array(["good", "bad"]).repeat(50)
        y = np.concatenate([np.ones(40), np.zeros(10), np.ones(5), np.zeros(45)])
        rng.shuffle(y)  # random shuffle to avoid pure correlation
        enc = LeakageSafeEncoder(method="woe", smoothing=1.0, cv=2)
        enc.fit(cats, y)
        out = enc.transform(["good", "bad"])
        assert out.shape == (2,)
        assert np.isfinite(out).all()


# =====================================================================
# 7. Validation
# =====================================================================


class TestValidation:
    """Groups tests covering validation."""
    def test_invalid_method_raises(self):
        """Invalid method raises."""
        with pytest.raises(ValueError, match="unknown method"):
            LeakageSafeEncoder(method="not_a_method")

    def test_cv_below_2_raises(self):
        """Cv below 2 raises."""
        with pytest.raises(ValueError, match="cv must be >= 2"):
            LeakageSafeEncoder(method="target_mean", cv=1)

    def test_negative_smoothing_raises(self):
        """Negative smoothing raises."""
        with pytest.raises(ValueError, match="smoothing must be >= 0"):
            LeakageSafeEncoder(method="target_mean", smoothing=-1.0)

    def test_x_y_length_mismatch_raises(self):
        """X y length mismatch raises."""
        cats = np.array(["A", "B", "C"])
        y = np.array([0.0, 1.0])  # one too few
        enc = LeakageSafeEncoder(method="target_mean", cv=2)
        with pytest.raises(ValueError, match="length mismatch"):
            enc.fit(cats, y)

    def test_transform_before_fit_raises(self):
        """Transform before fit raises."""
        enc = LeakageSafeEncoder(method="target_mean", cv=2)
        with pytest.raises(RuntimeError, match="before fit"):
            enc.transform(["A", "B"])
