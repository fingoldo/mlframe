"""
Tests for feature_selection/filters/fe_baselines.py.

The module supplies honest non-polynomial baselines (trivial pair / unary / triplet transforms + their
MI scoring) that any engineered feature must beat before being claimed as useful. Tests cover the
public API:

* ``trivial_pair_features`` / ``triplet_pair_features`` -- shape/finiteness/known-value sanity
* ``score_trivial_baselines`` / ``score_triplet_baselines`` -- descending-MI ordering, top pick on
  a multiplicative signal
* ``auto_unary_transforms`` -- identity always included, uplift gating works
* ``best_unary_transform`` / ``best_trivial_pair`` -- top-marginal style picks the highest-MI
  single feature
* Random baseline returns near-zero MI on signal-rich data (biz_value floor check)
* Determinism with fixed seed
* Edge cases: single feature, all-constant column, empty input
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.fe_baselines import (
    auto_unary_transforms,
    best_trivial_pair,
    best_unary_transform,
    score_trivial_baselines,
    score_triplet_baselines,
    triplet_pair_features,
    trivial_pair_features,
)
from mlframe.feature_selection.filters import (
    compute_mi_from_classes,
    discretize_array,
)

# ================================================================================================
# trivial_pair_features
# ================================================================================================


class TestTrivialPairFeatures:
    """Groups tests covering TestTrivialPairFeatures."""
    def test_keys_and_shapes(self):
        """All trivial pair transforms produce 1-D arrays of the same length as inputs."""
        rng = np.random.default_rng(0)
        n = 200
        x_a = rng.standard_normal(n)
        x_b = rng.standard_normal(n)

        feats = trivial_pair_features(x_a, x_b)

        expected_keys = {
            "mul",
            "add",
            "sub",
            "ratio_ab",
            "ratio_ba",
            "sq_dist",
            "sum_sq",
            "maxab",
            "minab",
            "log_abs_mul",
            "atan2",
            "geo_mean",
        }
        assert expected_keys.issubset(set(feats.keys()))
        for name, arr in feats.items():
            assert arr.shape == (n,), f"{name} wrong shape"

    def test_known_values_mul_add_sub(self):
        """mul/add/sub return the exact arithmetic outputs."""
        x_a = np.array([1.0, 2.0, 3.0])
        x_b = np.array([4.0, 5.0, 6.0])
        feats = trivial_pair_features(x_a, x_b)
        np.testing.assert_allclose(feats["mul"], [4.0, 10.0, 18.0])
        np.testing.assert_allclose(feats["add"], [5.0, 7.0, 9.0])
        np.testing.assert_allclose(feats["sub"], [-3.0, -3.0, -3.0])
        np.testing.assert_allclose(feats["maxab"], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(feats["minab"], [1.0, 2.0, 3.0])

    def test_ratio_safe_with_zero_denominator(self):
        """ratio_ab handles x_b == 0 without producing inf/nan due to the eps guard."""
        x_a = np.array([1.0, 2.0, -3.0])
        x_b = np.array([0.0, 0.0, 0.0])
        feats = trivial_pair_features(x_a, x_b)
        assert np.all(np.isfinite(feats["ratio_ab"]))
        assert np.all(np.isfinite(feats["ratio_ba"]))

    def test_determinism_fixed_seed(self):
        """Output is deterministic for the same input."""
        rng = np.random.default_rng(123)
        x_a = rng.standard_normal(100)
        x_b = rng.standard_normal(100)
        f1 = trivial_pair_features(x_a, x_b)
        f2 = trivial_pair_features(x_a.copy(), x_b.copy())
        for k in f1:
            np.testing.assert_array_equal(f1[k], f2[k])


# ================================================================================================
# score_trivial_baselines
# ================================================================================================


class TestScoreTrivialBaselines:
    """Groups tests covering TestScoreTrivialBaselines."""
    def test_descending_order(self):
        """Returned dict is iteration-ordered by descending MI."""
        rng = np.random.default_rng(7)
        n = 500
        x_a = rng.standard_normal(n)
        x_b = rng.standard_normal(n)
        # Discrete target loosely tied to product.
        y = (x_a * x_b > 0).astype(np.int64)
        scores = score_trivial_baselines(x_a, x_b, y, discrete_target=True)
        assert len(scores) > 0
        vals = list(scores.values())
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_multiplicative_signal_picks_multiplicative_feature(self):
        """When y = sign(x_a * x_b), a multiplicative-style trivial feature wins
        (mul / log_abs_mul / geo_mean are all faithful representations)."""
        rng = np.random.default_rng(42)
        n = 2000
        x_a = rng.standard_normal(n)
        x_b = rng.standard_normal(n)
        y = (x_a * x_b > 0).astype(np.int64)

        scores = score_trivial_baselines(x_a, x_b, y, discrete_target=True)
        winner = next(iter(scores))
        # The non-linear, non-multiplicative pair features (sub/add/maxab/minab/sq_dist/sum_sq)
        # carry no info on sign(x_a * x_b); the winner must be one of the multiplicative family.
        assert winner in {"mul", "log_abs_mul", "geo_mean", "ratio_ab", "ratio_ba", "atan2"}


# ================================================================================================
# best_trivial_pair (top-marginal-style pick)
# ================================================================================================


class TestBestTrivialPair:
    """Groups tests covering TestBestTrivialPair."""
    def test_returns_tuple_with_max_mi(self):
        """best_trivial_pair returns (name, arr, mi) with mi == max over all candidates."""
        rng = np.random.default_rng(0)
        n = 1000
        x_a = rng.standard_normal(n)
        x_b = rng.standard_normal(n)
        y = (x_a * x_b > 0).astype(np.int64)

        scores = score_trivial_baselines(x_a, x_b, y, discrete_target=True)
        best = best_trivial_pair(x_a, x_b, y, discrete_target=True)

        assert best is not None
        _name, arr, mi = best
        assert arr.shape == (n,)
        # mi must match the top of the ranked dict.
        np.testing.assert_allclose(mi, max(scores.values()), atol=1e-12)


# ================================================================================================
# auto_unary_transforms + best_unary_transform
# ================================================================================================


class TestAutoUnaryTransforms:
    """Groups tests covering TestAutoUnaryTransforms."""
    def test_identity_always_present(self):
        """The identity transform is always retained, regardless of uplift threshold."""
        rng = np.random.default_rng(1)
        n = 500
        x = rng.standard_normal(n)
        y = (x > 0).astype(np.int64)
        out = auto_unary_transforms(x, y, discrete_target=True, min_uplift=1e9)
        assert "identity" in out
        ident_arr, _ident_mi = out["identity"]
        np.testing.assert_allclose(ident_arr, x.astype(np.float64))

    def test_regression_identity_mi_not_recomputed(self, monkeypatch):
        """Wave 13 finding 6: the identity transform is the literal same input as ``base``, already
        scored once before the transform loop; it must not pay a second ``_mi_1d`` call. Equivalence:
        the reused value must equal a fresh direct computation on the same (x, y)."""
        from mlframe.feature_selection.filters import fe_baselines as fb

        orig_mi_1d = fb._mi_1d
        calls = {"n": 0}

        def counted(*a, **kw):
            """Helper that counted."""
            calls["n"] += 1
            return orig_mi_1d(*a, **kw)

        monkeypatch.setattr(fb, "_mi_1d", counted)
        rng = np.random.default_rng(11)
        n = 500
        x = rng.standard_normal(n)
        y = (x > 0).astype(np.int64)
        out = fb.auto_unary_transforms(x, y, discrete_target=True, min_uplift=0.0)
        n_transforms = 7  # identity, log_abs, sqrt_abs_signed, inv, square, cube, tanh
        # Pre-fix: 1 (base) + n_transforms (one per transform incl. identity) = 8.
        # Post-fix: 1 (base) + (n_transforms - 1) (identity skipped) = 7.
        assert calls["n"] == n_transforms, f"expected {n_transforms} _mi_1d calls (base once + 6 non-identity transforms), got {calls['n']}"

        _, ident_mi = out["identity"]
        direct_base = orig_mi_1d(x, y, discrete_target=True, mi_estimator="plugin", plugin_n_bins=20)
        assert ident_mi == pytest.approx(direct_base, abs=0.0), "identity MI must equal the base MI it was reused from, bit-for-bit"

    def test_min_uplift_gating(self):
        """Non-identity transforms appear only if mi >= base * min_uplift."""
        rng = np.random.default_rng(2)
        n = 500
        x = rng.standard_normal(n)
        y = (x > 0).astype(np.int64)
        loose = auto_unary_transforms(x, y, discrete_target=True, min_uplift=0.0)
        strict = auto_unary_transforms(x, y, discrete_target=True, min_uplift=1e6)
        # Loose threshold admits all finite candidates; strict admits only identity.
        assert len(loose) >= len(strict)
        assert set(strict.keys()) == {"identity"}


class TestBestUnaryTransform:
    """Groups tests covering TestBestUnaryTransform."""
    def test_picks_max_mi_unary(self):
        """best_unary_transform returns the highest-MI unary candidate (top-marginal pick)."""
        rng = np.random.default_rng(3)
        n = 1000
        # y = sign(x) -> identity / cube / tanh are all monotonic in sign(x), so any of those is fine.
        x = rng.standard_normal(n)
        y = (x > 0).astype(np.int64)
        _name, arr, mi = best_unary_transform(x, y, discrete_target=True)
        assert arr.shape == (n,)
        assert mi >= 0.0
        # Brute-force max from auto_unary_transforms with no gating.
        cands = auto_unary_transforms(x, y, discrete_target=True, min_uplift=0.0)
        top_mi = max(v[1] for v in cands.values())
        np.testing.assert_allclose(mi, top_mi, atol=1e-12)


# ================================================================================================
# triplet pair features + ranking
# ================================================================================================


class TestTriplets:
    """Groups tests covering TestTriplets."""
    def test_triplet_features_shape(self):
        """Triplet features shape."""
        rng = np.random.default_rng(4)
        n = 200
        a, b, c = (rng.standard_normal(n) for _ in range(3))
        feats = triplet_pair_features(a, b, c)
        for name, arr in feats.items():
            assert arr.shape == (n,), f"{name} bad shape"
            assert np.all(np.isfinite(arr)), f"{name} has non-finite values"

    def test_score_triplet_baselines_descending(self):
        """Score triplet baselines descending."""
        rng = np.random.default_rng(5)
        n = 800
        a, b, c = (rng.standard_normal(n) for _ in range(3))
        y = (a * b * c > 0).astype(np.int64)
        scores = score_triplet_baselines(a, b, c, y, discrete_target=True)
        vals = list(scores.values())
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        # 3-way sign product is captured by any feature that uses all three
        # inputs nonlinearly: ``abc_mul`` (a*b*c), ``geo_mean3``,
        # ``atan2_ab_c`` (atan2 of ab vs c), ``ab_div_c`` (ratio whose sign
        # tracks the same product). The top scorer flips between these
        # across seeds; allow any of the four-way set.
        assert next(iter(scores)) in {"abc_mul", "geo_mean3", "atan2_ab_c", "ab_div_c"}


# ================================================================================================
# Edge cases
# ================================================================================================


class TestEdgeCases:
    """Groups tests covering TestEdgeCases."""
    def test_constant_column_pair_features(self):
        """Constant inputs produce finite outputs (no inf/nan blow-ups)."""
        x_a = np.zeros(50)
        x_b = np.ones(50)
        feats = trivial_pair_features(x_a, x_b)
        for name, arr in feats.items():
            assert np.all(np.isfinite(arr)), f"{name} not finite on constant input"

    def test_constant_column_scoring_yields_near_zero_mi(self):
        """A constant pair feature carries (near-)zero MI with any target. The plug-in quantile
        binner partitions a constant column into arbitrary contiguous slices, so a tiny finite-sample
        bias (~ a few % of 1 nat) is unavoidable; we assert << 0.1."""
        rng = np.random.default_rng(6)
        n = 300
        x_a = np.zeros(n)
        x_b = np.zeros(n)
        y = rng.integers(0, 2, size=n).astype(np.int64)
        scores = score_trivial_baselines(x_a, x_b, y, discrete_target=True)
        for name, mi in scores.items():
            assert mi < 0.1, f"{name} MI={mi} too high on constant input"

    def test_empty_input_pair_features(self):
        """trivial_pair_features handles zero-length inputs without crashing."""
        x_a = np.array([], dtype=np.float64)
        x_b = np.array([], dtype=np.float64)
        feats = trivial_pair_features(x_a, x_b)
        for name, arr in feats.items():
            assert arr.shape == (0,), f"{name} non-empty on empty input"

    def test_single_sample_pair_features(self):
        """A single-sample input still produces one-element trivial features."""
        x_a = np.array([2.5])
        x_b = np.array([0.5])
        feats = trivial_pair_features(x_a, x_b)
        for arr in feats.values():
            assert arr.shape == (1,)
        np.testing.assert_allclose(feats["mul"], [1.25])

    def test_best_trivial_pair_all_nonfinite_returns_none(self):
        """If every trivial feature is non-finite, best_trivial_pair returns None.
        NaN inputs propagate through every transform (including atan2), so every candidate is filtered."""
        x_a = np.full(50, np.nan)
        x_b = np.full(50, np.nan)
        y = np.zeros(50, dtype=np.int64)
        result = best_trivial_pair(x_a, x_b, y, discrete_target=True)
        assert result is None


# ================================================================================================
# biz_value (fast): random baseline must lie far below the true-signal MI floor.
# ================================================================================================


@pytest.mark.fast
def test_biz_random_baseline_below_signal_floor():
    """Honest-baseline biz_value: an UNRELATED random feature must score MI <= 0.1 on data where
    the true signal carrier x_0 has MI ~ 0.5 with y.

    Constructs a binary classification problem with y = sign(x_0); x_0 then carries the full signal
    (MI(x_0, y) ~ log 2 nats ~ 0.69). A second independent draw x_rand must show near-zero MI.
    This is the property a baseline FE evaluator relies on: random features don't fake signal.
    """
    rng = np.random.default_rng(2026)
    n = 4000

    x_signal = rng.standard_normal(n)
    y = (x_signal > 0).astype(np.int64)
    x_random = rng.standard_normal(n)  # independent of y

    n_bins = 20
    classes_signal = discretize_array(x_signal, n_bins=n_bins).astype(np.int32)
    classes_random = discretize_array(x_random, n_bins=n_bins).astype(np.int32)
    classes_y = y.astype(np.int32)

    freqs_signal = np.bincount(classes_signal, minlength=n_bins).astype(np.float64) / n
    freqs_random = np.bincount(classes_random, minlength=n_bins).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64) / n

    mi_signal = compute_mi_from_classes(
        classes_x=classes_signal,
        freqs_x=freqs_signal,
        classes_y=classes_y,
        freqs_y=freqs_y,
    )
    mi_random = compute_mi_from_classes(
        classes_x=classes_random,
        freqs_x=freqs_random,
        classes_y=classes_y,
        freqs_y=freqs_y,
    )

    # True signal must clear 0.4 nats (theoretical ceiling ~ ln 2 = 0.693 for perfect binary recovery).
    assert mi_signal > 0.4, f"signal MI={mi_signal:.4f} below expected ~0.5"
    # Random baseline must lie an order of magnitude below the signal AND below the 0.1 floor.
    assert mi_random < 0.1, f"random baseline MI={mi_random:.4f} exceeds 0.1 floor"
    assert mi_random < 0.2 * mi_signal, f"random baseline MI={mi_random:.4f} not << signal MI={mi_signal:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
