"""Tests for ``linear_residual_multi`` transform (R10c extension #1).

Coverage:
- Round-trip ``y -> T -> y'`` on (n, K=2) and (n, K=3) base matrices
  (deterministic correctness; rtol=1e-7).
- Forward selection sanity: K=1 (degenerate) yields the same fit as
  the single-base ``linear_residual`` transform.
- Sample-weight propagation: weighted fit matches the unweighted fit
  when all weights are 1.
- Condition-number gate: near-collinear bases (cond > 30) trigger the
  collinear_fallback path with zero-alpha + train-mean intercept.
- Domain check: rejects rows where any base column is non-finite.
- Biz_value: on a synthetic 2-base regression DGP where the second
  base carries orthogonal structural signal, ``linear_residual_multi``
  produces a target (residual) that has STRICTLY lower variance and
  STRICTLY lower MI(T, base_1) + MI(T, base_2) than the single-base
  ``linear_residual`` on the dominant base alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _linear_residual_multi_fit,
    _linear_residual_multi_forward,
    _linear_residual_multi_inverse,
    _linear_residual_multi_domain,
    get_transform,
)

# ---------------------------------------------------------------------------
# Unit: round-trip / fit / forward / inverse
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Groups tests covering round trip."""
    def test_k1_matches_single_base_linear_residual(self) -> None:
        """K=1 multi-base must be numerically identical to single-base
        ``linear_residual``. This is the 'no regression' lock for
        legacy paths that opt into the new transform."""
        rng = np.random.default_rng(0)
        n = 200
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 0.95 * base + rng.normal(scale=0.3, size=n)
        single = get_transform("linear_residual")
        multi = get_transform("linear_residual_multi")
        p_single = single.fit(y, base)
        p_multi = multi.fit(y, base)
        # Coefficient match (alphas[0] == alpha, beta == beta).
        assert abs(p_single["alpha"] - p_multi["alphas"][0]) < 1e-9
        assert abs(p_single["beta"] - p_multi["beta"]) < 1e-9
        assert p_multi["collinear_fallback"] is False
        # Forward and inverse match exactly.
        T_single = single.forward(y, base, p_single)
        T_multi = multi.forward(y, base, p_multi)
        np.testing.assert_allclose(T_single, T_multi, rtol=1e-12, atol=1e-12)
        y_single = single.inverse(T_single, base, p_single)
        y_multi = multi.inverse(T_multi, base, p_multi)
        np.testing.assert_allclose(y_single, y_multi, rtol=1e-12, atol=1e-12)

    def test_k2_round_trip(self) -> None:
        """``y -> T -> y'`` on K=2 base must recover y exactly."""
        rng = np.random.default_rng(1)
        n = 300
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=5.0, scale=1.0, size=n)  # independent of b1
        y = 0.6 * b1 + 0.3 * b2 + rng.normal(scale=0.2, size=n)
        base = np.column_stack([b1, b2])
        t = get_transform("linear_residual_multi")
        p = t.fit(y, base)
        T = t.forward(y, base, p)
        y_back = t.inverse(T, base, p)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_k3_round_trip(self) -> None:
        """K3 round trip."""
        rng = np.random.default_rng(2)
        n = 500
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=0.0, scale=3.0, size=n)
        b3 = rng.uniform(low=-1, high=1, size=n)
        y = 0.5 * b1 + 0.2 * b2 - 0.4 * b3 + rng.normal(scale=0.1, size=n)
        base = np.column_stack([b1, b2, b3])
        t = get_transform("linear_residual_multi")
        p = t.fit(y, base)
        T = t.forward(y, base, p)
        y_back = t.inverse(T, base, p)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_1d_base_promoted_to_2d_internally(self) -> None:
        """Passing 1-D base to multi-fit must work (K=1 degenerate case)."""
        rng = np.random.default_rng(3)
        n = 100
        base = rng.normal(size=n)
        y = 2.0 * base + 1.5 + rng.normal(scale=0.1, size=n)
        p = _linear_residual_multi_fit(y, base)
        assert len(p["alphas"]) == 1
        # Forward / inverse with 1-D base also OK.
        T = _linear_residual_multi_forward(y, base, p)
        y_back = _linear_residual_multi_inverse(T, base, p)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)


class TestFitContract:
    """Groups tests covering fit contract."""
    def test_recovers_true_coefficients_on_clean_data(self) -> None:
        """When the DGP is exactly the linear combination + small noise,
        fitted alphas should be close to the true ones."""
        rng = np.random.default_rng(42)
        n = 5000  # large n -> tight estimate
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=0.0, scale=1.0, size=n)
        true_alphas = (0.85, -0.42)
        true_beta = 3.14
        y = true_alphas[0] * b1 + true_alphas[1] * b2 + true_beta + rng.normal(scale=0.05, size=n)
        base = np.column_stack([b1, b2])
        p = _linear_residual_multi_fit(y, base)
        assert abs(p["alphas"][0] - true_alphas[0]) < 0.01
        assert abs(p["alphas"][1] - true_alphas[1]) < 0.01
        assert abs(p["beta"] - true_beta) < 0.05
        assert p["collinear_fallback"] is False

    def test_sample_weight_uniform_matches_unweighted(self) -> None:
        """Sample weight uniform matches unweighted."""
        rng = np.random.default_rng(7)
        n = 300
        b1 = rng.normal(size=n)
        b2 = rng.normal(size=n)
        y = 0.5 * b1 + 0.3 * b2 + rng.normal(scale=0.1, size=n)
        base = np.column_stack([b1, b2])
        p_uw = _linear_residual_multi_fit(y, base, sample_weight=None)
        p_w = _linear_residual_multi_fit(y, base, sample_weight=np.ones(n))
        np.testing.assert_allclose(p_uw["alphas"], p_w["alphas"], rtol=1e-9)
        assert abs(p_uw["beta"] - p_w["beta"]) < 1e-9

    def test_sample_weight_zero_falls_back_to_mean(self) -> None:
        """All-zero weights => insufficient info => fallback to train
        mean intercept + zero alphas."""
        rng = np.random.default_rng(8)
        n = 100
        base = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        y = rng.normal(loc=5.0, scale=1.0, size=n)
        p = _linear_residual_multi_fit(y, base, sample_weight=np.zeros(n))
        assert p["alphas"] == [0.0, 0.0]
        assert abs(p["beta"] - float(np.mean(y))) < 1e-9
        assert p["collinear_fallback"] is True

    def test_collinear_fallback_when_bases_near_identical(self) -> None:
        """Two near-identical bases (cond >> 30) must trigger fallback
        rather than producing exploded alphas."""
        rng = np.random.default_rng(9)
        n = 200
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = b1 + 1e-9 * rng.normal(size=n)  # essentially b2 == b1
        y = 0.95 * b1 + rng.normal(scale=0.3, size=n)
        base = np.column_stack([b1, b2])
        p = _linear_residual_multi_fit(y, base)
        assert (
            p["collinear_fallback"] is True
        ), f"expected fallback for cond={p['condition_number']:.2e}; tightly collinear bases must NOT produce extreme alphas"
        assert p["alphas"] == [0.0, 0.0]
        # Round-trip still works (T = y - mean(y), inverse adds it back).
        T = _linear_residual_multi_forward(y, base, p)
        y_back = _linear_residual_multi_inverse(T, base, p)
        np.testing.assert_allclose(y, y_back, rtol=1e-12)

    def test_orthogonal_bases_no_fallback(self) -> None:
        """Truly orthogonal bases must NOT trigger the gate."""
        rng = np.random.default_rng(10)
        n = 500
        b1 = rng.normal(size=n)
        b2 = rng.normal(size=n)  # independent of b1
        # Make sure they're empirically near-orthogonal.
        assert abs(np.corrcoef(b1, b2)[0, 1]) < 0.15
        y = 0.5 * b1 + 0.3 * b2 + rng.normal(scale=0.1, size=n)
        base = np.column_stack([b1, b2])
        p = _linear_residual_multi_fit(y, base)
        assert p["collinear_fallback"] is False, f"unexpected fallback for orthogonal bases; cond={p['condition_number']:.2f}"


class TestDomain:
    """Groups tests covering domain."""
    def test_domain_rejects_nan_in_any_base_column(self) -> None:
        """Domain rejects nan in any base column."""
        b1 = np.array([1.0, 2.0, np.nan, 4.0])
        b2 = np.array([1.0, 2.0, 3.0, 4.0])
        base = np.column_stack([b1, b2])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        mask = _linear_residual_multi_domain(y, base)
        np.testing.assert_array_equal(mask, [True, True, False, True])

    def test_domain_rejects_inf_in_any_base_column(self) -> None:
        """Domain rejects inf in any base column."""
        b1 = np.array([1.0, 2.0, 3.0, 4.0])
        b2 = np.array([1.0, np.inf, 3.0, 4.0])
        base = np.column_stack([b1, b2])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        mask = _linear_residual_multi_domain(y, base)
        np.testing.assert_array_equal(mask, [True, False, True, True])

    def test_domain_y_none_only_checks_base(self) -> None:
        """Domain y none only checks base."""
        b1 = np.array([1.0, np.nan, 3.0])
        b2 = np.array([1.0, 2.0, 3.0])
        base = np.column_stack([b1, b2])
        mask = _linear_residual_multi_domain(None, base)
        np.testing.assert_array_equal(mask, [True, False, True])


class TestForwardInverseValidation:
    """Groups tests covering forward inverse validation."""
    def test_forward_raises_on_alpha_count_mismatch(self) -> None:
        """Forward raises on alpha count mismatch."""
        rng = np.random.default_rng(11)
        n = 100
        base = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        y = rng.normal(size=n)
        # Fitted on 2-D, then call forward with 3-D base -> dimension mismatch
        p = _linear_residual_multi_fit(y, base)
        base_3d = np.column_stack([base, rng.normal(size=n)])
        with pytest.raises(ValueError, match="3 columns"):
            _linear_residual_multi_forward(y, base_3d, p)

    def test_inverse_raises_on_alpha_count_mismatch(self) -> None:
        """Inverse raises on alpha count mismatch."""
        rng = np.random.default_rng(12)
        n = 100
        base = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        y = rng.normal(size=n)
        p = _linear_residual_multi_fit(y, base)
        T_hat = rng.normal(size=n)
        with pytest.raises(ValueError, match="1 columns"):
            _linear_residual_multi_inverse(T_hat, base[:, 0], p)


# ---------------------------------------------------------------------------
# Biz_value: when does multi-base actually beat single-base?
# ---------------------------------------------------------------------------


class TestBizValueMultiBaseBeatsSingle:
    """Multi-base linear_residual must produce a residual T that is
    MORE predictable from the remaining features than single-base when
    a second base carries orthogonal structural signal.

    Metric: variance of T (lower=better, removes more structural
    variance from y) AND correlation of T with the second base
    (should drop close to zero after multi-base; remains substantial
    after single-base).
    """

    def _make_two_base_dgp(
        self,
        n: int = 5000,
        seed: int = 0,
    ) -> tuple:
        """y = 0.95 * b1 + 0.5 * b2 + epsilon
        b1 (dominant lag-like) and b2 (orthogonal trend) are independent.
        """
        rng = np.random.default_rng(seed)
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=0.0, scale=3.0, size=n)
        y = 0.95 * b1 + 0.5 * b2 + rng.normal(scale=0.3, size=n)
        return b1, b2, y

    def test_multi_base_residual_variance_strictly_lower(self) -> None:
        """Multi base residual variance strictly lower."""
        b1, b2, y = self._make_two_base_dgp()
        single = get_transform("linear_residual")
        multi = get_transform("linear_residual_multi")
        p_single = single.fit(y, b1)
        T_single = single.forward(y, b1, p_single)
        p_multi = multi.fit(y, np.column_stack([b1, b2]))
        T_multi = multi.forward(y, np.column_stack([b1, b2]), p_multi)
        # Single-base residual still carries the b2 component (variance
        # ~ Var(0.5*b2 + eps) = 0.5^2 * 9 + 0.09 ~= 2.34).
        # Multi-base residual is just epsilon (variance ~= 0.09).
        var_single = float(np.var(T_single))
        var_multi = float(np.var(T_multi))
        assert (
            var_multi < var_single * 0.5
        ), f"expected multi-base residual variance to be <50% of single-base; got var_single={var_single:.4f}, var_multi={var_multi:.4f}"
        # And reasonably close to the true epsilon variance (~0.09).
        assert var_multi < 0.5

    def test_multi_base_residual_decorrelated_from_b2(self) -> None:
        """Multi base residual decorrelated from b2."""
        b1, b2, y = self._make_two_base_dgp()
        single = get_transform("linear_residual")
        multi = get_transform("linear_residual_multi")
        p_single = single.fit(y, b1)
        T_single = single.forward(y, b1, p_single)
        p_multi = multi.fit(y, np.column_stack([b1, b2]))
        T_multi = multi.forward(y, np.column_stack([b1, b2]), p_multi)
        # Single-base residual must still correlate strongly with b2
        # (the structural variance multi-base would have removed).
        corr_single = abs(float(np.corrcoef(T_single, b2)[0, 1]))
        corr_multi = abs(float(np.corrcoef(T_multi, b2)[0, 1]))
        assert corr_single > 0.5, f"sanity check: single-base residual must still carry b2 signal; got |corr|={corr_single:.3f}"
        assert corr_multi < 0.1, f"multi-base residual must be near-orthogonal to b2; got |corr|={corr_multi:.3f}"

    def test_round_trip_preserves_y_on_dgp(self) -> None:
        """Inverse must exactly recover y up to float precision."""
        b1, b2, y = self._make_two_base_dgp()
        multi = get_transform("linear_residual_multi")
        base = np.column_stack([b1, b2])
        p = multi.fit(y, base)
        T = multi.forward(y, base, p)
        y_back = multi.inverse(T, base, p)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# Wrapper integration: CompositeTargetEstimator with multi-base
# ---------------------------------------------------------------------------

import pandas as pd

lgb = pytest.importorskip("lightgbm")


class TestCompositeTargetEstimatorMultiBase:
    """``CompositeTargetEstimator`` fit/predict round-trip with
    ``transform_name='linear_residual_multi'`` and the new
    ``base_columns`` constructor argument."""

    def _make_dataset(self, n: int = 800, seed: int = 0) -> tuple:
        """Make dataset."""
        rng = np.random.default_rng(seed)
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=0.0, scale=3.0, size=n)
        x_other = rng.normal(size=n)  # unrelated
        y = 0.95 * b1 + 0.5 * b2 + 0.2 * x_other + rng.normal(scale=0.3, size=n)
        df = pd.DataFrame(
            {
                "b1": b1,
                "b2": b2,
                "x_other": x_other,
            }
        )
        return df, y

    def test_fit_predict_round_trip(self) -> None:
        """Fit predict round trip."""
        from mlframe.training.composite import CompositeTargetEstimator

        df, y = self._make_dataset(n=600)
        # Hold out 100 rows for predict.
        train_X = df.iloc[:500]
        train_y = y[:500]
        test_X = df.iloc[500:]
        test_y = y[500:]
        inner = lgb.LGBMRegressor(
            n_estimators=50,
            num_leaves=15,
            verbose=-1,
            random_state=0,
        )
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="linear_residual_multi",
            base_columns=("b1", "b2"),
            drop_invalid_rows=True,
        )
        wrap.fit(train_X, train_y)
        preds = wrap.predict(test_X)
        assert preds.shape == (len(test_X),)
        assert np.all(np.isfinite(preds))
        # Predictions in y-scale; RMSE should be a small fraction of
        # train-y std (DGP noise is 0.3 + 0.2*x_other unmodelled by the
        # transform but partially captured by inner; expect RMSE < 1.0
        # on data with std ~ 3-5).
        rmse = float(np.sqrt(np.mean((preds - test_y) ** 2)))
        train_std = float(np.std(train_y))
        assert rmse < train_std * 0.5, f"multi-base wrapper RMSE {rmse:.3f} should be << train-y std {train_std:.3f}"

    def test_backcompat_base_column_singleton_path(self) -> None:
        """Passing legacy ``base_column='b1'`` (single string) with
        ``linear_residual_multi`` works: wrapper resolves it to a
        one-element tuple internally."""
        from mlframe.training.composite import CompositeTargetEstimator

        df, y = self._make_dataset(n=400)
        inner = lgb.LGBMRegressor(
            n_estimators=30,
            num_leaves=7,
            verbose=-1,
            random_state=0,
        )
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="linear_residual_multi",
            base_column="b1",  # legacy single-column path
        )
        wrap.fit(df, y)
        preds = wrap.predict(df)
        assert preds.shape == (len(df),)
        assert np.all(np.isfinite(preds))

    def test_fit_requires_base_column_or_base_columns(self) -> None:
        """Fit requires base column or base columns."""
        from mlframe.training.composite import CompositeTargetEstimator

        df, y = self._make_dataset(n=200)
        inner = lgb.LGBMRegressor(n_estimators=10, num_leaves=5, verbose=-1)
        # Neither base_column nor base_columns set.
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="linear_residual_multi",
        )
        with pytest.raises(ValueError, match="base_column.*base_columns"):
            wrap.fit(df, y)

    def test_multi_base_beats_single_base_on_two_factor_dgp(self) -> None:
        """Biz_value: multi-base wrapper should produce lower test
        RMSE than single-base on the same DGP where b2 carries
        independent structural signal."""
        from mlframe.training.composite import CompositeTargetEstimator

        df, y = self._make_dataset(n=1200)
        train_X, test_X = df.iloc[:1000], df.iloc[1000:]
        train_y, test_y = y[:1000], y[1000:]

        def _rmse(base_columns):
            """Rmse."""
            inner = lgb.LGBMRegressor(
                n_estimators=80,
                num_leaves=15,
                verbose=-1,
                random_state=0,
            )
            wrap = CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual_multi",
                base_columns=base_columns,
            )
            wrap.fit(train_X, train_y)
            preds = wrap.predict(test_X)
            return float(np.sqrt(np.mean((preds - test_y) ** 2)))

        rmse_single = _rmse(("b1",))
        rmse_multi = _rmse(("b1", "b2"))
        assert (
            rmse_multi < rmse_single
        ), f"multi-base must beat single-base on a DGP where b2 carries orthogonal signal; got single={rmse_single:.4f}, multi={rmse_multi:.4f}"
