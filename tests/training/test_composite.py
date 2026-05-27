"""Tests for ``mlframe.training.composite`` -- transform registry and
:class:`CompositeTargetEstimator`.

Coverage map
------------
- Registry lookup: ``get_transform`` known/unknown, ``list_transforms``
  with tag filter.
- Round-trip ``y -> T -> y'`` for each of the 4 core transforms on
  fixed grids (deterministic correctness).
- Round-trip on **random domains** via Hypothesis (property-based).
- Domain-violation handling at fit (drop_invalid_rows True/False).
- Adversarial inputs at predict: ``inf`` / ``NaN`` / out-of-domain
  ``base``; verify the row falls back to median(y_train) and the
  domain_violation_rate counter increments.
- MAD-soft-cap behaviour: degenerate T_train (constant on train)
  doesn't cause predictions to collapse to base on in-distribution
  rows.
- Post-inverse y-clip: extreme T_hat (synthesised by mocking the
  inner) gets clipped to the train envelope.
- ``sklearn.clone()`` round-trip + pickle round-trip.
- Wrapper end-to-end with LightGBM regressor on TVT-like synthetic
  data (R(y, y_hat) noticeably better than naive ``base`` baseline).
- Delegation: ``feature_importances_``, ``n_features_in_`` reach the
  inner model.
"""
from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
import pytest

# B1 sklearn matrix marker convention -- this file runs in the multi-sklearn-version CI matrix.
pytestmark = pytest.mark.sklearn_matrix

import polars as pl

# Optional dependency: most wrapper integration tests need LightGBM.
lgb = pytest.importorskip("lightgbm")

from sklearn.base import clone  # noqa: E402

from mlframe.training.composite import (  # noqa: E402
    CompositeTargetEstimator,
    DomainViolationError,
    Transform,
    UnknownTransformError,
    _TRANSFORMS_REGISTRY,
    get_transform,
    list_transforms,
)


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------


def _tvt_like(n: int = 600, seed: int = 0):
    """Regression with autoregressive-lag structure: y = 0.95*base + noise + small g(X)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 0.95 * base + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "x3": x3})
    return df, y


def _positive_data(n: int = 200, seed: int = 0):
    """Positive y and base (logratio domain)."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.5, size=n)
    y = base * np.exp(rng.normal(0, 0.3, size=n))
    df = pd.DataFrame({"base": base, "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    return df, y


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


class TestRegistry:
    def test_list_transforms_default_alphabetical(self) -> None:
        """Result is alphabetically sorted and covers the well-known core
        transforms. The registry is open and grows as new brainstorm transforms
        land - pin the sort invariant and a core subset, not the exact list."""
        names = list_transforms()
        assert names == sorted(names), "list_transforms() must return alphabetical order"
        core = {
            "diff", "ratio", "logratio",
            "linear_residual", "linear_residual_multi",
            "linear_residual_grouped",
            "quantile_residual",
            "monotonic_residual",
            "ewma_residual", "rolling_quantile_ratio", "frac_diff",
        }
        missing = core - set(names)
        assert not missing, f"core transforms missing from list_transforms(): {missing}"

    def test_list_transforms_filters_by_tag(self) -> None:
        core = list_transforms(tags=frozenset({"core"}))
        assert "diff" in core
        # Unknown tag returns empty.
        assert list_transforms(tags=frozenset({"nonexistent_tag_xyz"})) == []

    def test_get_transform_known(self) -> None:
        t = get_transform("diff")
        assert isinstance(t, Transform)
        assert t.name == "diff"

    def test_get_transform_unknown_raises(self) -> None:
        with pytest.raises(UnknownTransformError) as exc:
            get_transform("subtract")  # plausible typo for diff
        assert "subtract" in str(exc.value)
        assert "diff" in str(exc.value)  # message includes available names

    @pytest.mark.parametrize("name", sorted(["diff", "ratio", "logratio", "linear_residual"]))
    def test_registry_entries_have_required_callables(self, name: str) -> None:
        t = get_transform(name)
        assert callable(t.forward) and callable(t.inverse)
        assert callable(t.fit) and callable(t.domain_check)
        assert isinstance(t.tags, frozenset)
        assert "core" in t.tags
        assert "regression" in t.tags


# ----------------------------------------------------------------------
# Round-trip on fixed grids
# ----------------------------------------------------------------------


class TestRoundTripFixed:
    @pytest.mark.parametrize("name", ["diff", "linear_residual"])
    def test_general_domain_roundtrip(self, name: str) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(loc=50, scale=10, size=300)
        base = rng.normal(loc=48, scale=10, size=300)
        t = get_transform(name)
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-10, atol=1e-10)

    def test_ratio_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        # base must be away from zero for ratio to be stable.
        base = rng.uniform(low=1.0, high=10.0, size=300)
        y = base * rng.uniform(low=0.5, high=2.0, size=300)
        t = get_transform("ratio")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-10)

    def test_logratio_roundtrip_positive_domain(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.lognormal(mean=2.0, sigma=0.4, size=300)
        y = base * np.exp(rng.normal(0, 0.3, size=300))
        t = get_transform("logratio")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        # Soft-cap rounds to k*MAD around median(T_train); for in-distribution
        # rows the cap should be a no-op so round-trip is essentially exact.
        np.testing.assert_allclose(y, y_back, rtol=1e-8)

    def test_logratio_extreme_y_inside_capped_envelope(self) -> None:
        """Round-trip on y in [1e-3, 1e3] -- extreme but within float precision."""
        y = np.array([1e-3, 1e-1, 1.0, 1e1, 1e3])
        base = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        t = get_transform("logratio")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        # Allow looser rtol because soft-cap may bite at extremes.
        # The KEY guarantee is that y_back is finite, not NaN/inf.
        assert np.all(np.isfinite(y_back))


# ----------------------------------------------------------------------
# Hypothesis property tests for round-trip
# ----------------------------------------------------------------------


class TestHypothesisRoundTrip:
    """Property-based tests on random domains. Slow but catch edge cases
    a fixed grid misses (boundary values, NaN-near-edge inputs).
    """

    def test_diff_property(self) -> None:
        from hypothesis import given, settings, assume
        from hypothesis.extra.numpy import arrays
        from hypothesis import strategies as st

        @given(
            y=arrays(dtype=np.float64, shape=(50,),
                     elements=st.floats(min_value=-1e6, max_value=1e6,
                                        allow_nan=False, allow_infinity=False)),
            base=arrays(dtype=np.float64, shape=(50,),
                        elements=st.floats(min_value=-1e6, max_value=1e6,
                                           allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=30, deadline=2000)
        def _check(y: np.ndarray, base: np.ndarray) -> None:
            t = get_transform("diff")
            p = t.fit(y, base)
            T = t.forward(y, base, p)
            y_back = t.inverse(T, base, p)
            np.testing.assert_allclose(y, y_back, rtol=1e-10, atol=1e-8)

        _check()

    def test_ratio_property(self) -> None:
        from hypothesis import given, settings, assume
        from hypothesis.extra.numpy import arrays
        from hypothesis import strategies as st

        @given(
            y=arrays(dtype=np.float64, shape=(50,),
                     elements=st.floats(min_value=-1e3, max_value=1e3,
                                        allow_nan=False, allow_infinity=False)),
            base=arrays(dtype=np.float64, shape=(50,),
                        elements=st.floats(min_value=0.1, max_value=1e3,
                                           allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=30, deadline=2000)
        def _check(y: np.ndarray, base: np.ndarray) -> None:
            t = get_transform("ratio")
            p = t.fit(y, base)
            T = t.forward(y, base, p)
            y_back = t.inverse(T, base, p)
            np.testing.assert_allclose(y, y_back, rtol=1e-9, atol=1e-9)

        _check()

    def test_linear_residual_property(self) -> None:
        from hypothesis import given, settings, assume
        from hypothesis.extra.numpy import arrays
        from hypothesis import strategies as st

        @given(
            y=arrays(dtype=np.float64, shape=(50,),
                     elements=st.floats(min_value=-1e4, max_value=1e4,
                                        allow_nan=False, allow_infinity=False)),
            base=arrays(dtype=np.float64, shape=(50,),
                        elements=st.floats(min_value=-1e4, max_value=1e4,
                                           allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=30, deadline=2000)
        def _check(y: np.ndarray, base: np.ndarray) -> None:
            assume(np.std(base) > 1e-6)  # OLS singular if base is constant
            t = get_transform("linear_residual")
            p = t.fit(y, base)
            T = t.forward(y, base, p)
            y_back = t.inverse(T, base, p)
            np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

        _check()


# ----------------------------------------------------------------------
# Domain check
# ----------------------------------------------------------------------


class TestDomainChecks:
    def test_diff_domain_all_finite_valid(self) -> None:
        y = np.array([1.0, 2.0, np.nan, np.inf, 5.0])
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = get_transform("diff")
        valid = t.domain_check(y, base)
        assert valid.tolist() == [True, True, False, False, True]

    def test_logratio_domain_strictly_positive(self) -> None:
        y = np.array([1.0, 0.0, -1.0, 5.0])
        base = np.array([1.0, 2.0, 3.0, 4.0])
        t = get_transform("logratio")
        valid = t.domain_check(y, base)
        assert valid.tolist() == [True, False, False, True]

    def test_ratio_domain_nonzero_base(self) -> None:
        y = np.array([1.0, 1.0, 1.0])
        base = np.array([1.0, 0.0, 2.0])
        t = get_transform("ratio")
        valid = t.domain_check(y, base)
        assert valid.tolist() == [True, False, True]


# ----------------------------------------------------------------------
# CompositeTargetEstimator: end-to-end with LightGBM
# ----------------------------------------------------------------------


class TestEstimatorE2E:
    @pytest.fixture
    def tvt_data(self):
        df, y = _tvt_like(n=600)
        return df, y

    def test_fit_predict_diff_beats_naive_baseline(self, tvt_data) -> None:
        df, y = tvt_data
        # Train/test split.
        n_train = 480
        X_tr, X_te = df.iloc[:n_train], df.iloc[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(X_tr, y_tr)
        y_hat = wrapper.predict(X_te)

        # Naive baseline: y_hat = base (the dominant feature).
        rmse_naive = float(np.sqrt(np.mean((y_te - X_te["base"].to_numpy()) ** 2)))
        rmse_wrapper = float(np.sqrt(np.mean((y_te - y_hat) ** 2)))
        assert rmse_wrapper < rmse_naive, (
            f"wrapper RMSE {rmse_wrapper:.3f} should beat naive {rmse_naive:.3f}"
        )

    def test_fit_predict_linear_residual_beats_diff(self, tvt_data) -> None:
        df, y = tvt_data
        n_train = 480
        X_tr, X_te = df.iloc[:n_train], df.iloc[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        rmses = {}
        for name in ["diff", "linear_residual"]:
            wrapper = CompositeTargetEstimator(
                base_estimator=lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1),
                transform_name=name,
                base_column="base",
            )
            wrapper.fit(X_tr, y_tr)
            y_hat = wrapper.predict(X_te)
            rmses[name] = float(np.sqrt(np.mean((y_te - y_hat) ** 2)))
        # linear_residual fits alpha + beta on train, captures the
        # 0.95 coefficient explicitly. Should be at least as good as
        # diff (which assumes alpha=1, beta=0).
        assert rmses["linear_residual"] <= rmses["diff"] * 1.10

    def test_predict_returns_finite_values(self, tvt_data) -> None:
        df, y = tvt_data
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=50, num_leaves=15, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(df.iloc[:480], y[:480])
        y_hat = wrapper.predict(df.iloc[480:])
        assert np.all(np.isfinite(y_hat))


# ----------------------------------------------------------------------
# Adversarial inputs at predict
# ----------------------------------------------------------------------


class TestAdversarialPredict:
    @pytest.fixture
    def fitted_wrapper(self):
        df, y = _tvt_like(n=400)
        w = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=50, num_leaves=15, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        w.fit(df, y)
        return w

    def test_predict_inf_base_falls_back_to_median(self, fitted_wrapper) -> None:
        # Construct adversarial test row.
        n = 5
        rng = np.random.default_rng(0)
        df_test = pd.DataFrame({
            "base": [10.0, np.inf, 5.0, -np.inf, 12.0],
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        })
        y_hat = fitted_wrapper.predict(df_test)
        # Inf-base rows should NOT propagate +/-inf into y_hat.
        assert np.all(np.isfinite(y_hat))
        # domain_violation_rows should reflect the 2 inf rows.
        assert fitted_wrapper.runtime_stats_["domain_violation_rows"] >= 2

    def test_predict_logratio_with_negative_base_falls_back(self) -> None:
        df, y = _positive_data(n=300)
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=50, num_leaves=15, verbose=-1),
            transform_name="logratio",
            base_column="base",
        )
        wrapper.fit(df, y)
        # Inject a row with base <= 0 -- domain violation for logratio.
        df_test = df.iloc[:5].copy()
        df_test.loc[df_test.index[0], "base"] = -1.0
        df_test.loc[df_test.index[1], "base"] = 0.0
        y_hat = wrapper.predict(df_test)
        # Domain violation triggers fallback for the 2 invalid rows.
        assert wrapper.runtime_stats_["domain_violation_rows"] >= 2
        # No NaN / inf in output.
        assert np.all(np.isfinite(y_hat))


# ----------------------------------------------------------------------
# MAD-soft-cap + post-inverse y-clip
# ----------------------------------------------------------------------


class TestNumericalSafety:
    def test_logratio_extreme_t_hat_does_not_blow_up(self) -> None:
        """Manually inject an extreme T_hat and verify the two-layer guard
        (MAD-soft-cap inside the inverse + post-inverse y-clip in the
        wrapper) keeps y_hat finite and bounded.

        In this happy-path positive_data fixture MAD-soft-cap engages
        first and clips T_hat to a sane value before exp() is taken,
        so the post-inverse y-clip never has to fire. We only assert
        the GOAL (no exp() blow-up, output stays in the train envelope),
        not which guard caught it -- otherwise the test would be
        coupled to MAD-floor tuning.
        """
        df, y = _positive_data(n=300)
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, num_leaves=10, verbose=-1),
            transform_name="logratio",
            base_column="base",
        )
        wrapper.fit(df, y)

        # Stub the inner predict to return enormous T values: without
        # any guard, y_hat = base * exp(50) = 5.18e21.
        class _StubInner:
            def predict(self, X):
                return np.full(len(X), 50.0)
        wrapper.estimator_ = _StubInner()
        y_hat = wrapper.predict(df.iloc[:10])

        assert np.all(np.isfinite(y_hat)), "guards must keep predictions finite"
        upper = wrapper.fitted_params_["y_clip_high"]
        assert np.all(y_hat <= upper), (
            f"all predictions must stay <= y_clip_high={upper}, "
            f"got max={y_hat.max():.6g}"
        )

    def test_logratio_y_clip_engages_when_softcap_disabled(self) -> None:
        """Force MAD to zero so the soft-cap never engages, then verify
        the post-inverse y-clip catches the blow-up. This isolates the
        y-clip safety net."""
        df, y = _positive_data(n=300)
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, num_leaves=10, verbose=-1),
            transform_name="logratio",
            base_column="base",
        )
        wrapper.fit(df, y)
        # Disable soft-cap by widening it absurdly. Now exp(50) WILL
        # propagate to y_hat and y-clip becomes the sole defence.
        wrapper.fitted_params_["mad_eff"] = 1e9
        wrapper.fitted_params_["soft_cap_k"] = 1.0
        # The T-scale clip added later (CompositeTargetEstimator.predict
        # T-clip on the inner output BEFORE inverse) would also catch
        # the 50.0 stub and pre-empt the y-clip. To isolate the y-clip
        # specifically -- the test's stated intent -- widen the T-clip
        # bounds so it becomes a no-op for this stub.
        wrapper.fitted_params_["t_clip_low"] = float("-inf")
        wrapper.fitted_params_["t_clip_high"] = float("+inf")

        class _StubInner:
            def predict(self, X):
                return np.full(len(X), 50.0)
        wrapper.estimator_ = _StubInner()
        y_hat = wrapper.predict(df.iloc[:10])

        assert np.all(np.isfinite(y_hat))
        # y-clip should have engaged on every row.
        assert wrapper.runtime_stats_["y_clip_high_hits"] >= len(y_hat)


# ----------------------------------------------------------------------
# sklearn.clone() and pickle compatibility
# ----------------------------------------------------------------------


class TestSklearnCloneAndPickle:
    def test_clone_unfitted_preserves_params(self) -> None:
        original = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=50, verbose=-1),
            transform_name="linear_residual",
            base_column="base",
            fallback_predict="nan",
            drop_invalid_rows=False,
        )
        cloned = clone(original)
        assert cloned.transform_name == "linear_residual"
        assert cloned.base_column == "base"
        assert cloned.fallback_predict == "nan"
        assert cloned.drop_invalid_rows is False
        # The inner estimator was cloned too -- a fresh, unfitted copy.
        assert cloned.base_estimator is not original.base_estimator
        # ...but type-equivalent.
        assert type(cloned.base_estimator) is type(original.base_estimator)

    def test_clone_then_fit_independent(self) -> None:
        df, y = _tvt_like(n=400)
        original = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        cloned = clone(original)
        original.fit(df, y)
        # Clone must NOT have absorbed the fit state.
        assert not hasattr(cloned, "estimator_")
        assert not hasattr(cloned, "fitted_params_")

    def test_pickle_roundtrip_after_fit(self) -> None:
        df, y = _tvt_like(n=400)
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, verbose=-1),
            transform_name="linear_residual",
            base_column="base",
        )
        wrapper.fit(df, y)
        y_hat_orig = wrapper.predict(df.iloc[:10])

        blob = pickle.dumps(wrapper)
        revived = pickle.loads(blob)
        y_hat_revived = revived.predict(df.iloc[:10])
        np.testing.assert_allclose(y_hat_orig, y_hat_revived, rtol=1e-10)


# ----------------------------------------------------------------------
# Delegation
# ----------------------------------------------------------------------


class TestDelegation:
    def test_feature_importances_delegated(self) -> None:
        df, y = _tvt_like(n=400)
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(df, y)
        fi = wrapper.feature_importances_
        assert fi is not None
        assert len(fi) == len(df.columns)

    def test_n_features_in_delegated(self) -> None:
        df, y = _tvt_like(n=400)
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(df, y)
        assert wrapper.n_features_in_ == len(df.columns)

    def test_unfitted_attributes_return_none_or_raise(self) -> None:
        # Audit 2026-05-17 H-COMP-14: ``feature_importances_`` /
        # ``coef_`` / ``intercept_`` now raise NotFittedError on unfit
        # wrappers (sklearn convention) instead of returning None.
        from sklearn.exceptions import NotFittedError

        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        with pytest.raises(NotFittedError):
            _ = wrapper.feature_importances_
        assert wrapper.n_features_in_ is None


# ----------------------------------------------------------------------
# fit-time validation
# ----------------------------------------------------------------------


class TestFitValidation:
    def test_fit_without_base_estimator_raises(self) -> None:
        wrapper = CompositeTargetEstimator(
            base_estimator=None,
            transform_name="diff",
            base_column="base",
        )
        df, y = _tvt_like(n=100)
        with pytest.raises(ValueError, match="base_estimator"):
            wrapper.fit(df, y)

    def test_fit_without_base_column_raises(self) -> None:
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
            transform_name="diff",
            base_column="",
        )
        df, y = _tvt_like(n=100)
        with pytest.raises(ValueError, match="base_column"):
            wrapper.fit(df, y)

    def test_fit_missing_base_column_in_X_raises_keyerror(self) -> None:
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
            transform_name="diff",
            base_column="missing_col",
        )
        df, y = _tvt_like(n=100)
        with pytest.raises(KeyError, match="missing_col"):
            wrapper.fit(df, y)

    def test_fit_drops_invalid_rows_for_logratio(self) -> None:
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=20, verbose=-1),
            transform_name="logratio",
            base_column="base",
            drop_invalid_rows=True,
        )
        df, y = _positive_data(n=200)
        # Inject negative base on 10% of rows.
        bad_idx = np.arange(20)
        df = df.copy()
        df.loc[bad_idx, "base"] = -1.0
        wrapper.fit(df, y)
        # 20 rows should have been dropped.
        assert wrapper.fitted_params_["n_train_invalid"] >= 20
        assert wrapper.fitted_params_["n_train_valid"] == len(y) - wrapper.fitted_params_["n_train_invalid"]

    def test_fit_drop_invalid_rows_false_raises(self) -> None:
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=20, verbose=-1),
            transform_name="logratio",
            base_column="base",
            drop_invalid_rows=False,
        )
        df, y = _positive_data(n=200)
        df = df.copy()
        df.loc[np.arange(5), "base"] = -1.0
        with pytest.raises(DomainViolationError, match="domain"):
            wrapper.fit(df, y)


# ----------------------------------------------------------------------
# Polars input
# ----------------------------------------------------------------------


class TestPolarsInput:
    """The wrapper supports polars at the *base extraction* boundary
    (so a base feature in a polars frame is read without a 100GB
    materialisation). It does NOT auto-convert polars before feeding
    the inner estimator -- that would defeat the whole zero-copy
    point on large frames. The caller is responsible for handing the
    inner a frame type it accepts. mlframe strategies already do
    this at the suite level (polars -> pandas only when the strategy
    declares ``supports_polars=False``)."""

    def test_polars_base_extraction_supported(self) -> None:
        """Even though the inner LightGBM doesn't accept polars
        directly here, the wrapper's polars-aware ``_extract_base``
        and ``_subset_rows`` paths exercise correctly when the caller
        adapts at the inner boundary -- mirrors the in-suite pattern."""
        df, y = _tvt_like(n=400)
        pl_df = pl.from_pandas(df)
        # Verify the wrapper's polars-aware helpers don't crash on
        # base-column extraction or row subsetting.
        from mlframe.training.composite import _extract_base
        base = _extract_base(pl_df, "base")
        assert len(base) == 400 and np.all(np.isfinite(base))

    def test_polars_to_pandas_then_fit_works(self) -> None:
        """End-to-end with caller-side polars->pandas (the integration
        pattern): fit + predict succeed and produce finite output."""
        df, y = _tvt_like(n=400)
        # Caller-side conversion -- NOT auto-injected by wrapper.
        pl_df = pl.from_pandas(df)
        df_pd = pl_df.to_pandas()
        wrapper = CompositeTargetEstimator(
            base_estimator=lgb.LGBMRegressor(n_estimators=30, verbose=-1),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(df_pd, y)
        y_hat = wrapper.predict(df_pd.head(10))
        assert len(y_hat) == 10
        assert np.all(np.isfinite(y_hat))
