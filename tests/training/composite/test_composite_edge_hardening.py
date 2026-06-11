"""Comprehensive predict-path edge hardening for ``CompositeTargetEstimator``.

Pins five robustness contracts on the estimator predict family
(``estimator/_predict.py``):

1. UNSEEN groups at predict (grouped transforms) fall back to the global
   (alpha, beta) -- predict never raises and never leaks NaN for a label that
   was absent at fit.
2. DEGENERATE domains (every base row out-of-domain at predict, incl. an
   all-NaN ``y_train`` whose stored ``y_train_median`` is non-finite) route
   every row to the fallback and return all-finite predictions.
3. DUPLICATE rows predict deterministically and identically for identical rows.
4. pandas vs polars PARITY -- the same X as pandas and as polars yields
   bit-identical predictions across several transforms.
5. SINGLE-row and EMPTY-frame predict do not crash and return the right shape.

These are bug-detector tests: a regression that re-introduces a NaN leak, a
broadcast crash, a non-deterministic predict, or a pandas/polars divergence
fails here rather than weeks into prod.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.estimator._predict import _finite_median_fallback

try:
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - polars optional
    pl = None  # type: ignore
    _HAS_POLARS = False


class _LinearInner(BaseEstimator, RegressorMixin):
    """Deterministic inner: T_hat = w . X + b fit by lstsq.

    Pure-numpy and seed-free so predict is bit-stable across pandas / polars /
    duplicate-row inputs (no tree randomness, no float-order nondeterminism).
    """

    def fit(self, X, y, **kw):
        Xm = self._to_matrix(X)
        self.n_features_in_ = Xm.shape[1]
        A = np.column_stack([Xm, np.ones(Xm.shape[0])])
        coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=np.float64), rcond=None)
        self._coef = coef
        return self

    def predict(self, X):
        Xm = self._to_matrix(X)
        A = np.column_stack([Xm, np.ones(Xm.shape[0])])
        return A @ self._coef

    def predict_quantile(self, X, alpha=0.5):
        base = self.predict(X)
        if np.isscalar(alpha):
            return base + (float(alpha) - 0.5)
        return np.column_stack([base + (float(a) - 0.5) for a in alpha])

    @staticmethod
    def _to_matrix(X) -> np.ndarray:
        if _HAS_POLARS and isinstance(X, pl.DataFrame):
            return X.to_numpy().astype(np.float64, copy=False)
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)


def _frame(n=200, seed=0, with_group=False, n_groups=4):
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 1.0, size=n)  # >0 so logratio/ratio domains hold
    feat = rng.normal(0.0, 1.0, size=n)
    y = 1.3 * base + 0.7 * feat + rng.normal(0.0, 0.1, size=n)
    data = {"base": base, "feat": feat}
    if with_group:
        data["grp"] = np.array([f"g{i % n_groups}" for i in range(n)], dtype=object)
    return pd.DataFrame(data), y


# Transforms exercised for pandas/polars parity + determinism. All single-base,
# requires_base=True, and tolerant of a positive base column.
_PARITY_TRANSFORMS = ["diff", "linear_residual", "logratio", "ratio"]


def _fit_est(transform_name, X, y, **kw):
    est = CompositeTargetEstimator(
        base_estimator=_LinearInner(),
        transform_name=transform_name,
        base_column="base",
        **kw,
    )
    return est.fit(X, y)


# ---------------------------------------------------------------------------
# (1) Unseen groups -> global fallback, finite, no crash.
# ---------------------------------------------------------------------------
class TestUnseenGroups:
    def test_unseen_group_falls_back_global_finite(self) -> None:
        X, y = _frame(n=240, with_group=True, n_groups=4)
        est = CompositeTargetEstimator(
            base_estimator=_LinearInner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
        )
        est.fit(X, y)
        X_pred = X.copy()
        # Inject group labels never seen at fit on a slice of rows.
        X_pred.loc[X_pred.index[:20], "grp"] = "UNSEEN_NEW"
        X_pred.loc[X_pred.index[20:40], "grp"] = "ANOTHER_NEW"
        y_hat = est.predict(X_pred)
        assert y_hat.shape == (len(X_pred),)
        assert np.all(np.isfinite(y_hat)), "unseen-group rows must stay finite"

    def test_all_unseen_groups_does_not_crash(self) -> None:
        X, y = _frame(n=160, with_group=True, n_groups=3)
        est = CompositeTargetEstimator(
            base_estimator=_LinearInner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
        )
        est.fit(X, y)
        X_pred = X.copy()
        X_pred["grp"] = "ALL_BRAND_NEW"  # every row unseen
        y_hat = est.predict(X_pred)
        assert y_hat.shape == (len(X_pred),)
        assert np.all(np.isfinite(y_hat))

    def test_unseen_group_with_nan_base_still_finite(self) -> None:
        """Unseen group AND a non-finite base on the same rows: the domain gate
        must route those rows to the (finite) fallback, not leak NaN."""
        X, y = _frame(n=200, with_group=True, n_groups=4)
        est = CompositeTargetEstimator(
            base_estimator=_LinearInner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
            fallback_predict="y_train_median",
        )
        est.fit(X, y)
        X_pred = X.copy()
        X_pred.loc[X_pred.index[:10], "grp"] = "UNSEEN"
        X_pred.loc[X_pred.index[:10], "base"] = np.nan
        y_hat = est.predict(X_pred)
        assert np.all(np.isfinite(y_hat))


# ---------------------------------------------------------------------------
# (2) Degenerate domains -> every row to fallback, all-finite.
# ---------------------------------------------------------------------------
class TestDegenerateDomain:
    @pytest.mark.parametrize("transform_name", ["logratio", "ratio"])
    def test_all_invalid_base_routes_to_fallback(self, transform_name) -> None:
        X, y = _frame(n=150)
        est = _fit_est(transform_name, X, y, fallback_predict="y_train_median")
        X_pred = X.copy()
        # logratio needs base>0, ratio needs base!=0; zero breaks both.
        X_pred["base"] = 0.0
        y_hat = est.predict(X_pred)
        assert y_hat.shape == (len(X_pred),)
        assert np.all(np.isfinite(y_hat)), "all-invalid domain must stay finite"
        # Every row took the same fallback constant -> constant prediction.
        assert np.allclose(y_hat, y_hat[0])

    def test_all_nonfinite_base_routes_to_fallback(self) -> None:
        X, y = _frame(n=120)
        est = _fit_est("logratio", X, y, fallback_predict="y_train_median")
        X_pred = X.copy()
        X_pred["base"] = np.inf
        y_hat = est.predict(X_pred)
        assert np.all(np.isfinite(y_hat))

    def test_finite_median_fallback_coerces_nan_to_zero(self) -> None:
        """Unit pin on the guard: a non-finite stored ``y_train_median`` (the
        from_fitted_inner degenerate-y_train case) resolves to a finite 0.0."""
        assert _finite_median_fallback({"y_train_median": float("nan")}) == 0.0
        assert _finite_median_fallback({"y_train_median": float("inf")}) == 0.0
        assert _finite_median_fallback({}) == 0.0
        assert _finite_median_fallback({"y_train_median": 3.5}) == 3.5

    def test_degenerate_ytrain_median_does_not_leak_nan(self) -> None:
        """End-to-end: a fitted instance whose stored y_train_median is NaN
        (degenerate domain) must still produce all-finite fallback predictions.

        Mirrors the from_fitted_inner path that can stash NaN; we patch the
        fitted params directly to isolate the predict-side guard."""
        X, y = _frame(n=120)
        est = _fit_est("logratio", X, y, fallback_predict="y_train_median")
        est.fitted_params_["y_train_median"] = float("nan")
        X_pred = X.copy()
        X_pred["base"] = -1.0  # all out-of-domain for logratio
        y_hat = est.predict(X_pred)
        assert np.all(np.isfinite(y_hat)), "NaN median must not leak through"
        assert np.allclose(y_hat, 0.0), "degenerate median coerces to 0.0"


# ---------------------------------------------------------------------------
# (3) Duplicate rows -> deterministic + identical predictions.
# ---------------------------------------------------------------------------
class TestDuplicateRows:
    @pytest.mark.parametrize("transform_name", _PARITY_TRANSFORMS)
    def test_identical_rows_identical_predictions(self, transform_name) -> None:
        X, y = _frame(n=180)
        est = _fit_est(transform_name, X, y)
        # Build a frame whose rows 0..9 are exact copies of row 0.
        X_pred = X.iloc[:10].copy().reset_index(drop=True)
        for i in range(1, 10):
            X_pred.iloc[i] = X.iloc[0]
        y_hat = est.predict(X_pred)
        assert np.all(y_hat == y_hat[0]), "identical rows must predict identically"

    @pytest.mark.parametrize("transform_name", _PARITY_TRANSFORMS)
    def test_predict_is_deterministic(self, transform_name) -> None:
        X, y = _frame(n=180)
        est = _fit_est(transform_name, X, y)
        a = est.predict(X)
        b = est.predict(X)
        assert np.array_equal(a, b), "repeated predict must be bit-identical"


# ---------------------------------------------------------------------------
# (4) pandas vs polars parity -> bit-identical predictions.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")
class TestPandasPolarsParity:
    @pytest.mark.parametrize("transform_name", _PARITY_TRANSFORMS)
    def test_point_predict_parity(self, transform_name) -> None:
        X, y = _frame(n=200, seed=3)
        X_pl = pl.from_pandas(X)
        est_pd = _fit_est(transform_name, X, y)
        est_pl = _fit_est(transform_name, X_pl, y)
        y_pd = est_pd.predict(X)
        y_pl = est_pl.predict(X_pl)
        assert y_pd.shape == y_pl.shape
        assert np.array_equal(y_pd, y_pl), (
            f"pandas vs polars predict diverged for '{transform_name}'"
        )

    @pytest.mark.parametrize("transform_name", ["diff", "linear_residual"])
    def test_quantile_predict_parity(self, transform_name) -> None:
        X, y = _frame(n=200, seed=4)
        X_pl = pl.from_pandas(X)
        est_pd = _fit_est(transform_name, X, y)
        est_pl = _fit_est(transform_name, X_pl, y)
        q_pd = est_pd.predict_quantile(X, [0.1, 0.5, 0.9])
        q_pl = est_pl.predict_quantile(X_pl, [0.1, 0.5, 0.9])
        assert q_pd.shape == q_pl.shape
        assert np.array_equal(q_pd, q_pl)


# ---------------------------------------------------------------------------
# (5) Single-row + empty-frame predict -> no crash, right shape.
# ---------------------------------------------------------------------------
class TestSingleAndEmpty:
    @pytest.mark.parametrize("transform_name", _PARITY_TRANSFORMS)
    def test_single_row_predict(self, transform_name) -> None:
        X, y = _frame(n=150)
        est = _fit_est(transform_name, X, y)
        X_one = X.iloc[[0]].copy()
        y_hat = est.predict(X_one)
        assert y_hat.shape == (1,)
        assert np.all(np.isfinite(y_hat))

    @pytest.mark.parametrize("transform_name", _PARITY_TRANSFORMS)
    def test_empty_frame_predict(self, transform_name) -> None:
        X, y = _frame(n=150)
        est = _fit_est(transform_name, X, y)
        X_empty = X.iloc[0:0].copy()
        y_hat = est.predict(X_empty)
        assert y_hat.shape == (0,)

    def test_single_row_grouped_unseen_predict(self) -> None:
        X, y = _frame(n=160, with_group=True, n_groups=4)
        est = CompositeTargetEstimator(
            base_estimator=_LinearInner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
        )
        est.fit(X, y)
        X_one = X.iloc[[0]].copy()
        X_one["grp"] = "UNSEEN_SINGLE"
        y_hat = est.predict(X_one)
        assert y_hat.shape == (1,)
        assert np.all(np.isfinite(y_hat))

    def test_empty_frame_quantile_predict(self) -> None:
        X, y = _frame(n=150)
        est = _fit_est("diff", X, y)
        X_empty = X.iloc[0:0].copy()
        q = est.predict_quantile(X_empty, [0.1, 0.5, 0.9])
        assert q.shape == (0, 3)
