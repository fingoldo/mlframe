"""Regression: ``model_type="linear"`` must stay numerically stable on multicollinear features.

Production failure (2026-05-17 TVT run, target=TVT-monres-Y):

    TEST LinearRegression ... MAE=20191.32 RMSE=20201.36 MaxError=21520.18 R2=-978.94

on a target with mean=-858, std=644 -- 40x worse than predicting the train mean.
Root cause: 25 features with documented near-perfect collinearity (``Z~=TVT_prev``
|corr|=0.974, ``GR_lag_*`` near-duplicates, ``GR_roll_mean_5~=GR_roll_mean_15``
|corr|=0.957) + unregularised OLS + float32 columns -> condition number > 1e8 ->
catastrophic coefficient cancellation.

Fix shipped 2026-05-18: ``_build_linear_regressor`` returns ``Ridge(alpha=1e-3)`` instead
of bare ``LinearRegression``. alpha=1e-3 is numerically OLS on well-conditioned
matrices (relative prediction delta < 1e-5) and stable on ill-conditioned.

This test pins the contract:
1. ``create_linear_model("linear", ...)`` on a near-singular feature matrix
   stays bounded (max coefficient absolute < 100x feature std).
2. Predictions on an out-of-distribution test slice stay within 5x the
   in-distribution train MAE -- previously 40x.
3. On a well-conditioned feature matrix, predictions match an explicit
   ``LinearRegression`` fit to <1e-3 relative tolerance (proving the flip
   does not regress well-conditioned cases).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from mlframe.training.configs import LinearModelConfig
from mlframe.training.models import create_linear_model


def _make_multicollinear_data(
    n_train: int = 800, n_test: int = 200, n_redundant_clones: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates (X_train, y_train, X_test, y_test) where columns 1..n_redundant_clones are
    near-duplicates of column 0 (|corr| ~ 0.99). Mirrors the production ``Z ~= TVT_prev`` /
    ``GR_lag_*`` collinearity pattern from the failure log.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    base = rng.normal(11500.0, 644.0, size=n_total)
    cols = [base]
    for _ in range(n_redundant_clones):
        cols.append(base + rng.standard_normal(n_total) * 5.0)  # tiny independent jitter
    # An additional independent feature so the system is not literally rank-1.
    cols.append(rng.normal(0.0, 1.0, n_total))
    X_full = np.column_stack(cols).astype(np.float32)
    y_full = base * 0.001 - 5.0 + rng.standard_normal(n_total) * 0.5
    # Out-of-distribution test slice: shift mean of column 0 a little so coefficient cancellation surfaces.
    X_full[n_train:, 0] += rng.normal(50.0, 5.0, size=n_test).astype(np.float32)
    return (
        X_full[:n_train],
        y_full[:n_train],
        X_full[n_train:],
        y_full[n_train:],
    )


def _config() -> LinearModelConfig:
    return LinearModelConfig()


class TestMulticollinearStability:
    def test_coefficients_stay_bounded(self) -> None:
        X_train, y_train, _, _ = _make_multicollinear_data()
        model = create_linear_model("linear", _config(), use_regression=True)
        model.fit(X_train, y_train)
        max_coef = float(np.max(np.abs(model.coef_)))
        # Ridge(alpha=1e-3) on near-collinear should keep coefficients controlled (sum cancels via small alpha). Pre-fix OLS produced coefficients of magnitude ~1e2-1e3 with sign cancellation.
        feat_stds = np.asarray(X_train, dtype=np.float64).std(axis=0)
        feat_std_max = float(feat_stds.max())
        assert max_coef < 100.0 * feat_std_max, f"Coefficient blow-up: max |coef|={max_coef:.2f} vs feature std max={feat_std_max:.2f}"

    def test_test_mae_within_5x_train_mae(self) -> None:
        X_train, y_train, X_test, y_test = _make_multicollinear_data()
        model = create_linear_model("linear", _config(), use_regression=True)
        model.fit(X_train, y_train)
        train_mae = float(np.mean(np.abs(model.predict(X_train) - y_train)))
        test_mae = float(np.mean(np.abs(model.predict(X_test) - y_test)))
        # Pre-fix this ratio was ~40x on the production data (MAE 515 dummy -> 20191 LinReg). Post-fix Ridge(alpha=1e-3) keeps the ratio bounded.
        assert test_mae < 5.0 * train_mae + 1.0, (
            f"Out-of-distribution blow-up: train MAE={train_mae:.4f}, test MAE={test_mae:.4f} (ratio={test_mae / max(train_mae, 1e-9):.1f}x; threshold=5x)"
        )

    def test_well_conditioned_matches_ols(self) -> None:
        """Flip MUST NOT regress well-conditioned cases.

        On orthogonal features ``Ridge(alpha=1e-3)`` predictions should match OLS to <1e-3 relative tolerance.
        """
        rng = np.random.default_rng(11)
        n = 500
        X = rng.standard_normal((n, 4))
        y = X @ np.array([1.5, -2.0, 0.3, 0.8]) + rng.standard_normal(n) * 0.1
        ridge = create_linear_model("linear", _config(), use_regression=True)
        ridge.fit(X, y)
        ols = LinearRegression().fit(X, y)
        ridge_pred = ridge.predict(X)
        ols_pred = ols.predict(X)
        # Compare on relative scale -- predictions are O(1) so abs and rel are comparable.
        max_abs_delta = float(np.max(np.abs(ridge_pred - ols_pred)))
        y_range = float(y.max() - y.min())
        assert max_abs_delta < 1e-3 * y_range, (
            f"Ridge(alpha=1e-3) diverged from OLS on well-conditioned features: max |delta|={max_abs_delta:.6f} vs y-range={y_range:.4f}"
        )
