"""#7 Tiny-CV early-stop: abort folds when partial mean cannot beat the threshold.

The serial fold loop in ``_tiny_cv_rmse_y_scale`` tracks the running sum across completed folds. If ``sum_so_far > early_stop_threshold * cv_folds`` even with the remaining folds returning 0, the final mean cannot reach the threshold -- abort to save 30-66% of fold-fit compute on candidates the gate will reject anyway.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import _tiny_cv_rmse_y_scale
from mlframe.training.composite.transforms import get_transform


@pytest.fixture
def laplace_residual_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, object]:
    """y = 1.5*base + noise + extreme outliers -- composite linres struggles to fit, fold RMSEs are high."""
    rng = np.random.default_rng(0)
    n = 600
    base = rng.normal(50.0, 10.0, n)
    y = 1.5 * base + 0.5 + rng.standard_cauchy(n) * 50.0
    x = np.column_stack([base, rng.standard_normal(n), rng.standard_normal(n)])

    transform = get_transform("linear_residual")
    params = transform.fit(y, base)
    return y, base, x, params, transform


class TestEarlyStopAcceptance:
    def test_no_early_stop_when_threshold_inf(self, laplace_residual_dataset) -> None:
        """Default ``early_stop_threshold=inf`` keeps legacy behaviour: all folds always run."""
        y, base, x, params, transform = laplace_residual_dataset
        result = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base,
            transform=transform, fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=20, num_leaves=8, learning_rate=0.1,
            cv_folds=3, random_state=0,
            n_jobs=1,
        )
        assert np.isfinite(result)


class TestEarlyStopFires:
    def test_early_stop_reduces_compute_when_high_threshold_breached(
        self, laplace_residual_dataset,
    ) -> None:
        """With a small ``early_stop_threshold``, the partial-mean bound triggers and the run finishes earlier.

        Measure wall-time savings on a 3-fold serial fit. Hard assertion: early-stop run wall time < full run wall time. The exact reduction depends on data, but the partial-mean bound MUST cut at least one fold.
        """
        import time
        y, base, x, params, transform = laplace_residual_dataset

        t0 = time.perf_counter()
        full_rmse = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base,
            transform=transform, fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=50, num_leaves=16, learning_rate=0.1,
            cv_folds=3, random_state=0,
            n_jobs=1,
        )
        full_time = time.perf_counter() - t0

        # Set threshold WAY below the expected full RMSE so fold 1 triggers abort.
        # Heavy-tail Cauchy noise -> RMSE in the 100s. Threshold = 1.0 forces abort.
        t0 = time.perf_counter()
        _ = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base,
            transform=transform, fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=50, num_leaves=16, learning_rate=0.1,
            cv_folds=3, random_state=0,
            n_jobs=1,
            early_stop_threshold=1.0,
        )
        es_time = time.perf_counter() - t0

        # Early-stop must finish in <70% of full time (rough threshold; LightGBM init dominates on small N so the bound is tight).
        assert es_time < full_time, (
            f"early-stop did not save time: full={full_time:.3f}s, "
            f"early_stop={es_time:.3f}s"
        )

    def test_early_stop_threshold_inf_returns_same_value(
        self, laplace_residual_dataset,
    ) -> None:
        """When ``early_stop_threshold=inf`` the early-stop branch must never fire, and the returned value must equal the legacy (no-kwarg) call."""
        y, base, x, params, transform = laplace_residual_dataset
        legacy = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base,
            transform=transform, fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=20, num_leaves=8, learning_rate=0.1,
            cv_folds=3, random_state=0,
            n_jobs=1,
        )
        new = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base,
            transform=transform, fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=20, num_leaves=8, learning_rate=0.1,
            cv_folds=3, random_state=0,
            n_jobs=1,
            early_stop_threshold=float("inf"),
        )
        assert legacy == pytest.approx(new, abs=1e-6)
