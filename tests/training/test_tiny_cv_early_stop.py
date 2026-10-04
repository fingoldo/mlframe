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
    """Groups the tiny-CV early-stop acceptance contract: default threshold=inf must run every fold."""

    def test_no_early_stop_when_threshold_inf(self, laplace_residual_dataset) -> None:
        """Default ``early_stop_threshold=inf`` keeps legacy behaviour: all folds always run."""
        y, base, x, params, transform = laplace_residual_dataset
        result = _tiny_cv_rmse_y_scale(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=20,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
            n_jobs=1,
        )
        assert np.isfinite(result)


class TestEarlyStopFires:
    """Groups the tiny-CV early-stop firing contract: a breached threshold must actually cut compute."""

    def test_early_stop_reduces_compute_when_high_threshold_breached(
        self,
        laplace_residual_dataset,
        monkeypatch,
    ) -> None:
        """With a small ``early_stop_threshold``, the partial-mean bound triggers and later folds are not fitted.

        Counts the fold fits actually performed on a 3-fold serial run instead of timing it: the full run fits every fold, the early-stopped
        run stops after the first fold, and the value it returns is the partial mean, above the threshold.
        """
        from mlframe.training.composite.discovery import _screening_tiny_perbin as perbin

        y, base, x, params, transform = laplace_residual_dataset
        fits: list = []
        real_fit_fold_model = perbin._fit_fold_model

        def counting_fit_fold_model(*args, **kwargs):
            """Count one fold fit, then fit it."""
            fits.append(1)
            return real_fit_fold_model(*args, **kwargs)

        monkeypatch.setattr(perbin, "_fit_fold_model", counting_fit_fold_model)
        kwargs = dict(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=50,
            num_leaves=16,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
            n_jobs=1,
        )
        full = _tiny_cv_rmse_y_scale(**kwargs)
        assert len(fits) == 3

        # Threshold WAY below the expected full RMSE: heavy-tail Cauchy noise gives RMSE in the 100s, so fold 1 already breaches it.
        fits.clear()
        early = _tiny_cv_rmse_y_scale(**kwargs, early_stop_threshold=1.0)
        assert len(fits) == 1
        assert early > 1.0
        assert np.isfinite(full) and np.isfinite(early)

    def test_early_stop_threshold_inf_returns_same_value(
        self,
        laplace_residual_dataset,
    ) -> None:
        """When ``early_stop_threshold=inf`` the early-stop branch must never fire, and the returned value must equal the legacy (no-kwarg) call."""
        y, base, x, params, transform = laplace_residual_dataset
        legacy = _tiny_cv_rmse_y_scale(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=20,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
            n_jobs=1,
        )
        new = _tiny_cv_rmse_y_scale(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=x,
            family="lgb",
            n_estimators=20,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
            n_jobs=1,
            early_stop_threshold=float("inf"),
        )
        assert legacy == pytest.approx(new, abs=1e-6)
