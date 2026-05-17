"""Tests for the heavy-tail auto-loss recommendation helper (Pack H).

The helper is a pure function on a 1-D target array; it returns per-backend loss names. Tests pin the policy on:
- Gaussian residual -> RMSE (the no-op case).
- Laplace residual -> MAE / L1 (the production failure mode this helper fixes).
- Contaminated / Student-t -> Huber.
- Tiny / empty / constant inputs -> safe RMSE default.

Wiring into ``_phase_train_one_target`` is left to a follow-up; the helper output shape is the public contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.loss_recommendation import recommend_boosting_regression_loss


def _gaussian(n: int = 5000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _laplace(n: int = 5000, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).laplace(loc=0.0, scale=1.0, size=n)


def _student_t(n: int = 5000, df: float = 3.0, seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).standard_t(df=df, size=n)


def _contaminated(n: int = 5000, contam_frac: float = 0.05, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    main = rng.standard_normal(n)
    n_contam = int(n * contam_frac)
    idx = rng.choice(n, size=n_contam, replace=False)
    main[idx] = rng.standard_normal(n_contam) * 30.0
    return main


class TestRecommendation:
    def test_gaussian_picks_rmse(self) -> None:
        rec = recommend_boosting_regression_loss(_gaussian())
        assert rec["cb"] == "RMSE"
        assert rec["lgb"] == "regression"
        assert rec["xgb"] == "reg:squarederror"
        assert abs(rec["excess_kurt"]) < 0.5  # well within Gaussian tolerance

    def test_laplace_picks_mae(self) -> None:
        rec = recommend_boosting_regression_loss(_laplace())
        assert rec["cb"] == "MAE"
        assert rec["lgb"] == "regression_l1"
        assert rec["xgb"] == "reg:absoluteerror"
        assert rec["excess_kurt"] > 1.5

    def test_student_t_picks_robust(self) -> None:
        """Student-t with df=3 has theoretical kurt -> infinity; ladder picks Huber (extreme) or MAE depending on sample."""
        rec = recommend_boosting_regression_loss(_student_t(df=3.0))
        assert rec["cb"] in {"MAE", "Huber:delta=1.345"}
        assert rec["lgb"] in {"regression_l1", "huber"}
        assert rec["excess_kurt"] > 1.5

    def test_contaminated_picks_huber(self) -> None:
        """5% contamination at sigma=30 -> excess_kurt blows past 10 -> Huber."""
        rec = recommend_boosting_regression_loss(_contaminated())
        assert rec["cb"].startswith("Huber"), f"got {rec['cb']!r}"
        assert rec["lgb"] == "huber"
        assert rec["xgb"] == "reg:pseudohubererror"
        assert rec["excess_kurt"] > 10.0

    @pytest.mark.parametrize("y", [np.array([]), np.array([1.0]), np.array([np.nan, np.inf])])
    def test_safe_fallback_on_degenerate_input(self, y) -> None:
        """Empty / single / all-non-finite must return the RMSE default rather than crash."""
        rec = recommend_boosting_regression_loss(y)
        assert rec["cb"] == "RMSE"
        assert "insufficient sample" in rec["rationale"] or "Gaussian" in rec["rationale"]

    def test_constant_target_safe_fallback(self) -> None:
        """A constant target has var=0; recommendation falls back to RMSE without dividing by zero."""
        rec = recommend_boosting_regression_loss(np.full(200, 7.0))
        assert rec["cb"] == "RMSE"
        assert rec["excess_kurt"] == 0.0


class TestProductionRepro:
    """Hardened against the actual production residual moments (TVT-linres-TVT_prev, 2026-05-17).

    The composite-CB residual chart logged skew=-0.08 excess_kurt=+2.40 (Laplace-like). A synthetic Laplace with similar moments must trigger MAE -- otherwise the helper would NOT have flipped the loss for that real run.
    """

    def test_kurt_around_2_4_picks_mae(self) -> None:
        """A target with excess_kurt ~ +2.4 (between Laplace 3 and Logistic 1.2) must pick MAE (the threshold is 1.5)."""
        rng = np.random.default_rng(20260517)
        # Mix 80% Normal + 20% scaled Normal: gives excess_kurt ~ 2-3 reliably.
        n = 8000
        main = rng.standard_normal(n)
        heavy_idx = rng.choice(n, size=n // 5, replace=False)
        main[heavy_idx] *= 4.0
        rec = recommend_boosting_regression_loss(main)
        assert rec["excess_kurt"] > 1.5, f"setup failed: kurt={rec['excess_kurt']:.2f}"
        assert rec["cb"] == "MAE"
        assert rec["lgb"] == "regression_l1"
