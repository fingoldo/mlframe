"""Unit + biz_value tests for the robust float-ensemble aggregator and its production default.

The float (regression / quantile) suite ensemble in ``_predict_main_suite`` / ``_predict_main_from_models``
routes stacked member predictions through ``combine_float_predictions`` with the flavour resolved by
``_resolve_float_ensemble_flavour`` (default ``"robust"``). These tests pin both the numeric behaviour of
the aggregator and the production default, so a future "just use np.mean" cannot silently regress the
outlier-fold robustness.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling import combine_float_predictions, robust_float_ensemble


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_robust_equals_mean_when_members_agree():
    s = np.stack([np.array([1.0, 2.0, 3.0]), np.array([1.05, 2.0, 3.0]), np.array([0.95, 2.0, 3.0])])
    np.testing.assert_allclose(robust_float_ensemble(s), s.mean(axis=0))


def test_robust_drops_per_column_outlier_member():
    s = np.stack([np.array([1.0, 2.0]), np.array([1.1, 2.0]), np.array([99.0, 2.0])])
    out = robust_float_ensemble(s)
    assert out[0] == pytest.approx(1.05, abs=1e-9)
    assert out[1] == pytest.approx(2.0, abs=1e-9)


def test_fewer_than_three_members_is_plain_mean():
    s = np.stack([np.array([1.0, 5.0]), np.array([99.0, 5.0])])
    np.testing.assert_allclose(robust_float_ensemble(s), s.mean(axis=0))


def test_flavour_mean_and_median_paths():
    s = np.stack([np.array([1.0]), np.array([2.0]), np.array([90.0])])
    assert combine_float_predictions(s, flavour="mean")[0] == pytest.approx(31.0)
    assert combine_float_predictions(s, flavour="median")[0] == pytest.approx(2.0)


def test_unknown_flavour_raises():
    with pytest.raises(ValueError, match="unknown flavour"):
        combine_float_predictions(np.zeros((3, 2)), flavour="nope")


def test_production_default_flavour_is_mean():
    # Production default holds at the legacy raw mean (optimal on clean folds); the robust MAD-gated
    # flavour is opt-in via metadata because the 3.5-MAD gate over-fires at small K (~6% clean-regime cost).
    from mlframe.training.core._predict_main_suite import _resolve_float_ensemble_flavour

    assert _resolve_float_ensemble_flavour({}) == "mean"
    assert _resolve_float_ensemble_flavour(None) == "mean"
    assert _resolve_float_ensemble_flavour({"float_ensemble_flavour": "robust"}) == "robust"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biz_val_robust_float_ensemble_beats_mean_on_outlier_folds(seed):
    """Floor: robust RMSE <= 0.55 * mean RMSE when 2 of 6 members are corrupted outlier folds.
    Measured ~0.20-0.50x across the 5 bench scenarios; raw mean has zero breakdown point."""
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.normal(size=(n, 4))
    y = x @ np.array([1.5, -2.0, 0.7, 1.1]) + rng.normal(0.0, 0.5, size=n)
    sd = float(np.std(y))
    members = [y + rng.normal(0.0, 0.3 * sd, size=n) for _ in range(4)]
    members.append(y + 4.0 * sd)            # biased outlier fold
    members.append(2.5 * y + rng.normal(0.0, 0.3 * sd, size=n))  # scale outlier fold
    stacked = np.stack(members)

    rmse_mean = _rmse(combine_float_predictions(stacked, flavour="mean"), y)
    rmse_robust = _rmse(combine_float_predictions(stacked), y)  # default flavour
    assert rmse_robust <= 0.55 * rmse_mean, f"robust {rmse_robust:.4f} vs mean {rmse_mean:.4f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biz_val_robust_float_ensemble_clean_regime_within_10pct_of_mean(seed):
    """Robust must NOT materially hurt when all members are clean: RMSE within 10% of raw mean.
    Measured ~6% penalty (MAD-gate keeps mean efficiency when nothing is flagged)."""
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.normal(size=(n, 4))
    y = x @ np.array([1.5, -2.0, 0.7, 1.1]) + rng.normal(0.0, 0.5, size=n)
    sd = float(np.std(y))
    stacked = np.stack([y + rng.normal(0.0, 0.3 * sd, size=n) for _ in range(6)])

    rmse_mean = _rmse(combine_float_predictions(stacked, flavour="mean"), y)
    rmse_robust = _rmse(combine_float_predictions(stacked), y)
    assert rmse_robust <= 1.10 * rmse_mean, f"robust {rmse_robust:.4f} vs mean {rmse_mean:.4f}"
