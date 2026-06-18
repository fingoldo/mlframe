"""Biz-value: ``cv_selector_mode`` (+ alpha / confidence / quantile) picks the STABLE candidate.

``forward_stepwise`` ranks multi-base candidates by ``aggregate_fold_scores(fold_rmses, mode=cv_selector_mode, ...)``.
The default ``"mean"`` is bit-identical to ``np.mean`` and silently rewards a LUCKY candidate whose mean wins by less
than the per-fold std. The robust modes (``mean_minus_std`` / ``median_minus_mad`` / ``t_lcb`` / ``quantile``) add a
dispersion penalty so a stable-mediocre candidate beats an unstable-lucky one.

The win is constructed at the aggregator level (the single decision point the config field routes into): a candidate
whose per-fold RMSE is low-on-average but HIGH-variance vs a candidate that is slightly-higher-on-average but tight.
``mean`` selects the volatile one; every robust mode selects the stable one. A regression that drops the penalty (mode
ignored / routed to mean) flips the selection back and FAILS the test.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._cv_aggregation import aggregate_fold_scores


# Per-fold RMSE (direction="min": lower is better).
# unstable_lucky: lower mean but broad spread across ALL folds (high std AND high MAD,
# so even the median/MAD-robust mode penalises it -- not a single outlier fold).
# stable_mediocre: slightly higher mean but tight spread (low std, low MAD).
UNSTABLE_LUCKY = [0.40, 0.55, 1.10, 1.35]   # mean 0.850, wide spread every fold
STABLE_MEDIOCRE = [0.94, 0.96, 0.98, 1.00]  # mean 0.970, tight


def _argmin(score_fn):
    return min(
        ("unstable", "stable"),
        key=lambda name: score_fn(UNSTABLE_LUCKY if name == "unstable" else STABLE_MEDIOCRE),
    )


def test_biz_val_cv_selector_mean_picks_unstable_lucky():
    """Baseline: plain ``mean`` rewards the low-mean high-variance candidate -- the failure mode."""
    pick = _argmin(lambda fs: aggregate_fold_scores(fs, mode="mean", direction="min"))
    assert pick == "unstable", "mean must select the lower-mean candidate (the bug cv_selector_mode fixes)"


@pytest.mark.parametrize("mode", ["mean_minus_std", "median_minus_mad", "t_lcb", "quantile"])
def test_biz_val_cv_selector_robust_modes_pick_stable(mode):
    """Every robust mode penalises dispersion enough to flip the pick to the stable candidate."""
    pick = _argmin(lambda fs: aggregate_fold_scores(
        fs, mode=mode, direction="min", alpha=1.0, confidence=0.9, quantile_level=0.9,
    ))
    assert pick == "stable", f"cv_selector_mode={mode!r} must prefer the stable-mediocre candidate over the lucky one"


def test_biz_val_cv_selector_alpha_strengthens_penalty():
    """Larger ``cv_selector_alpha`` raises the dispersion penalty on the volatile candidate."""
    low = aggregate_fold_scores(UNSTABLE_LUCKY, mode="mean_minus_std", direction="min", alpha=0.5)
    high = aggregate_fold_scores(UNSTABLE_LUCKY, mode="mean_minus_std", direction="min", alpha=2.0)
    assert high > low + 0.30, "higher alpha must materially increase the penalised score of a high-variance candidate"


def test_biz_val_cv_selector_quantile_level_controls_conservatism():
    """Higher ``cv_selector_quantile_level`` reads a worse (higher) RMSE tail for direction=min."""
    q50 = aggregate_fold_scores(UNSTABLE_LUCKY, mode="quantile", direction="min", quantile_level=0.5)
    q90 = aggregate_fold_scores(UNSTABLE_LUCKY, mode="quantile", direction="min", quantile_level=0.9)
    assert q90 > q50 + 0.30, "a higher quantile_level must surface the disastrous-fold tail of the unstable candidate"


def test_biz_val_cv_selector_confidence_widens_t_lcb():
    """Higher ``cv_selector_confidence`` widens the one-sided t penalty for the volatile candidate."""
    c60 = aggregate_fold_scores(UNSTABLE_LUCKY, mode="t_lcb", direction="min", confidence=0.6)
    c95 = aggregate_fold_scores(UNSTABLE_LUCKY, mode="t_lcb", direction="min", confidence=0.95)
    assert c95 > c60, "higher confidence must widen the t-LCB interval (larger penalised score for min direction)"
