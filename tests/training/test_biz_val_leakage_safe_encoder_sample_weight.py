"""biz_value tests for ``LeakageSafeEncoder`` weighted means.

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
  * Unweighted vs weighted per-category means must differ when y varies within a category and weights tilt
    toward one regime.
  * Under recency weighting, the recent rows' category mean dominates the encoded value; under uniform
    weighting the cross-regime average wins. Same encoder call, two weight schemas, opposite per-category
    mean.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _build_regime_split_dataset(n=500, seed=0):
    """Two categories ``a`` / ``b`` and two regimes (recent / older). In category ``a``, recent rows have
    y near 1.0 and older rows have y near 0.0. Weighted-mean for ``a`` under recency weighting should be
    near 1.0; unweighted mean should be near the cross-regime average (~0.5)."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(["a", "b"], size=n)
    # Older = first 2/3, recent = last 1/3.
    is_recent = np.zeros(n, dtype=bool)
    is_recent[-(n // 3):] = True
    y = np.where(is_recent, 1.0 + 0.05 * rng.normal(size=n), 0.0 + 0.05 * rng.normal(size=n))
    return cats, y, is_recent


def test_biz_val_leakage_safe_encoder_weighted_mean_dominates_recent_regime():
    """Recency-weighted encoder mean for category ``a`` must shift toward the recent-row y (~1.0); unweighted
    mean must sit at the cross-regime average (~0.5)."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    cats, y, is_recent = _build_regime_split_dataset()
    sw_recency = np.where(is_recent, 1.0, 0.001)
    unweighted = LeakageSafeEncoder(method="target_mean", cv=3, random_state=13).fit(cats, y)
    weighted = LeakageSafeEncoder(method="target_mean", cv=3, random_state=13).fit(cats, y, sample_weight=sw_recency)

    # Cross-regime average for category ``a`` should sit near 0.5 (it's just below since two-thirds of the rows are older).
    mean_a_uniform = unweighted._category_means["a"]
    mean_a_weighted = weighted._category_means["a"]
    assert 0.2 < mean_a_uniform < 0.6, (
        f"unweighted category-a mean should be around the cross-regime average; got {mean_a_uniform:.3f}"
    )
    assert mean_a_weighted > 0.85, (
        f"recency-weighted category-a mean should dominate to the recent-row y (~1.0); got {mean_a_weighted:.3f}"
    )
    # Shift must be at least 0.3 in absolute terms -- a real, measurable biz_value delta.
    shift = abs(mean_a_weighted - mean_a_uniform)
    assert shift > 0.3, f"recency vs uniform shift must exceed 0.3 to count as a meaningful biz_value win; got {shift:.3f}"


def test_biz_val_leakage_safe_encoder_global_prior_shifts_with_weights():
    """Global prior (used for unseen categories at transform time) must also reflect the weighting schema."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    cats, y, is_recent = _build_regime_split_dataset()
    sw_recency = np.where(is_recent, 1.0, 0.001)
    unweighted = LeakageSafeEncoder(method="target_mean", cv=3, random_state=13).fit(cats, y)
    weighted = LeakageSafeEncoder(method="target_mean", cv=3, random_state=13).fit(cats, y, sample_weight=sw_recency)
    # Unweighted prior ~ 0.33 (one-third recent at y=1, two-thirds older at y=0).
    assert 0.15 < unweighted._global_prior < 0.45
    # Recency-weighted prior should approach 1.0.
    assert weighted._global_prior > 0.9
