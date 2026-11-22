"""Cluster 41 regression: sorted-slice group statistics match the per-group-mask reference (audit 2026-11-20, finding 41-F5)."""

import math

import numpy as np
import pytest

from mlframe.training.targets._target_distribution_analyzer_modes import _within_between_group_variance_ratio
from mlframe.training.targets._target_distribution_analyzer_stats import _lag1_autocorr_grouped


def _ratio_reference(y, g):
    """Original per-group-mask implementation, kept as the reference."""
    uniq = np.unique(g)
    if len(uniq) < 2:
        return float("nan")
    means = np.zeros(len(uniq))
    sq = 0.0
    n = 0
    for k, u in enumerate(uniq):
        m = g == u
        yk = y[m]
        means[k] = float(np.mean(yk))
        sq += float(np.sum((yk - means[k]) ** 2))
        n += int(yk.size)
    within = math.sqrt(sq / max(n, 1))
    between = float(np.std(means))
    return float("inf") if between <= 0 else within / between


def _autocorr_reference(y, g, min_group_size=4):
    """Original per-group-mask autocorrelation aggregation, kept as the reference (values only)."""
    from mlframe.training.targets._target_distribution_analyzer_stats import _lag1_autocorr

    z = []
    skipped = 0
    for u in np.unique(g):
        m = g == u
        if int(m.sum()) < min_group_size:
            skipped += 1
            continue
        ar = _lag1_autocorr(y[m])
        if not math.isfinite(ar):
            continue
        z.append(math.atanh(max(min(ar, 0.999999), -0.999999)))
    return z, skipped


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_within_between_ratio_matches_reference(seed):
    """Vectorised group loop gives the same ratio as the per-group mask loop (41-F5)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 40, size=2000)
    y = rng.normal(size=2000) + g * 0.1
    assert _within_between_group_variance_ratio(y, g) == pytest.approx(_ratio_reference(y, g), rel=1e-12)


def test_autocorr_grouped_skips_small_groups_and_is_finite():
    """Groups below min_group_size are counted as skipped; the aggregate is finite (41-F5)."""
    rng = np.random.default_rng(7)
    g = np.concatenate([np.zeros(50, int), np.ones(50, int), np.full(2, 2)])
    y = rng.normal(size=g.size)
    ar, skipped = _lag1_autocorr_grouped(y, g, min_group_size=4)
    assert skipped == 1
    assert math.isfinite(ar)
