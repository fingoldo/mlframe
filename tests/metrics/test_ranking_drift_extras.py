"""Tests for mlframe.metrics._ranking_extras and mlframe.metrics._drift."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import (
    dcg_at_k, expected_reciprocal_rank, hit_at_k, precision_at_k,
    population_stability_index, kl_divergence, js_divergence,
    wasserstein_1d, ks_distribution_distance,
)


# ----- Ranking -----


def test_dcg_zero_when_no_relevance():
    y = np.zeros(20, dtype=np.float64)
    s = np.random.default_rng(0).uniform(size=20)
    g = np.zeros(20, dtype=np.int64)  # one group
    assert dcg_at_k(y, s, g, k=10) == pytest.approx(0.0)


def test_dcg_higher_when_relevance_at_top():
    """Same labels, but better ranking -> higher DCG."""
    g = np.zeros(10, dtype=np.int64)
    y = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_good = -np.arange(10, dtype=np.float64)   # rel descending
    s_bad = np.arange(10, dtype=np.float64)     # rel ascending (worst)
    assert dcg_at_k(y, s_good, g, k=10) > dcg_at_k(y, s_bad, g, k=10)


def test_err_in_unit_interval():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(20), 10)
    y = rng.integers(0, 5, size=200).astype(np.float64)
    s = rng.uniform(size=200)
    val = expected_reciprocal_rank(y, s, g, k=10)
    assert 0.0 <= val <= 1.0


def test_hit_at_k_perfect_score():
    """If top-1 is always relevant -> Hit@1 == 1.0."""
    g = np.repeat(np.arange(10), 5)
    y = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], 10)
    s = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], 10)
    assert hit_at_k(y, s, g, k=1) == pytest.approx(1.0)


def test_precision_at_k_matches_definition():
    """top-5: 2 relevant -> precision@5 = 0.4."""
    g = np.zeros(10, dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = -np.arange(10, dtype=np.float64)
    val = precision_at_k(y, s, g, k=5)
    assert val == pytest.approx(0.4)


def test_ranking_metric_rejects_bad_k():
    g = np.zeros(5, dtype=np.int64)
    y = np.ones(5)
    s = np.ones(5)
    for fn in (dcg_at_k, expected_reciprocal_rank, hit_at_k, precision_at_k):
        with pytest.raises(ValueError):
            fn(y, s, g, k=0)


# ----- Drift -----


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(5000)
    # Same data -> empirical PSI should be at exact zero modulo binning
    assert population_stability_index(a, a) == pytest.approx(0.0, abs=1e-12)


def test_psi_increases_with_shift():
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(5000)
    small_shift = rng.standard_normal(5000) + 0.5
    big_shift = rng.standard_normal(5000) + 2.0
    psi_small = population_stability_index(ref, small_shift)
    psi_big = population_stability_index(ref, big_shift)
    assert psi_big > psi_small > 0


def test_kl_zero_for_identical():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(2000)
    assert kl_divergence(a, a) == pytest.approx(0.0, abs=1e-10)


def test_kl_asymmetric():
    """KL(P||Q) != KL(Q||P) in general."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal(3000)
    b = rng.standard_normal(3000) + 0.5
    kl_ab = kl_divergence(a, b)
    kl_ba = kl_divergence(b, a)
    # Allow small fp ratio; the asymmetry should be measurable.
    assert abs(kl_ab - kl_ba) > 1e-3


def test_js_symmetric_and_bounded():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000) + 0.5
    js_ab = js_divergence(a, b)
    js_ba = js_divergence(b, a)
    assert js_ab == pytest.approx(js_ba, abs=1e-10)
    assert 0.0 <= js_ab <= np.log(2.0) + 1e-9


def test_js_pre_binned_matches_scipy():
    """Pre-binned probability vectors -> JS = scipy.spatial.distance.jensenshannon^2."""
    from scipy.spatial.distance import jensenshannon
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.4, 0.1, 0.5])
    # scipy returns the JS DISTANCE (sqrt of divergence in bit-units);
    # we return the divergence in nat-units (natural log). Convert:
    js_scipy = jensenshannon(p, q, base=np.e) ** 2
    js_ours = js_divergence(q, p, pre_binned=True)
    assert js_ours == pytest.approx(js_scipy, abs=1e-10)


def test_wasserstein_matches_scipy():
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(7)
    a = rng.standard_normal(500)
    b = rng.standard_normal(500) + 1.0
    expected = wasserstein_distance(a, b)
    assert wasserstein_1d(a, b) == pytest.approx(expected, abs=1e-10)


def test_ks_distribution_distance_matches_scipy():
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(8)
    a = rng.standard_normal(500)
    b = rng.standard_normal(500) + 0.3
    expected = ks_2samp(a, b).statistic
    assert ks_distribution_distance(a, b) == pytest.approx(expected, abs=1e-12)


def _wasserstein_1d_numpy_reference(reference, target):
    """Pre-fix numpy implementation: concat+sort merged support + two searchsorted scans."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    all_values = np.concatenate((a, b)); all_values.sort(kind="quicksort")
    deltas = np.diff(all_values)
    cdf_a = np.searchsorted(np.sort(a), all_values[:-1], side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), all_values[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


def _ks_distance_numpy_reference(reference, target):
    a = np.asarray(reference, dtype=np.float64); b = np.asarray(target, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    a_s = np.sort(a); b_s = np.sort(b)
    all_values = np.concatenate((a_s, b_s)); all_values.sort()
    cdf_a = np.searchsorted(a_s, all_values, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, all_values, side="right") / b_s.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def test_fused_drift_kernels_bit_identical_to_numpy_reference():
    """The fused single-pass njit merge (replacing concat+sort+2x searchsorted) must reproduce the
    numpy reference bit-for-bit on ties/discrete data and to FP-order tolerance on continuous data."""
    from mlframe.metrics._drift import _wasserstein_1d_fused, _ks_distance_fused

    rng = np.random.default_rng(123)
    # Tied / discrete: exact equality required (positional ties handled identically to searchsorted-right).
    a = rng.integers(0, 5, 400).astype(np.float64); b = rng.integers(0, 7, 400).astype(np.float64)
    assert wasserstein_1d(a, b) == _wasserstein_1d_numpy_reference(a, b)
    assert ks_distribution_distance(a, b) == _ks_distance_numpy_reference(a, b)
    # Continuous, larger n: FP reduction-order tolerance.
    a = rng.random(20000); b = rng.random(20000) + 0.15
    assert wasserstein_1d(a, b) == pytest.approx(_wasserstein_1d_numpy_reference(a, b), abs=1e-12)
    assert ks_distribution_distance(a, b) == _ks_distance_numpy_reference(a, b)


def test_drift_metrics_empty_input():
    """All five drift metrics return NaN on empty input."""
    empty = np.array([])
    a = np.array([1.0, 2.0])
    assert np.isnan(population_stability_index(empty, a))
    assert np.isnan(kl_divergence(empty, a))
    assert np.isnan(js_divergence(empty, a))
    assert np.isnan(wasserstein_1d(empty, a))
    assert np.isnan(ks_distribution_distance(empty, a))


def test_ranking_extras_stable_tiebreak_matches_input_order():
    """DCG / ERR break tied scores by stable input order, matching the core ranking.py NDCG/MAP/MRR convention.

    The score vector has a tied 0.0 block; the sole relevant doc (row 7) must land at its stable-order rank (numpy's unstable quicksort would place it
    elsewhere, but the @njit kernels sort stably). Invariant test pinning the stable-order DCG@k / ERR values so a future change of the kernel sort
    (or numba's internal default) is caught.
    """
    score = np.array([2., 1., 1., 0., 0., 0., 0., 0., 0., 2., 1., 2.], dtype=np.float64)
    rel = np.zeros_like(score)
    rel[7] = 3.0  # sole relevant doc, sits inside the tied 0.0 block
    g = np.zeros(score.shape[0], dtype=np.int64)

    stable_order = np.argsort(-score, kind="mergesort")

    dcg_ref = sum(((2.0 ** rel[stable_order[i]]) - 1.0) / np.log2(i + 2.0) for i in range(score.shape[0]))
    assert dcg_at_k(rel, score, g, k=12) == pytest.approx(dcg_ref)

    # max_grade is now a FIXED comparability scale, not the per-call rel.max(); pass it explicitly
    # so this invariant pins the stable tie-break, not the (corrected) default scale.
    mg = 4.0
    err_ref = 0.0
    p_remain = 1.0
    for i in range(score.shape[0]):
        R = ((2.0 ** rel[stable_order[i]]) - 1.0) / (2.0 ** mg)
        err_ref += p_remain * R / (i + 1.0)
        p_remain *= (1.0 - R)
    assert expected_reciprocal_rank(rel, score, g, k=12, max_grade=mg) == pytest.approx(err_ref)
