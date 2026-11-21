"""The chunked NDCG bootstrap equals a loop-built reference with the same seed (audit 2026-11-20, finding 20-F3)."""
import numpy as np

from mlframe.reporting.charts.ltr import bootstrap_ndcg_ci


def test_chunked_bootstrap_matches_the_reference_loop():
    """Per-query NDCG values and a fixed seed: the interval must equal the one from drawing the same resamples row by row."""
    vals = np.random.default_rng(3).uniform(0.2, 0.9, 400)
    mean, lo, hi = bootstrap_ndcg_ci(vals, n_boot=200, alpha=0.1, seed=7)
    rng = np.random.default_rng(7)
    ref = np.array([vals[rng.integers(0, vals.size, size=vals.size)].mean() for _ in range(200)])
    assert abs(mean - vals.mean()) < 1e-12
    assert abs(lo - np.percentile(ref, 5.0)) < 1e-9
    assert abs(hi - np.percentile(ref, 95.0)) < 1e-9
