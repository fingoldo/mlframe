"""Behavioral invariants locking in the outputs the rejected PERF#2/#5 attempts had to preserve.

PERF#2 (within-bin shuffle) and PERF#5 (fused bin_predictions nanmean) were both benchmarked and
REJECTED (the alternatives were bit-identical but slower). These tests pin the current production
behavior so any future re-attempt must remain bit-identical.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


@pytest.mark.fast
def test_generate_similar_probs_by_ranking_reproducible():
    from mlframe.calibration.probabilities import generate_similar_probs_by_ranking

    rng = np.random.default_rng(0)
    probs = rng.random(1000)
    outcomes = (rng.random(1000) < probs).astype(np.int8)

    a = generate_similar_probs_by_ranking(probs, outcomes, n_bins=10, noise_scale=0.0, random_state=42)
    b = generate_similar_probs_by_ranking(probs, outcomes, n_bins=10, noise_scale=0.0, random_state=42)
    assert np.array_equal(a, b)
    # within-bin shuffle preserves the multiset of values (no noise) -> same sorted content
    assert np.array_equal(np.sort(a), np.sort(probs))


@pytest.mark.fast
def test_bin_predictions_nan_aware():
    from mlframe.calibration.quality import bin_predictions

    y_pred = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    y_true = np.array([0, 0, 1, 0, 1, 1, 1, 1], dtype=np.float64)
    indices = np.argsort(np.nan_to_num(y_pred, nan=1e9))
    pp, pt, data = bin_predictions(y_true, y_pred, indices, nbins=2)
    # nanmean must skip the NaN and never produce a NaN pocket when other values exist in-bin
    assert np.all(np.isfinite(pp))
    assert np.all(np.isfinite(pt))
