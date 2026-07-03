"""Regression: the analytic-gate occupied-y-class count in
``_dispatch_batch_mi_with_noise_gate`` is derived from ``np.count_nonzero(freqs_y)``
(O(nbins)) instead of ``np.unique(classes_y).size`` (O(n log n)). The two must be
identical for every valid ``freqs_y`` = bincount(classes_y)/total, including
distributions with absent (gap) labels -- otherwise the analytic-null applicability
check would see a wrong df and change which candidates the gate keeps.
"""
from __future__ import annotations

import numpy as np
import pytest


def _freqs_y_from(classes_y: np.ndarray) -> np.ndarray:
    counts = np.bincount(classes_y.astype(np.int64))
    total = counts.sum()
    return counts.astype(np.float64) / float(total)


@pytest.mark.parametrize("labels", [
    [0, 1],                         # binary, both present
    [0, 0, 0, 1],                   # binary, imbalanced
    [0, 1, 2, 3, 4, 5, 6],          # dense multiclass
    [0, 2, 4, 6],                   # gaps: labels 1,3,5 absent -> freqs_y has interior zeros
    [3, 3, 3],                      # single occupied high label -> leading zeros
])
def test_count_nonzero_freqs_equals_unique_size(labels):
    classes_y = np.array(labels, dtype=np.int64)
    freqs_y = _freqs_y_from(classes_y)
    assert int(np.count_nonzero(freqs_y)) == int(np.unique(classes_y).size)


def test_matches_on_random_gapped_distributions():
    rng = np.random.default_rng(123)
    for _ in range(50):
        ncls = int(rng.integers(2, 40))
        cy = rng.integers(0, ncls, size=5000).astype(np.int64)
        # drop a random label to force a gap
        drop = int(rng.integers(0, ncls))
        cy = cy[cy != drop]
        if cy.size == 0:
            continue
        freqs_y = _freqs_y_from(cy)
        assert int(np.count_nonzero(freqs_y)) == int(np.unique(cy).size)
