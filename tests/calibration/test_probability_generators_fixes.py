"""Synthetic probability generators: default flip keeps a positive signal, chunks are shuffled, rank bins are n_bins wide."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.calibration.probabilities import generate_probs_from_outcomes, generate_similar_probs_by_ranking


def test_default_flip_leaves_auc_above_half_on_iid_outcomes():
    """With default arguments the generated probabilities still rank positives above negatives."""
    outcomes = (np.random.default_rng(0).random(10000) < 0.3).astype(np.int64)
    assert roc_auc_score(outcomes, generate_probs_from_outcomes(outcomes, random_state=1)) > 0.5


def test_sorted_outcomes_do_not_leak_into_chunk_frequencies():
    """Outcomes sorted by class must not give near-perfect AUC: chunks are random subsets, not consecutive slices."""
    outcomes = np.array([0] * 5000 + [1] * 5000)
    probs = generate_probs_from_outcomes(outcomes, flip_percent=0.0, random_state=1)
    assert roc_auc_score(outcomes, probs) < 0.7


def test_rank_binning_has_exactly_n_bins_groups():
    """The top-ranked row shares a bin with others instead of sitting alone in an extra bin: a bin-wide shuffle can move it."""
    n, n_bins = 1000, 10
    predicted = np.linspace(0.0, 1.0, n)
    top_moved = False
    for seed in range(5):
        out = generate_similar_probs_by_ranking(predicted, (predicted > 0.5).astype(int), n_bins=n_bins, noise_scale=0.0, random_state=seed)
        top_moved = top_moved or out[-1] != predicted[-1]
    assert top_moved
