"""Audit 2026-11-20 finding 04-F14: bayes_auc's tie-group pass is vectorised and gives the loop's answer exactly.

The reference is the original per-row loop over tie groups, kept here so the vectorised version is checked against it
on inputs with heavy ties, a single tie group and no ties at all.
"""

import numpy as np

from mlframe.data.datasets._target import bayes_auc


def _loop_reference(probability):
    """The pre-vectorisation implementation, used only as the oracle."""
    p = np.asarray(probability, dtype=np.float64).ravel()
    if p.size < 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    q_sorted = 1.0 - p_sorted
    cumulative_q = np.concatenate(([0.0], np.cumsum(q_sorted)))
    less_mass = np.empty_like(p_sorted)
    tie_mass = np.empty_like(p_sorted)
    start = 0
    for end in range(1, p_sorted.size + 1):
        if end == p_sorted.size or p_sorted[end] != p_sorted[start]:
            group_q = cumulative_q[end] - cumulative_q[start]
            less_mass[start:end] = cumulative_q[start]
            tie_mass[start:end] = 0.5 * (group_q - q_sorted[start:end])
            start = end
    numerator = float(np.sum(p_sorted * (less_mass + tie_mass)))
    denominator = float(np.sum(p_sorted) * np.sum(q_sorted) - np.sum(p_sorted * q_sorted))
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def test_matches_loop_with_heavy_ties():
    """Probabilities drawn from a handful of values, so most rows share a tie group."""
    rng = np.random.default_rng(0)
    p = rng.choice([0.1, 0.4, 0.4, 0.9], size=500)
    assert abs(bayes_auc(p) - _loop_reference(p)) < 1e-12


def test_matches_loop_on_continuous_probabilities():
    """No ties at all: every group has one row."""
    p = np.random.default_rng(1).uniform(0.01, 0.99, size=400)
    assert abs(bayes_auc(p) - _loop_reference(p)) < 1e-12


def test_all_identical_gives_one_half():
    """Every pair a tie: the ceiling is one half, the case the tie handling exists for."""
    assert abs(bayes_auc(np.full(50, 0.3)) - 0.5) < 1e-12


def test_degenerate_inputs_unchanged():
    """Fewer than two rows is NaN, as before."""
    assert np.isnan(bayes_auc(np.array([0.5])))
