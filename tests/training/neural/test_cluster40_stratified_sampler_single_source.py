"""Cluster 40 regression: one stratified sampler for the wrapper and the DataModule (audit 2026-11-20, finding 40-F31).

Both call sites now go through ``build_stratified_sampler``. With a seed the sample order is reproducible; without one the sampler has no
private generator. A single-class or empty label set gives no sampler.
"""

import numpy as np

from mlframe.training.neural._recurrent_data import build_stratified_sampler


def _labels() -> np.ndarray:
    """Imbalanced binary labels: 3 zeros for every 1."""
    return np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])


def test_single_class_returns_none():
    """Nothing to balance when only one class is present (40-F31)."""
    assert build_stratified_sampler(np.zeros(8, dtype=np.int64), num_samples=8) is None


def test_seeded_sampler_is_reproducible():
    """Two samplers built with the same seed draw the same order (40-F31)."""
    a = list(build_stratified_sampler(_labels(), num_samples=50, seed=7))
    b = list(build_stratified_sampler(_labels(), num_samples=50, seed=7))
    assert a == b


def test_unseeded_sampler_has_no_private_generator():
    """Without a seed the sampler uses the global RNG, as the DataModule did before the merge (40-F31)."""
    sampler = build_stratified_sampler(_labels(), num_samples=10)
    assert sampler is not None
    assert sampler.generator is None


def test_minority_class_is_drawn_more_often():
    """Inverse-frequency weights: the rare class is drawn far more than its base rate (40-F31)."""
    sampler = build_stratified_sampler(_labels(), num_samples=2000, seed=0)
    drawn = np.array(list(sampler))
    labels = _labels()
    rare_rate = float(np.mean(labels[drawn] == 1))
    assert rare_rate > 0.3
