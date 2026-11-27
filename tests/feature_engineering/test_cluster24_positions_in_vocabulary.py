"""Cluster 24 regression: the event-to-vocabulary lookup is vectorised and agrees with the dictionary form (audit 2026-11-20, 24-F37)."""

import numpy as np
import pytest

from mlframe.feature_engineering.latent_interaction_svd import _positions_in


def _dict_reference(vocab, values):
    """The pre-fix dictionary lookup."""
    index = {v: i for i, v in enumerate(vocab)}
    return np.array([index.get(v, -1) for v in values], dtype=np.int64)


@pytest.mark.parametrize("kind", ["int", "str"])
def test_sorted_vocabulary_matches_the_dictionary_lookup(kind):
    """Present, absent, below-range and above-range values map exactly as the dict did (24-F37)."""
    rng = np.random.default_rng(0)
    if kind == "int":
        vocab = np.unique(rng.integers(0, 500, 200))
        values = rng.integers(-20, 520, 3000)
    else:
        vocab = np.unique(rng.choice([f"id{i:03d}" for i in range(300)], 150))
        values = rng.choice([f"id{i:03d}" for i in range(340)] + ["aaa", "zzz"], 3000)
    np.testing.assert_array_equal(_positions_in(vocab, values), _dict_reference(vocab, values))


def test_unsorted_or_empty_vocabulary_uses_the_dictionary_path():
    """A vocabulary that is not strictly increasing, or an empty one, still gives the dict's answer (24-F37)."""
    vocab = np.array([5, 1, 9, 3])
    values = np.array([9, 4, 5, 3, 1])
    np.testing.assert_array_equal(_positions_in(vocab, values), _dict_reference(vocab, values))
    np.testing.assert_array_equal(_positions_in(np.array([]), np.array([1, 2])), [-1, -1])
