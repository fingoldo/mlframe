"""The serial per-node permutation test and the batched one draw the same permutations: their verdicts agree draw for draw."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_permutation_batch_njit
from mlframe.feature_selection.filters._numba_utils_cross_module import _accept_split_serial
from mlframe.feature_selection.filters.supervised_binning import _mdlp_best_split_njit


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("effect", [0.0, 0.35])
def test_serial_and_batched_verdicts_agree(seed, effect):
    """Borderline nodes (weak effect, small n) are where two different permutation streams would disagree; the two twins must not."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(60, 400))
    x = np.sort(rng.standard_normal(n))
    y = (rng.random(n) < 0.5 + effect * (x > 0)).astype(np.int64)
    _, gain, _, _, _ = _mdlp_best_split_njit(x, y, 2, 5)
    n_perm, alpha, base_seed = 99, 0.05, 1234 + seed
    serial = _accept_split_serial(x, y, 2, float(gain), 5, n_perm, base_seed, alpha)
    batched = _mdlp_permutation_batch_njit(
        x[None, :].copy(), y[None, :].copy(), np.array([n], dtype=np.int64), np.array([gain]), np.array([2], dtype=np.int64), 5, n_perm, np.array([base_seed], dtype=np.int64), alpha
    )
    assert bool(serial) == bool(batched[0])


@pytest.mark.parametrize("seed", [0, 1, 99, 123456789])
def test_shared_shuffle_is_the_honest_null_permutation(seed):
    """The labels every MDLP split test permutes are read through the permutation the honest null draws for the same seed (``seeded_permutation``)."""
    from mlframe.feature_selection.filters._numba_utils_shared import seeded_permutation
    from mlframe.feature_selection.filters._numba_utils_shuffle import shuffle_labels_into

    y = np.random.default_rng(seed).integers(0, 5, 300).astype(np.int64)
    out = np.empty_like(y)
    shuffle_labels_into(y, out, seed)
    assert np.array_equal(out, y[seeded_permutation(len(y), seed)])
    assert np.array_equal(np.sort(out), np.sort(y))
