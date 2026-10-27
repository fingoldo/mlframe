"""The two-class split search returns the general search's split on the same sorted data."""
import numpy as np
import pytest

from mlframe.feature_selection.filters.supervised_binning import _mdlp_best_split_binary_njit, _mdlp_best_split_inc_njit, _mdlp_best_split_njit


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("shape", ["continuous", "tied", "separated", "rare_class"])
def test_binary_kernel_matches_the_general_kernel(seed, shape):
    """Same best index and counts, gain equal to 1e-12, on continuous and tied x, perfectly separated labels and a 5 percent class."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(120, 6000))
    x = np.sort(rng.normal(size=n))
    if shape == "tied":
        x = np.round(x, 1)
    y = (rng.random(n) < (0.05 if shape == "rare_class" else 0.4)).astype(np.int64)
    if shape == "separated":
        y = (x > np.quantile(x, 0.37)).astype(np.int64)
    m = int(rng.integers(5, 40))
    general = _mdlp_best_split_inc_njit(x, y, 2, m)
    binary = _mdlp_best_split_binary_njit(x, y, m)
    assert binary[0] == general[0] and binary[3:] == general[3:]
    if general[0] >= 0:
        assert binary[1] == pytest.approx(general[1], abs=1e-12)
    # and the dispatching kernel takes the binary path for two classes
    assert _mdlp_best_split_njit(x, y, 2, m)[0] == binary[0]


def test_single_class_and_too_small_nodes_return_no_split():
    """A one-class node has zero entropy and a node below twice the minimum size has no admissible split."""
    x = np.sort(np.random.default_rng(0).normal(size=200))
    assert _mdlp_best_split_binary_njit(x, np.zeros(200, dtype=np.int64), 5)[0] == -1
    assert _mdlp_best_split_binary_njit(x[:8], (np.arange(8) % 2).astype(np.int64), 5)[0] == -1
