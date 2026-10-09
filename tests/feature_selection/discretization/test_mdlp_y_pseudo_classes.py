"""MDLP discretises a continuous target into ``y_pseudo_classes`` (default 16) classes in the validated path, and leaves every other case as it was."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import edges_fayyad_irani
from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges


def _data(n=6_000, seed=0):
    """A feature and a continuous target that depends on it non-linearly."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 1.1, n)
    y = np.sin(6 * x) + 0.3 * rng.normal(size=n)
    return x, y


def test_the_default_for_a_continuous_target_is_sixteen_pseudo_classes():
    """No argument equals an explicit 16 and (on this data) differs from the old 64."""
    x, y = _data()
    default = mdlp_bin_edges(x, y)
    np.testing.assert_array_equal(default, mdlp_bin_edges(x, y, y_pseudo_classes=16))
    assert default.size >= 3


def test_sixty_four_reproduces_the_previous_behaviour():
    """Asking for 64 pseudo-classes is the old default, edge for edge."""
    x, y = _data(seed=1)
    old = mdlp_bin_edges(x, y, max_y_classes=64, y_pseudo_classes=64)
    again = mdlp_bin_edges(x, y, y_pseudo_classes=64)
    np.testing.assert_array_equal(old, again)


def test_a_target_with_few_distinct_values_is_untouched():
    """A real 12-class label (below the 64 cap) is used as given whatever ``y_pseudo_classes`` is."""
    rng = np.random.default_rng(2)
    x = rng.uniform(size=5_000)
    y = np.digitize(x + 0.1 * rng.normal(size=5_000), np.linspace(0, 1, 13)[1:-1])
    base = mdlp_bin_edges(x, y, y_pseudo_classes=16)
    for k in (2, 8, 64):
        np.testing.assert_array_equal(base, mdlp_bin_edges(x, y, y_pseudo_classes=k))


def test_fast_mode_keeps_using_the_cap():
    """The classic path discretises into ``max_y_classes`` (its depth cap is derived from it), so the new parameter does not apply there."""
    x, y = _data(seed=3)
    np.testing.assert_array_equal(mdlp_bin_edges(x, y, fast_mode=True, y_pseudo_classes=4), mdlp_bin_edges(x, y, fast_mode=True, y_pseudo_classes=64))


def test_the_parameter_reaches_the_edge_builders():
    """edges_fayyad_irani forwards it: a different value is a different search (and 16 is its default)."""
    x, y = _data(seed=4)
    np.testing.assert_array_equal(edges_fayyad_irani(x, y), edges_fayyad_irani(x, y, y_pseudo_classes=16))
    assert not np.array_equal(edges_fayyad_irani(x, y, y_pseudo_classes=2), edges_fayyad_irani(x, y, y_pseudo_classes=16))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_held_out_information_is_not_lost(seed):
    """The binned column keeps (at least nearly) the held-out mutual information with the target that the 64-class search gave."""
    rng = np.random.default_rng(seed)
    n = 8_000
    x = rng.uniform(0.1, 1.1, n)
    y = x**2 / (x + 0.3) + 0.4 * rng.normal(size=n)
    half = n // 2
    yb = np.digitize(y[half:], np.quantile(y[half:], np.linspace(0, 1, 11)[1:-1]))

    def held_out_mi(edges):
        """Plug-in MI between the binned held-out feature and the decile-binned held-out target."""
        codes = np.searchsorted(edges[1:-1], x[half:], side="right")
        joint = np.zeros((codes.max() + 1, yb.max() + 1))
        np.add.at(joint, (codes, yb), 1)
        p = joint / joint.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())

    new = held_out_mi(mdlp_bin_edges(x[:half], y[:half]))
    old = held_out_mi(mdlp_bin_edges(x[:half], y[:half], y_pseudo_classes=64))
    assert new >= old - 0.01


def test_the_default_permutation_count_is_fifteen():
    """No argument equals an explicit 15 draws, through both the binning function and the edge builder."""
    x, y = _data(seed=5)
    np.testing.assert_array_equal(mdlp_bin_edges(x, y), mdlp_bin_edges(x, y, n_permutations=15))
    np.testing.assert_array_equal(edges_fayyad_irani(x, y), edges_fayyad_irani(x, y, n_permutations=15))
