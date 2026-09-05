"""`local_density` must not saturate on ordinary high-dimensional coordinates.

The density is `k / r^d` where `r` is the distance to the k-th neighbour. Guarding the division with an
additive `+ 1e-12` puts an ABSOLUTE floor under a denominator whose scale is arbitrary: `r^d` falls off
geometrically in `d`, so at `d=8, r=0.01` the true density of 1e17 came back as 9.999e12 -- a 99.99% error,
with every dense row saturating onto the same value and the feature losing all variation precisely where it
is most informative. Clamping to the smallest positive normal instead engages only on a genuine underflow.

The degenerate case the guard exists for is still covered: duplicate reference points, and query rows whose
coordinates were non-finite and were mapped to the origin, both give `r == 0`, and must come back finite
rather than as `inf` or a division warning. (Non-finite REFERENCE rows never reach the division -- they are
dropped before the tree is built.)
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.feature_engineering.spatial import local_density_features


def _grid(d: int, spacing: float, n: int = 64):
    """`n` points on a line at a fixed spacing, embedded in `d` dimensions.

    The function takes `d` from the coordinate width, so the embedding IS how the dimension is set.

    The k-th neighbour distance is then a known multiple of `spacing`, so the expected density is exact
    rather than approximate.
    """
    coords = np.zeros((n, d))
    coords[:, 0] = np.arange(n) * spacing
    return coords


def test_a_small_kth_distance_in_eight_dimensions_is_not_saturated():
    """d=8 with spacing 0.01 is the measured 99.99%-error case."""
    d, spacing, k = 8, 0.01, 4
    coords = _grid(d, spacing)
    out = local_density_features(coords, coords, k=k)
    r = out["dist_to_kth"]
    expected = float(k) / r**d
    got = out["local_density"]
    interior = slice(k + 1, -(k + 1))
    assert np.allclose(got[interior], expected[interior], rtol=1e-9), f"density saturated: {got[interior][:3]} against {expected[interior][:3]}"
    assert got[interior].max() > 1e14, "the fixture no longer reaches the regime the additive floor destroyed"


@pytest.mark.parametrize("d", [2, 4, 12])
def test_the_density_is_the_exact_reciprocal_across_dimensions(d: int):
    """No floor may perturb a denominator the format can represent."""
    coords = _grid(d, 0.05)
    out = local_density_features(coords, coords, k=3)
    r = out["dist_to_kth"]
    interior = slice(5, -5)
    assert np.allclose(out["local_density"][interior], (3.0 / r**d)[interior], rtol=1e-9)


def test_duplicate_points_give_a_finite_density():
    """The guard's real job: a zero k-th distance must not divide by zero or return inf."""
    coords = np.zeros((32, 3))
    out = local_density_features(coords, coords, k=4)
    assert np.all(np.isfinite(out["local_density"])), "duplicate reference points produced a non-finite density"
    assert np.all(out["local_density"] > 0)


def test_non_finite_query_coordinates_still_give_a_finite_density():
    """Non-finite QUERY rows are mapped to the origin before the lookup; non-finite refs are dropped instead.

    A query row at the origin sits on top of the reference points placed there, so its k-th distance is zero
    -- the same degeneracy as a duplicate, arrived at through the path a caller actually hits.
    """
    ref = np.zeros((32, 3))
    q = np.full((4, 3), np.nan)
    out = local_density_features(q, ref, k=4)
    assert np.all(np.isfinite(out["local_density"])), "an origin-mapped query row produced a non-finite density"
