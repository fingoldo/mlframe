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

from mlframe.feature_engineering.spatial import inverse_distance_weighted_aggregate, local_density_features


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


def test_idw_still_weights_by_distance_at_fine_coordinate_scales():
    """The sibling of the density bug, found by the check written to catch that one.

    `inverse_distance_weighted_aggregate` guarded its weights with `1.0 / (d**power + 1e-12)`. `d**power`
    falls off geometrically, so at power=4 the pad already dominates by d=1e-3, and it dominates UNEVENLY
    across a row: near neighbours whose d**power sits under the pad collapse onto one weight while farther
    ones keep theirs. Measured on distances (1, 2, 5, 9) * 1e-4 at power=4, the old weights came out
    [0.282, 0.282, 0.266, 0.170] -- an almost unweighted average -- against the true
    [0.940, 0.059, 0.0015, 0.0001]. The nearest neighbour's influence had been given away entirely.
    """
    # Two reference points, one much nearer the query than the other, at a fine coordinate scale.
    scale = 1e-4
    # A fourth reference: the lookup requires strictly more refs than k.
    ref = np.array([[0.0, 0.0], [3.0 * scale, 0.0], [9.0 * scale, 0.0], [50.0 * scale, 0.0]])
    labels = np.array([10.0, 0.0, 0.0, 0.0])
    q = np.array([[1.0 * scale, 0.0]])

    out = inverse_distance_weighted_aggregate(q, ref, labels, k=3, power=4.0)
    d = np.array([1.0, 2.0, 8.0]) * scale
    w = (d.min() / d) ** 4.0
    expected = float((labels[:3] * (w / w.sum())).sum())  # k=3: only the three nearest are aggregated
    assert float(out["idw"][0]) == pytest.approx(expected, rel=1e-9), f"{out['idw'][0]} against the unpadded {expected}"
    # The point of the metric: the nearest neighbour must dominate, not be averaged away.
    assert float(out["idw"][0]) > 9.0, "the nearest label barely counted; the weighting has flattened"


def test_idw_is_unchanged_on_well_conditioned_distances():
    """The fix must move only the regime the pad was corrupting."""
    ref = np.array([[0.0, 0.0], [3.0, 0.0], [9.0, 0.0], [50.0, 0.0]])
    labels = np.array([10.0, 0.0, 0.0, 0.0])
    q = np.array([[1.0, 0.0]])
    out = inverse_distance_weighted_aggregate(q, ref, labels, k=3, power=2.0)
    d = np.array([1.0, 2.0, 8.0])
    w = 1.0 / d**2.0
    assert float(out["idw"][0]) == pytest.approx(float((labels[:3] * (w / w.sum())).sum()), rel=1e-9)


def test_idw_gives_a_coincident_reference_point_the_whole_vote():
    """A zero distance is the limit of 1/d^p, not a value the guard should cap."""
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    labels = np.array([7.0, 0.0, 0.0, 0.0])
    out = inverse_distance_weighted_aggregate(np.array([[0.0, 0.0]]), ref, labels, k=3, power=2.0)
    assert float(out["idw"][0]) == pytest.approx(7.0)
