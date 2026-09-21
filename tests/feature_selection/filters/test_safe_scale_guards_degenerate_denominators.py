"""Scale denominators are guarded against the data's own magnitude, not padded with an absolute 1e-12.

``x / (std + 1e-12)`` compresses a legitimately tiny spread: a column whose spread is ~1e-13 is divided by ~1.1e-12 instead of by its own
spread, so the standardised values come out roughly 10x too small while still looking finite. An exactly-constant column, conversely, is
divided by 1e-12 and explodes instead of reading as "no variation".
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._safe_scale import (
    silverman_bandwidth,
    standardise,
    unit_interval,
    unit_vector,
)


def _tiny_spread(n: int = 500, scale: float = 1e-13, seed: int = 0) -> np.ndarray:
    """A column whose spread is real but ~1e-13, the regime the absolute pad corrupts."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) * scale


def test_standardise_recovers_unit_variance_on_a_tiny_spread():
    """A ~1e-13 spread standardises to unit variance; the padded form returned roughly a tenth of that."""
    x = _tiny_spread()
    z = standardise(x)
    assert abs(float(z.std()) - 1.0) < 1e-6, f"standardised spread {float(z.std())}"
    padded = (x - x.mean()) / (x.std() + 1e-12)
    assert float(padded.std()) < 0.5, "fixture precondition: the padded form must visibly compress this column"


def test_standardise_returns_zeros_for_a_constant_column():
    """No spread means no standardised variation, rather than ~1e12-magnitude noise."""
    x = np.full(256, 3.5)
    z = standardise(x)
    assert np.array_equal(z, np.zeros_like(x))
    padded = (x - x.mean()) / (x.std() + 1e-12)
    assert np.array_equal(padded, np.zeros_like(x)), "an exactly-constant column is the case the pad happens to survive"


def test_standardise_is_per_column_for_a_matrix():
    """Each column is judged on its own magnitude: a degenerate column zeroes without affecting its neighbours."""
    rng = np.random.default_rng(1)
    mat = np.column_stack([rng.normal(size=400), np.full(400, 2.0), _tiny_spread(400, seed=2)])
    z = standardise(mat)
    assert abs(float(z[:, 0].std()) - 1.0) < 1e-6
    assert np.array_equal(z[:, 1], np.zeros(400))
    assert abs(float(z[:, 2].std()) - 1.0) < 1e-6


def test_unit_interval_spans_the_full_range_on_a_tiny_spread():
    """A ~1e-13 range maps onto [0, 1]; the padded form squeezed it into a fraction of the interval."""
    c = _tiny_spread(seed=3)
    u = unit_interval(c)
    assert float(u.min()) == 0.0 and abs(float(u.max()) - 1.0) < 1e-12
    padded = (c - c.min()) / (np.ptp(c) + 1e-12)
    assert float(padded.max()) < 0.5, "fixture precondition: the padded form must visibly compress this range"
    assert np.array_equal(unit_interval(np.full(32, -7.0)), np.zeros(32))


def test_unit_vector_normalises_a_tiny_direction_and_zeroes_a_degenerate_one():
    """A small-norm direction still comes back unit length; an all-zero vector stays zero instead of exploding."""
    v = np.array([3e-13, -4e-13])
    d = unit_vector(v)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-9
    assert np.array_equal(unit_vector(np.zeros(4)), np.zeros(4))


def test_silverman_bandwidth_declines_a_column_without_spread():
    """A constant column has no density to estimate, so the helper says so instead of returning ~1e-12."""
    assert silverman_bandwidth(np.full(100, 5.0)) is None
    bw = silverman_bandwidth(_tiny_spread(seed=4))
    assert bw is not None and bw > 0.0


def test_wired_site_extra_basis_unit_map_uses_the_guard():
    """``_to_unit`` (the Jacobi/Gegenbauer domain map) spans [0, 1] on a tiny-spread column."""
    from mlframe.feature_selection.filters._extra_basis_fe_proto import _to_unit

    u = _to_unit(_tiny_spread(seed=5))
    assert abs(float(u.max()) - 1.0) < 1e-12, f"the basis domain map compressed the column: max={float(u.max())}"


def test_wired_site_rbf_bandwidth_scales_with_the_column():
    """``_rbf_fit``'s bandwidth follows Silverman on a tiny-spread column instead of collapsing onto the 1e-12 floor."""
    from mlframe.feature_selection.filters.bases import _rbf_fit

    x = _tiny_spread(seed=6)
    _, params = _rbf_fit(x)
    expected = 1.06 * float(x.std()) * (len(x) ** (-1.0 / 5.0))
    assert abs(params["bandwidth"] - expected) / expected < 1e-9, f"bandwidth {params['bandwidth']} vs Silverman {expected}"
