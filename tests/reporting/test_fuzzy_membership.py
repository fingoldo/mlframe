"""Tests for the fuzzy-partition membership chart (charts/fuzzy_membership.py).

Covers: shapes + n_partitions honored (unit), and biz_value -- a triangular (Ruspini) partition is a partition-of-unity
(memberships sum to ~1 across the grid) and each set peaks near ~1 at its centre, while a gaussian partition is dense
(every set active everywhere) versus the triangular's sparse "only two sets active" structure.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.fuzzy_membership import (
    compose_fuzzy_membership_figure,
    fuzzy_membership_curves,
    fuzzy_membership_panel,
)
from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def _feature(n=5000, seed=0):
    """Helper: Feature."""
    return np.random.default_rng(seed).normal(size=n)


# ----------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------


def test_curves_shape_and_n_partitions_honored():
    """Curves shape and n partitions honored."""
    grid_x, memb = fuzzy_membership_curves(_feature(), n_partitions=5, grid=200)
    assert grid_x.shape == (200,)
    assert memb.shape == (5, 200)


def test_curves_respects_grid_and_partition_counts():
    """Curves respects grid and partition counts."""
    for n_p in (3, 4, 7):
        grid_x, memb = fuzzy_membership_curves(_feature(), n_partitions=n_p, grid=128)
        assert grid_x.shape == (128,)
        assert memb.shape == (n_p, 128)


def test_panel_and_figure_shapes():
    """Panel and figure shapes."""
    panel = fuzzy_membership_panel(_feature(), n_partitions=5)
    assert isinstance(panel, LinePanelSpec)
    assert len(panel.y) == 5 and len(panel.series_labels) == 5
    fig = compose_fuzzy_membership_figure(_feature())
    assert isinstance(fig, FigureSpec)
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 1 and isinstance(panels[0], LinePanelSpec)


# ----------------------------------------------------------------------------
# biz_value
# ----------------------------------------------------------------------------


def test_biz_triangular_is_partition_of_unity():
    """A triangular (Ruspini) partition sums to 1 at every grid point. Measured max|sum-1| ~0 (< 1e-6).
    A regression breaking the tent overlap (e.g. non-adjacent sets active) inflates the deviation."""
    _, memb = fuzzy_membership_curves(_feature(), n_partitions=5, kind="triangular", grid=200)
    col_sums = memb.sum(axis=0)
    assert np.max(np.abs(col_sums - 1.0)) < 1e-6


def test_biz_triangular_each_set_peaks_near_one_at_its_centre():
    """Each triangular set reaches membership ~1 near its own centre. Floor 0.9 (grid discretisation keeps the
    closest grid point just under an exact 1.0 at interior centres)."""
    _, memb = fuzzy_membership_curves(_feature(), n_partitions=5, kind="triangular", grid=400)
    per_set_peak = memb.max(axis=1)
    assert np.all(per_set_peak >= 0.9), per_set_peak


def test_biz_triangular_sparse_vs_gaussian_dense():
    """Triangular activates at most 2 sets at any point (sparse Ruspini); the normalized gaussian is dense (every
    set carries positive mass everywhere). This separation is the interpretability contract distinguishing the two
    families -- a regression collapsing gaussian to a hard/sparse form would drop the active count below all-active."""
    _, tri = fuzzy_membership_curves(_feature(), n_partitions=5, kind="triangular", grid=200)
    _, gau = fuzzy_membership_curves(_feature(), n_partitions=5, kind="gaussian", grid=200)
    tri_active = (tri > 1e-9).sum(axis=0)
    gau_active = (gau > 1e-9).sum(axis=0)
    assert tri_active.max() <= 2
    assert gau_active.min() == 5  # gaussian: all sets active at every grid point (infinite support)
