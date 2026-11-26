"""Cluster 21 regression: low-evidence flags follow a scatter's downsample selection (audit 2026-11-20, 21-F10)."""

import numpy as np

from mlframe.reporting.renderers._shared_helpers import low_evidence_mask, remap_low_evidence_indices


def test_flags_are_mapped_through_the_kept_rows():
    """Full-set indices {2, 5, 9} with kept rows [9, 0, 5, 3] flag positions 0 and 2 of the kept set (21-F10)."""
    got = remap_low_evidence_indices(np.array([2, 5, 9]), np.array([9, 0, 5, 3]), 10)
    assert sorted(got.tolist()) == [0, 2]
    assert low_evidence_mask(got, 4).tolist() == [True, False, True, False]


def test_no_downsample_and_no_flags_pass_through():
    """Without a selection the indices are unchanged, and None stays None (21-F10)."""
    idx = np.array([1, 2])
    assert remap_low_evidence_indices(idx, None, 5) is idx
    assert remap_low_evidence_indices(None, np.array([0, 1]), 5) is None


def test_matplotlib_subsample_carries_the_flags(monkeypatch):
    """The matplotlib subsample helper keeps flagged points flagged after capping the point count (21-F10)."""
    from mlframe.reporting.renderers import _matplotlib_scatter as ms
    from mlframe.reporting.spec import ScatterPanelSpec

    monkeypatch.setattr(ms, "_SCATTER_MAX_POINTS", 50)
    n = 400
    x = np.arange(n, dtype=float)
    y = np.sin(x / 7.0)
    flagged = np.arange(0, n, 8)
    p = ScatterPanelSpec(x=x, y=y, low_evidence_indices=flagged)
    xs, _ys, _s, _c, p2, raster = ms._subsample_scatter(x, y, p, None, None, n)
    assert raster and len(xs) < n
    mask = low_evidence_mask(p2.low_evidence_indices, len(xs))
    assert mask.sum() > 0
    assert set(xs[mask].astype(int).tolist()) <= set(flagged.tolist())
    assert all(int(v) in set(flagged.tolist()) for v in xs[mask])
    assert not any(int(v) % 8 == 0 for v in xs[~mask])
