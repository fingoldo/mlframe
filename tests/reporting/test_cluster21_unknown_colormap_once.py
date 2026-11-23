"""Cluster 21 regression: an unknown colormap warns once per name, not on every call (audit 2026-11-20, finding 21-F22).

The mapping is called per trace, so an unthrottled warning flooded the log for one unmapped colormap. Repeated calls with the same unknown
name now produce a single warning; a known name produces none.
"""

import logging

from mlframe.reporting.renderers import _plotly_color as pc


def test_unknown_colormap_warns_once_for_repeated_calls(caplog):
    """Ten calls with the same unknown name produce one warning (21-F22)."""
    name = "definitely_not_a_colormap_xyz_21f22"
    with caplog.at_level(logging.WARNING, logger=pc.logger.name):
        for _ in range(10):
            assert pc._mpl_to_plotly_cmap(name) == "Viridis"
    hits = [r for r in caplog.records if "Unknown colormap" in r.getMessage() and name in r.getMessage()]
    assert len(hits) == 1


def test_known_colormap_is_silent(caplog):
    """Control: a mapped colormap name logs nothing (21-F22)."""
    with caplog.at_level(logging.WARNING, logger=pc.logger.name):
        out = pc._mpl_to_plotly_cmap("viridis")
    assert out
    assert not any("Unknown colormap" in r.getMessage() for r in caplog.records)
