"""REPORTING_B-3 regression test: MatplotlibRenderer._histogram's bar-width expression must respect an
explicitly-supplied bin_width, matching PlotlyRenderer's (correctly parenthesized) behavior.

The bug (fixed): `float(p.bin_width or (bin_centers[1]-bin_centers[0]) if len(bin_centers)>1 else 1.0)`
relies on Python's ternary binding looser than `or`, parsing as `(p.bin_width or diff) if len>1 else 1.0`
-- an explicitly-supplied bin_width was silently discarded and forced to 1.0 whenever bin_centers had a
single element (the single-bin histogram case), even though bin_width was explicitly provided. Fixed to
`float(p.bin_width or ((bin_centers[1]-bin_centers[0]) if len(bin_centers)>1 else 1.0))`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import HistogramPanelSpec

pytestmark = pytest.mark.fast


def test_explicit_bin_width_respected_on_single_bin_histogram():
    """A single bin_center with an explicit bin_width must use that width, not be forced to 1.0."""
    renderer = MatplotlibRenderer()
    ax = MagicMock()
    spec = HistogramPanelSpec(
        values=np.array([5.0]),
        bin_centers=np.array([10.0]),
        bin_width=0.25,
    )

    renderer._histogram(ax, spec)

    _, kwargs = ax.bar.call_args
    assert kwargs["width"] == pytest.approx(0.25), f"explicit bin_width=0.25 should be used, got width={kwargs['width']}"


def test_default_width_still_falls_back_to_1_when_unset_and_single_bin():
    """Sanity: with bin_width unset (None) and a single bin_center, the fallback width is still 1.0."""
    renderer = MatplotlibRenderer()
    ax = MagicMock()
    spec = HistogramPanelSpec(
        values=np.array([5.0]),
        bin_centers=np.array([10.0]),
        bin_width=None,
    )

    renderer._histogram(ax, spec)

    _, kwargs = ax.bar.call_args
    assert kwargs["width"] == pytest.approx(1.0)


def test_multi_bin_width_derives_from_bin_centers_spacing_when_unset():
    """Sanity: with bin_width unset and 2+ bin_centers, the fallback width derives from center spacing."""
    renderer = MatplotlibRenderer()
    ax = MagicMock()
    spec = HistogramPanelSpec(
        values=np.array([5.0, 6.0]),
        bin_centers=np.array([10.0, 12.5]),
        bin_width=None,
    )

    renderer._histogram(ax, spec)

    _, kwargs = ax.bar.call_args
    assert kwargs["width"] == pytest.approx(2.5)
