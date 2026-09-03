"""PREPROCESSING-10 regression test: clusterize's show_plot=True path must actually render the figure.

The bug (fixed): the show_plot branch built the full matplotlib figure (scatter data, title, annotations)
then immediately called plt.close(fig) without ever calling .show(), saving, or returning it -- so the
default show_plot=True call produced no visible or persisted output at all, despite the docstring's
"render a scatter plot of the clusters" promise.
"""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for CI; plt.show() is a harmless no-op under it

import numpy as np
import pytest

from mlframe.preprocessing.cluster import clusterize

pytestmark = pytest.mark.fast


def test_show_plot_true_calls_plt_show():
    """show_plot=True must call plt.show(), not silently close the figure unshown."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))

    with patch("matplotlib.pyplot.show") as mock_show:
        clusterize(X=X, show_plot=True, show_metrics=False, list_members=False)
        mock_show.assert_called_once()


def test_show_plot_false_does_not_call_plt_show():
    """show_plot=False must not touch matplotlib at all."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))

    with patch("matplotlib.pyplot.show") as mock_show:
        clusterize(X=X, show_plot=False, show_metrics=False, list_members=False)
        mock_show.assert_not_called()


def test_show_plot_true_still_closes_the_figure_to_avoid_leaking():
    """The figure must still be closed after rendering (no figure-leak regression)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 2))

    with patch("matplotlib.pyplot.close") as mock_close:
        clusterize(X=X, show_plot=True, show_metrics=False, list_members=False)
        mock_close.assert_called_once()
