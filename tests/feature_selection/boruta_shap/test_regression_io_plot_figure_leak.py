"""Regression test for the BorutaShap plot/box_plot figure leak (LEAK-P2).

Pre-fix: box_plot created a figure via plt.figure() that plot() never closed on the
display=True path (and box_plot returned None, so plot had no handle to close).
"""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mlframe.feature_selection.boruta_shap import _io_plot


def _make_fake_self():
    """Minimal stand-in exposing only what plot/box_plot read on `self`."""
    self = types.SimpleNamespace()
    # history_x: row 0 is a header sentinel (sliced off via .iloc[1:]), then importance rows.
    self.history_x = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [0.0, 0.5, 1.5]})
    self.accepted = ["f1"]
    self.tentative = []
    self.rejected = ["f2"]

    self.create_mapping_of_features_to_attribute = types.MethodType(_io_plot.create_mapping_of_features_to_attribute, self)
    self.filter_data = staticmethod(_io_plot.filter_data)
    self.check_if_which_features_is_correct = staticmethod(_io_plot.check_if_which_features_is_correct)
    self.box_plot = types.MethodType(_io_plot.box_plot, self)
    self.create_list = staticmethod(_io_plot.create_list)
    self.to_dictionary = staticmethod(_io_plot.to_dictionary)
    self.plot = types.MethodType(_io_plot.plot, self)
    return self


def test_plot_does_not_leak_figures_display_true():
    """Plot does not leak figures display true."""
    self = _make_fake_self()
    before = len(plt.get_fignums())
    self.plot(display=True)
    after = len(plt.get_fignums())
    assert after == before, f"plot(display=True) leaked {after - before} figure(s)"


def test_plot_does_not_leak_figures_display_false():
    """Plot does not leak figures display false."""
    self = _make_fake_self()
    before = len(plt.get_fignums())
    self.plot(display=False)
    after = len(plt.get_fignums())
    assert after == before, f"plot(display=False) leaked {after - before} figure(s)"
