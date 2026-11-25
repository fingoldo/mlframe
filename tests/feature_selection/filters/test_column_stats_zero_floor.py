"""The interaction floor recorded in fit metadata keeps a legitimate 0.0 and reports NaN only when the fit never set one."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._column_stats import record_column_stats


def _fit_meta(floor_attr):
    """Run the recorder on a one-column fit whose floor attribute is ``floor_attr`` (absent when ``None``)."""
    self = SimpleNamespace(nbins_strategy=None, quantization_nbins=None)
    if floor_attr is not None:
        self.fe_interaction_ii_floor_ = floor_attr
    record_column_stats(
        self,
        cols=["a"],
        nbins=None,
        cached_MIs=None,
        screen_caches=None,
        freqs_y=np.array([0.5, 0.5]),
        classes_y=np.array([0, 1]),
    )
    return self.fit_meta_


def test_zero_floor_is_kept_not_replaced_by_nan():
    """A floor of exactly 0.0 is a real value and must come through as 0.0."""
    assert _fit_meta(0.0)["fe_interaction_ii_floor"] == 0.0


def test_missing_floor_is_nan():
    """A fit that never set the floor reports NaN."""
    assert math.isnan(_fit_meta(None)["fe_interaction_ii_floor"])
