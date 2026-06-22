"""Regression for two calibration/quality.py bugs:

1. show_classifier_calibration read `performances` / `data` even when the per-interval loop never
   ran (nintervals == 0), causing NameError. They are now initialised before the loop.
2. show_custom_calibration_plot's competing_probs branch indexed ax_probs[plot_idx] without the
   `nclasses == 1` guard used for the primary plot, crashing for single-class + competing_probs.
   The guard is now mirrored.
"""

from __future__ import annotations

import inspect

import numpy as np

from mlframe.calibration import quality


def test_performances_and_data_initialised_before_loop():
    # `data` and `performances` are read in the show_table / empty-all_performances return
    # branches; they must be bound before the per-interval loop so neither branch can hit a
    # NameError when the loop produced no iteration result. Pin the initialisation in source.
    src = inspect.getsource(quality.show_classifier_calibration)
    pre_loop = src.split("for i in range(nintervals):")[0]
    assert "data: list = []" in pre_loop
    assert "performances: dict = {}" in pre_loop


def test_show_classifier_calibration_returns_dict_normal_path():
    rng = np.random.default_rng(0)
    y_pred = rng.uniform(0.01, 0.99, size=500)
    y_true = (rng.uniform(size=500) < y_pred).astype(np.int8)
    res = quality.show_classifier_calibration(
        y_true, y_pred, title="t", nbins=5, nintervals=1, show_table=False, skip_plotting=True
    )
    assert isinstance(res, dict)


def test_competing_probs_branch_mirrors_nclasses_guard():
    # Pin the source-level fix: the competing-probs show_classifier_calibration call uses the
    # same `ax_probs if nclasses == 1 else ax_probs[plot_idx]` guard as the primary call, so a
    # single-class plot (ax_probs is a single Axes, not an array) does not index-crash.
    src = inspect.getsource(quality.make_custom_calibration_plot)
    # There must be at least two guarded ax selections of this shape (primary + competing).
    guarded = src.count("ax_probs if nclasses == 1 else ax_probs[plot_idx]")
    assert guarded >= 2, "competing_probs branch must mirror the nclasses==1 ax guard"
