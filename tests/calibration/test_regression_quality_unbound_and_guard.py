"""Regression for two calibration/quality.py bugs:

1. show_classifier_calibration read `performances` / `data` even when the per-interval loop never
   ran (nintervals == 0), causing NameError. They are now initialised before the loop.
2. show_custom_calibration_plot's competing_probs branch indexed ax_probs[plot_idx] without the
   `nclasses == 1` guard used for the primary plot, crashing for single-class + competing_probs.
   The guard is now mirrored.
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration import quality


def test_show_table_path_returns_dataframe_without_unbound_name():
    # `data` / `performances` are read in the show_table / empty-all_performances return
    # branches; before the fix they were bound only inside the per-interval loop body, so the
    # show_table return branch (which reads `data`) hit a NameError. Exercise that exact branch
    # and assert a DataFrame comes back rather than a NameError.
    rng = np.random.default_rng(0)
    y_pred = rng.uniform(0.01, 0.99, size=500)
    y_true = (rng.uniform(size=500) < y_pred).astype(np.int8)
    res = quality.show_classifier_calibration(y_true, y_pred, title="t", nbins=5, nintervals=1, show_table=True, skip_plotting=True)
    import pandas as pd

    assert isinstance(res, pd.DataFrame)


def test_show_classifier_calibration_returns_dict_normal_path():
    rng = np.random.default_rng(0)
    y_pred = rng.uniform(0.01, 0.99, size=500)
    y_true = (rng.uniform(size=500) < y_pred).astype(np.int8)
    res = quality.show_classifier_calibration(y_true, y_pred, title="t", nbins=5, nintervals=1, show_table=False, skip_plotting=True)
    assert isinstance(res, dict)


def test_competing_probs_single_class_does_not_index_crash():
    # The competing-probs branch indexed ax_probs[plot_idx] without the nclasses==1 guard the
    # primary call uses. With a single class and skip_plotting=True, ax_probs is None, so the
    # unguarded ax_probs[plot_idx] was `None[0]` -> TypeError. The guarded branch passes ax_probs
    # straight through. Drive the actual competing path and assert it completes without crashing.
    rng = np.random.default_rng(0)
    n = 400
    probs = rng.uniform(0.01, 0.99, size=(n, 1))
    y = (rng.uniform(size=n) < probs[:, 0]).astype(np.int8)
    import pandas as pd

    competing_col = "p_competitor"
    X = pd.DataFrame({competing_col: rng.uniform(0.01, 0.99, size=n)})

    fig, metrics = quality.make_custom_calibration_plot(
        y,
        probs,
        nclasses=1,
        nbins=5,
        competing_probs=[[competing_col]],
        X=X,
        skip_plotting=True,
    )
    assert isinstance(metrics, dict)


def test_multiclass_skip_plotting_does_not_index_none_ax_probs():
    # Bug: `ax=ax_probs if nclasses == 1 else ax_probs[plot_idx]` evaluated the indexing
    # expression eagerly regardless of `skip_plotting`. With skip_plotting=True and nclasses > 1,
    # ax_probs is None (the plotting branch is skipped entirely), so `ax_probs[plot_idx]` raised
    # TypeError: 'NoneType' object is not subscriptable, even though ax is never used downstream
    # when skip_plotting=True. Fixed by short-circuiting to None when skip_plotting.
    rng = np.random.default_rng(0)
    n = 300
    nclasses = 3
    probs = rng.dirichlet(np.ones(nclasses), size=n)
    y = rng.integers(0, nclasses, size=n)

    fig, metrics = quality.make_custom_calibration_plot(
        y,
        probs,
        nclasses=nclasses,
        nbins=5,
        skip_plotting=True,
    )
    assert isinstance(metrics, dict)
    assert len(metrics) == nclasses
