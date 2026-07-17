"""Regression test: ICE.evaluate's periodic calibration plot no longer leaks matplotlib Figures.

Pre-fix: the ``fast_calibration_report(...)`` return-tuple unpacked ``fig`` (plus 7 numeric metrics)
and then discarded it entirely -- only ``metrics_string`` was logged. ``fast_calibration_report``
already shows/saves the plot internally (``show_plots`` defaults True), so the show/save side effect
itself worked, but the returned ``Figure`` object was never closed: over a long CatBoost training run
with ``calibration_plot_period`` set, this leaked one open Figure per period. Post-fix the figure is
always closed after logging, and ICE gained an optional ``plot_file`` param forwarded to
``fast_calibration_report`` so callers can also persist a disk copy each period.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np


def test_ice_evaluate_closes_calibration_figure():
    import matplotlib.pyplot as plt

    from mlframe.metrics.core import ICE, compute_probabilistic_multiclass_error

    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    logits = np.log(p / (1 - p))

    ice = ICE(
        metric=lambda y_true, y_score: compute_probabilistic_multiclass_error(y_true=y_true, y_score=y_score),
        higher_is_better=False,
        calibration_plot_period=1,
    )

    before = len(plt.get_fignums())
    ice.evaluate([logits], y, weight=None)
    after = len(plt.get_fignums())
    assert after == before, f"ICE.evaluate leaked a matplotlib Figure: {before} open before, {after} after"


def test_ice_plot_file_param_saves_to_disk(tmp_path):
    from mlframe.metrics.core import ICE, compute_probabilistic_multiclass_error

    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    logits = np.log(p / (1 - p))

    plot_file = str(tmp_path / "calib.png")
    ice = ICE(
        metric=lambda y_true, y_score: compute_probabilistic_multiclass_error(y_true=y_true, y_score=y_score),
        higher_is_better=False,
        calibration_plot_period=1,
        plot_file=plot_file,
    )
    ice.evaluate([logits], y, weight=None)
    assert (tmp_path / "calib.png").exists(), "ICE(plot_file=...) must persist the periodic calibration plot to disk"
