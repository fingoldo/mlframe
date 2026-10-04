"""Plot helpers must not leave pyplot figures open after they return."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_stability_cv_performance_plot_leaves_no_open_figure(tmp_path):
    """Saving the CV curve to a file must leave the pyplot figure count unchanged."""
    from mlframe.feature_selection.wrappers.rfecv._stability_select import _plot_cv_performance

    plt.close("all")
    nf = np.array([1, 2, 3, 4])
    perf = np.array([0.5, 0.6, 0.7, 0.65])
    before = len(plt.get_fignums())
    _plot_cv_performance(False, str(tmp_path / "cv.png"), 10, (4, 3), nf, perf, perf * 0.1, perf, 2, perf)
    assert len(plt.get_fignums()) == before
    assert (tmp_path / "cv.png").exists()


def test_stability_cv_performance_plot_closes_even_when_saving_fails(tmp_path):
    """An exception while saving still closes the figure."""
    from mlframe.feature_selection.wrappers.rfecv._stability_select import _plot_cv_performance

    plt.close("all")
    nf = np.array([1, 2, 3])
    perf = np.array([0.5, 0.6, 0.7])
    bad = tmp_path / "missing_dir_is_created" / "x.unknownext"
    with pytest.raises(ValueError):
        _plot_cv_performance(False, str(bad), 10, (4, 3), nf, perf, perf, perf, 1, perf)
    assert len(plt.get_fignums()) == 0


def test_votenrank_stability_pic_leaves_no_open_figure(tmp_path):
    """create_exp_pic with a filename closes its figure."""
    pytest.importorskip("seaborn")
    from mlframe.votenrank.stability_exp import create_exp_pic

    plt.close("all")
    rng = np.random.default_rng(0)
    exp_range = np.linspace(0.0, 0.2, 5)
    exp_res = {m: rng.random((5, 3)).mean(axis=1) for m in ("a", "b", "c", "d")}
    try:
        create_exp_pic(exp_range, exp_res, filename=str(tmp_path / "s.pdf"))
    except (TypeError, ValueError) as exc:
        pytest.skip(f"seaborn version rejects the legacy ci argument: {exc}")
    assert len(plt.get_fignums()) == 0
