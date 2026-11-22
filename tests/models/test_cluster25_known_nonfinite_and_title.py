"""Audit 2026-11-20 findings 25-F28 and 25-F32 in the optimisation search.

25-F28: a non-finite warm-start evaluation is dropped before it reaches the surrogate, as submit_evaluations already does.
25-F32: the plot title formats the best candidate only when there is one, so a NaN first evaluation does not raise.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from mlframe.models._optimization_search import MBHOptimizer
from mlframe.models._optimization_shared import plot_search_state


def test_non_finite_warm_start_evaluation_is_dropped():
    """The NaN pair is removed from both arrays, so the candidates and evaluations stay aligned for the surrogate."""
    opt = MBHOptimizer.__new__(MBHOptimizer)
    cands = opt._init_known_data(np.array([1, 3, 5]), np.array([0.5, np.nan, 0.7]))
    assert opt.known_evaluations.tolist() == [0.5, 0.7]
    assert opt.known_candidates.tolist() == [1, 5]
    assert cands.tolist() == [1, 5]


def test_plot_title_survives_no_best_candidate(monkeypatch):
    """With best_candidate None (a NaN first evaluation) the figure is drawn and the title says None."""
    import matplotlib.pyplot as plt

    # The function closes its own figure, so keep it open long enough to read the title back.
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    plot_search_state(
        search_space=np.arange(10),
        next_cand=4,
        new_y=0.5,
        best_candidate=None,
        best_evaluation=0.0,
        nsteps=1,
        expected_fitness=None,
        y_pred=None,
        y_std=None,
        ground_truth=None,
        known_candidates=np.array([], dtype=int),
        known_evaluations=np.array([], dtype=float),
        skip_candidates=[],
        acquisition_method="EI",
        mode="Regular",
        additional_info="",
    )
    titles = [ax.get_title() for ax in plt.gcf().axes]
    plt.close("all")
    assert any("best=0.000000@None" in title for title in titles), titles
