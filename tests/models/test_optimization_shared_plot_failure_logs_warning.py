"""MODELS-11 (2026-08-05 audit): several broad ``except Exception`` handlers across the models cluster
logged only at DEBUG, inconsistent with sibling broad-excepts in the same package that log at WARNING --
a real bug in the guarded code could go unnoticed. ``plot_search_state``'s plt.show/pause failure path is
one of the fixed sites; this test pins it now logs at WARNING.
"""

from __future__ import annotations

import logging

import numpy as np


def test_plt_show_failure_logs_at_warning_not_debug(caplog, monkeypatch):
    """A plt.show/pause failure (e.g. headless backend) must log at WARNING, not DEBUG."""
    import matplotlib.pyplot as plt

    from mlframe.models._optimization_shared import plot_search_state

    def _raise(*a, **k):
        """Always raises RuntimeError, simulating a headless-backend plt.show failure."""
        raise RuntimeError("simulated headless backend failure")

    monkeypatch.setattr(plt, "show", _raise)

    search_space = np.arange(5)
    known_candidates = np.array([0, 1, 2])
    known_evaluations = np.array([0.1, 0.2, 0.3])

    with caplog.at_level(logging.WARNING, logger="mlframe.models._optimization_shared"):
        plot_search_state(
            search_space=search_space,
            next_cand=3,
            new_y=0.4,
            best_candidate=2,
            best_evaluation=0.3,
            nsteps=1,
            expected_fitness=None,
            y_pred=None,
            y_std=None,
            ground_truth=None,
            known_candidates=known_candidates,
            known_evaluations=known_evaluations,
            skip_candidates=[],
            acquisition_method="EI",
            mode="test",
            additional_info="",
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "plt.show/pause failed" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one WARNING-level plt.show/pause failure log, got: {[r.getMessage() for r in caplog.records]}"
