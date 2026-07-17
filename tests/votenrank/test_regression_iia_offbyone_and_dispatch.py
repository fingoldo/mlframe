"""Regression: compute_iia_for_fixed_models used range(3, len(models_order)) which never
considered the LAST model (off-by-one); it must be range(3, len(models_order) + 1). Also the
eval(...) dispatch in iia_exp / Leaderboard is replaced with getattr."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.votenrank import Leaderboard
from mlframe.votenrank import iia_exp


def _table(n_models, n_tasks=5):
    """Helper that table."""
    rng = np.random.default_rng(0)
    idx = [f"M{i}" for i in range(n_models)]
    cols = [f"t{j}" for j in range(n_tasks)]
    return pd.DataFrame(rng.uniform(size=(n_models, n_tasks)), index=idx, columns=cols)


def test_compute_iia_processes_the_last_model():
    """Compute iia processes the last model."""
    table = _table(6)
    models_order = table.index.tolist()
    weights = {c: 1.0 for c in table.columns}

    # The number of comparison iterations is len(models_order) - 2 once the last model is
    # included. The IIA result counts how many of those iterations show a ranking change, so
    # it cannot exceed the iteration count. Pre-fix the loop ran one fewer iteration, capping
    # the maximum possible result one lower. We assert by counting iterations via a spy.
    calls = {"n": 0}
    orig = iia_exp.fine_sorted_ranking

    def spy(ranking):
        """Helper that spy."""
        calls["n"] += 1
        return orig(ranking)

    iia_exp.fine_sorted_ranking = spy
    try:
        iia_exp.compute_iia_for_fixed_models("borda", table, models_order, weights)
    finally:
        iia_exp.fine_sorted_ranking = orig

    # 1 base call + (len-2) per-iteration calls when the last model is included.
    expected = 1 + (len(models_order) - 2)
    assert calls["n"] == expected, f"expected {expected} ranking calls, got {calls['n']}"


def test_leaderboard_getattr_dispatch_runs():
    # Exercises getattr-based dispatch in elect_all / rank_all (formerly eval(...)).
    """Leaderboard getattr dispatch runs."""
    lb = Leaderboard(_table(4))
    elected = lb.elect_all()
    assert not elected.empty
    ranked = lb.rank_all()
    assert not ranked.empty
