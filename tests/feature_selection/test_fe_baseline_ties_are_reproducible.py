"""Tied quantised MIs must not let enumeration order decide the baseline or the seed columns."""

import numpy as np

import mlframe.feature_selection.filters.fe_baselines as fb


def test_the_best_baseline_among_ties_is_chosen_by_name(monkeypatch):
    """np.argmax took the lowest index, so the order of the trivial features decided the gate."""
    names = ["zeta", "alpha", "mid"]
    feats = {n: np.arange(10, dtype=float) + i for i, n in enumerate(names)}
    monkeypatch.setattr(fb, "trivial_pair_features", lambda a, b: feats)
    import mlframe.feature_selection.filters.hermite_fe as hf

    monkeypatch.setattr(hf, "_plugin_mi_classif_batch_njit", lambda X, y, nb: np.array([0.3, 0.3, 0.1]))
    name, _arr, mi = fb.best_trivial_pair(np.zeros(10), np.zeros(10), np.zeros(10, dtype=int))
    assert (name, mi) == ("alpha", 0.3), "of the two tied maxima, the name must win, not the position"

