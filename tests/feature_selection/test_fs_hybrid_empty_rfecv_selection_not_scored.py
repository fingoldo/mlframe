"""Audit 2026-11-20 finding 12-F20: an empty RFECV selection is recorded as empty and never scored as a fallback column."""

import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.fs_hybrid import importance_shootout as shootout


def test_empty_selection_is_not_scored(monkeypatch):
    """No columns: the row is flagged empty, and the downstream fit is never called."""
    calls = []
    monkeypatch.setattr(shootout, "downstream", lambda *a, **k: calls.append(a) or {"lgbm": 0.5})
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    row = shootout.score_selection({}, [], {"base": ["a"], "noise": ["b"]}, X, X, np.array([0, 1]), np.array([0, 1]), X.columns)
    assert row == {"empty_selection": True, "n_feat": 0}
    assert calls == []


def test_nonempty_selection_is_scored(monkeypatch):
    """A real selection is scored once with the recall and noise counts."""
    monkeypatch.setattr(shootout, "downstream", lambda *a, **k: {"lgbm": 0.9, "logit": 0.9, "knn": 0.9})
    monkeypatch.setattr(shootout, "compute_auc_mean", lambda auc: 0.9)
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    row = shootout.score_selection({}, ["a"], {"base": ["a"], "noise": ["b"]}, X, X, np.array([0, 1]), np.array([0, 1]), X.columns)
    assert row["n_feat"] == 1 and row["base_recall"] == 1.0 and row["noise_sel"] == 0
