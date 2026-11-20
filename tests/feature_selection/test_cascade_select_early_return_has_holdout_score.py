"""cascade_select documents a ``holdout_score`` key; the early-return paths must carry it too."""

import importlib

import numpy as np
import pandas as pd

cs = importlib.import_module("mlframe.feature_selection.cascade_select")


def test_no_boruta_survivor_result_still_has_a_holdout_score_key(monkeypatch):
    """When Boruta confirms nothing the result still has ``holdout_score`` (None)."""
    monkeypatch.setattr(cs, "boruta_select", lambda *a, **k: {"feature_names": ["a"], "decision": ["rejected"]})
    res = cs.cascade_select(pd.DataFrame({"a": np.arange(10.0)}), np.zeros(10), lambda: None)
    assert res["final_selected"] == [] and res["holdout_score"] is None


def test_no_forward_survivor_result_still_has_a_holdout_score_key(monkeypatch):
    """When forward selection keeps nothing the result still has ``holdout_score`` (None)."""
    monkeypatch.setattr(cs, "boruta_select", lambda *a, **k: {"feature_names": ["a"], "decision": ["confirmed"]})
    monkeypatch.setattr(cs, "forward_select", lambda *a, **k: [])
    res = cs.cascade_select(pd.DataFrame({"a": np.arange(10.0)}), np.zeros(10), lambda: None)
    assert res["final_selected"] == [] and res["holdout_score"] is None
