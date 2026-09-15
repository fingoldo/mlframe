"""The frozen smart_log anchor replays a shared nested parent once per materialise call, not once per log-side candidate.

Many sibling candidates in one FE step nest the same parent recipe. Each log-side candidate replayed that parent over the whole frame only to
take one ``nanmin``, although the value is fixed for the call.
"""

from __future__ import annotations

import pandas as pd

from mlframe.feature_selection.filters import engineered_recipes
from mlframe.feature_selection.filters._mrmr_fe_step._step_log_anchor import smart_log_anchor


class _Parent:
    """A stand-in nested recipe: no quantization, replay computes ``a - 3``."""

    quantization = None


def _spy_apply(monkeypatch):
    """Replace ``apply_recipe`` with a counting ``a - 3`` replay and return the counter."""
    calls = {"n": 0}

    def _apply(recipe, X):
        """Count the replay and return the parent's continuous values."""
        calls["n"] += 1
        return X["a"].to_numpy() - 3.0

    monkeypatch.setattr(engineered_recipes, "apply_recipe", _apply)
    return calls


def test_shared_nested_parent_is_replayed_once(monkeypatch):
    """Five candidates sharing one parent cost one replay and all get the same, correct anchor."""
    calls = _spy_apply(monkeypatch)
    X = pd.DataFrame({"a": [0.5, 1.0, 2.0, 4.0]})
    parent = _Parent()
    memo: dict = {}
    anchors = [smart_log_anchor("a", parent, X, memo) for _ in range(5)]
    assert calls["n"] == 1, f"parent replayed {calls['n']} times"
    assert anchors == [1e-5 - (0.5 - 3.0)] * 5


def test_distinct_parents_and_raw_columns_are_keyed_separately(monkeypatch):
    """Different parents, and a raw column of the same name, each get their own anchor."""
    calls = _spy_apply(monkeypatch)
    X = pd.DataFrame({"a": [0.5, 1.0, 2.0, 4.0]})
    memo: dict = {}
    p1, p2 = _Parent(), _Parent()
    assert smart_log_anchor("a", p1, X, memo) == smart_log_anchor("a", p2, X, memo)
    assert calls["n"] == 2
    assert smart_log_anchor("a", None, X, memo) == 0.0  # raw a is positive: no shift
    assert smart_log_anchor("a", None, X.assign(a=[-1.0, 0.0, 1.0, 2.0]), {}) == 1e-5 + 1.0


def test_failed_reconstruction_returns_none(monkeypatch):
    """A missing source column yields None, as the legacy refit path expects."""
    assert smart_log_anchor("missing", None, pd.DataFrame({"a": [1.0]}), {}) is None
