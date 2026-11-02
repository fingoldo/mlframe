"""A declined device gate of the binned-aggregate family is retried once after the caches are released, and then falls back to the host."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _binned_numeric_agg_cands as C
from mlframe.feature_selection.filters import _binned_numeric_agg_resident as R


def _run(monkeypatch, outcomes):
    """``device_born_candidates`` with a scripted gate: each call returns the next of ``outcomes`` (an exception instance is raised)."""
    calls: list = []
    released: list = []

    def gate(*a, **k):
        """The scripted device gate."""
        calls.append(1)
        out = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(R, "local_mi_gate_binagg_resident", gate)
    monkeypatch.setattr(C, "_release_device_caches", lambda: released.append(1))
    raw = {"n1": {"group_col": "g", "agg_col": "a", "edges": [], "global": 0.0}}

    def fit_fn(*a, **k):
        """Recipes-only fit stand-in."""
        return None, raw

    X = pd.DataFrame({"g": np.arange(10.0), "a": np.arange(10.0)})
    state = C.device_born_candidates(fit_fn, X, np.zeros(10), ["g"], ["a"], ["mean"], 5, 3, 0, None, lambda names, r: list(names), False, None)[0]
    return state, len(calls), len(released)


def test_a_declined_gate_is_retried_after_releasing_the_caches(monkeypatch):
    """The first pass declines (``None``), the caches are released, the second pass answers: no host fallback."""
    state, calls, released = _run(monkeypatch, [None, []])
    assert (state, calls, released) == ("empty", 2, 1)


def test_two_declines_fall_back_to_the_host(monkeypatch):
    """Both passes decline: the state is ``unavailable`` and the caches were released once."""
    state, calls, released = _run(monkeypatch, [None, None])
    assert (state, calls, released) == ("unavailable", 2, 1)


def test_a_gate_that_answers_is_not_retried(monkeypatch):
    """A first-pass answer (here: no survivors) is final."""
    state, calls, released = _run(monkeypatch, [[]])
    assert (state, calls, released) == ("empty", 1, 0)
