"""An explicit value for a fast-search knob wins, even when it equals the package value.

The fast-search profile used to override a knob whenever it still equalled its constructor default. For the boolean
knobs the default WAS the value a caller would ask for, so ``fe_stability_vote_enable=True`` was indistinguishable
from not passing it and was silently switched off for the whole fit. The knobs now default to ``None`` (auto), so
"not passed" and "passed the package value" are different values, and that survives ``sklearn.clone``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters.mrmr import MRMR

_KNOBS = ("fe_stability_vote_enable", "fe_escalation_underdelivery_enable", "fe_pair_prewarp_enable")


def _frame():
    """A small frame with a real signal, enough for the FE step to run."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 5)), columns=list("abcde"))
    return X, (X["a"] + X["b"] > 0).astype(int)


def _values_seen_during_fit(monkeypatch, **kwargs):
    """Fit and return the knob values the fit actually ran with, read at the start of the FE step."""
    import mlframe.feature_selection.filters._mrmr_fe_step as fe_step
    import mlframe.feature_selection.filters._mrmr_fe_step._step_score as step_score

    seen: list = []
    real = step_score.materialise_and_finalise_fe_candidates

    def spy(self, *a, **k):
        """Record the knob values in force when the FE step runs, then run it."""
        seen.append({name: getattr(self, name) for name in _KNOBS})
        return real(self, *a, **k)

    monkeypatch.setattr(step_score, "materialise_and_finalise_fe_candidates", spy)
    for name in dir(fe_step):
        if getattr(fe_step, name, None) is real:
            monkeypatch.setattr(fe_step, name, spy)
    X, y = _frame()
    MRMR._FIT_CACHE.clear()
    est = MRMR(random_state=0, verbose=0, fe_max_steps=1, full_npermutations=2, baseline_npermutations=2, **kwargs).fit(X, y)
    assert seen, "the FE step never ran, so this test observed nothing"
    return seen[0], est


@pytest.mark.parametrize("knob", _KNOBS)
def test_an_explicit_true_survives_fast_search(monkeypatch, knob):
    """The failure this pins: asking for the package value under fast search used to get the fast value instead."""
    during, est = _values_seen_during_fit(monkeypatch, fe_fast_search=True, **{knob: True})
    assert during[knob] is True, f"{knob}=True was overridden to {during[knob]!r} by the fast-search profile"
    assert getattr(est, knob) is True, "the explicit value must still be there after the fit"


@pytest.mark.parametrize("knob", _KNOBS)
def test_auto_follows_fast_search_and_is_restored(monkeypatch, knob):
    """Not passing the knob still gets the profile's value, and the constructor value comes back after the fit."""
    during, est = _values_seen_during_fit(monkeypatch, fe_fast_search=True)
    assert during[knob] is False, f"auto {knob} should resolve to the fast-search value, got {during[knob]!r}"
    assert getattr(est, knob) is None, "the auto value must be restored after the fit"


def test_auto_is_on_without_fast_search(monkeypatch):
    """Without the profile, auto means the package value, which is on for all three."""
    during, _ = _values_seen_during_fit(monkeypatch, fe_fast_search=False)
    assert len(_KNOBS) == 3
    assert all(during[k] is True for k in _KNOBS), during


def test_an_explicit_value_survives_clone():
    """clone() re-passes every parameter; an explicit True must still read as explicit, and auto as auto."""
    explicit = clone(MRMR(fe_stability_vote_enable=True))
    auto = clone(MRMR())
    assert explicit.fe_stability_vote_enable is True
    assert auto.fe_stability_vote_enable is None
