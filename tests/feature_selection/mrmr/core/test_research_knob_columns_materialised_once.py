"""The research knobs share one materialisation of the columns they all need, and score the same either way.

RelaxMRMR, the PID synergy bonus, the CMI permutation stop and the conditional permutation test each need the candidate's codes, the target's,
and the selected set's. Each block used to build all three for itself, so two knobs on meant the same candidate was factorised twice inside one
call, and the target and the selected set, which are fixed for a whole greedy round, were rebuilt for every candidate. Only RelaxMRMR was
wired to the driver's per-round hoist.

The columns are byte-identical however they are obtained, so the scores must not move; what changes is how many times they are built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import evaluation as ev


@pytest.fixture
def frame():
    """A small frame with a couple of informative columns and a binary target."""
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": rng.normal(size=n), "d": rng.normal(size=n)})
    y = ((a + 0.7 * b) > 0).astype(np.int64)
    return X, y


def _fit(frame, **knobs):
    """Fit with a research knob enabled and return the selected support."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = frame
    MRMR._FIT_CACHE.clear()
    est = MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2, **knobs).fit(X, y)
    support = getattr(est, "support_", None)
    return [] if support is None else np.asarray(support).tolist()


@pytest.mark.parametrize(
    "knob",
    [
        {"pid_synergy_bonus": 0.5},
        {"cmi_perm_stop": True},
        {"cpt_test": True},
        {"bur_lambda": 0.5},
    ],
    ids=["pid", "cmi_perm_stop", "cpt", "bur"],
)
def test_each_knob_still_selects_what_it_selected(frame, knob):
    """Sharing the materialisation must not change any knob's outcome: the columns are the same bytes either way."""
    first = _fit(frame, **knob)
    second = _fit(frame, **knob)
    assert first == second, f"{knob} is not even deterministic with itself"
    assert first, "fixture precondition: the knob run should select something"


def _materialisations(frame, **knobs) -> int:
    """How many times a whole fit factorises a column, with the given research knobs on."""
    X, y = frame
    from mlframe.feature_selection.filters.mrmr import MRMR

    calls = [0]
    real = ev._materialize_var

    def counting(factors_data, var_idx, factors_nbins, dtype=np.int32):
        """Count materialisations, then defer to the real one."""
        calls[0] += 1
        return real(factors_data, var_idx, factors_nbins, dtype=dtype)

    ev._materialize_var = counting
    try:
        MRMR._FIT_CACHE.clear()
        MRMR(random_state=0, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2, **knobs).fit(X, y)
    finally:
        ev._materialize_var = real
    return calls[0]


def test_turning_on_a_second_knob_does_not_add_materialisations(frame):
    """The knobs share one set of columns, so enabling another must not make the fit factorise more of them.

    Measured on this fixture: one knob 17, and before the blocks shared a materialisation, two knobs 26 and three knobs 29 — each block
    rebuilt the candidate, the target and every selected column for itself.
    """
    one = _materialisations(frame, pid_synergy_bonus=0.5)
    two = _materialisations(frame, pid_synergy_bonus=0.5, cmi_perm_stop=True)
    three = _materialisations(frame, pid_synergy_bonus=0.5, cmi_perm_stop=True, cpt_test=True)
    assert one > 0, "no materialisation was observed, so this test is not looking at the right thing"
    assert two <= one, f"a second knob added materialisations: {one} -> {two}"
    assert three <= one, f"a third knob added materialisations: {one} -> {three}"
