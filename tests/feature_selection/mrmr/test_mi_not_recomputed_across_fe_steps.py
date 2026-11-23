"""The plug-in MI of a column against the target is computed once per fit, however many FE steps run (PLAN-08).

The screen of a later FE step reuses ``cached_MIs`` for every column an earlier step already scored. This counts the direct MI computations (``mi_direct`` and its device twin) per single-column key over a fit that
really runs two FE steps, and asserts that none repeats and that the second step adds no computation for a column the first one scored.
"""

from __future__ import annotations

import collections
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import evaluation as ev
from mlframe.feature_selection.filters.mrmr import MRMR

pytestmark = pytest.mark.slow


def _frame(n: int = 20000, p: int = 20, seed: int = 0):
    """Regression frame whose fit runs two FE steps (the second step recommends columns, the third does not)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)).astype(np.float32), columns=[f"x{i}" for i in range(p)])
    y = (np.sin(X["x0"] * X["x1"]) * X["x2"] + 0.2 * rng.standard_normal(n)).to_numpy()
    return X, y


def _spy(monkeypatch) -> list:
    """Record the column key of every direct MI computation (``mi_direct`` and ``mi_direct_gpu`` as the evaluation module binds them)."""
    calls: list = []

    def wrap(name: str) -> None:
        """Replace ``ev.<name>`` by a recorder that delegates."""
        real = getattr(ev, name)

        def spy(*args, **kwargs):
            """Record the ``x`` argument as a tuple of column indices and call the real function."""
            x = kwargs.get("x", args[1] if len(args) > 1 else None)
            try:
                key = tuple(int(v) for v in np.asarray(x).ravel())
            except (TypeError, ValueError):
                key = None
            calls.append(key)
            return real(*args, **kwargs)

        monkeypatch.setattr(ev, name, spy)

    for name in ("mi_direct", "mi_direct_gpu"):
        if hasattr(ev, name):
            wrap(name)
    return calls


def _fit(X, y, steps: int):
    """One cold fit with ``steps`` FE steps allowed."""
    MRMR._FIT_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=0, fe_max_steps=steps, skip_retraining_on_same_content=False).fit(X, y)


def test_no_single_column_mi_is_computed_twice_and_the_second_step_adds_none(monkeypatch):
    """One step against two executed steps: the same direct MI computations, none repeated."""
    X, y = _frame()
    calls = _spy(monkeypatch)
    one = _fit(X, y, 1)
    one_steps = one._fe_steps_executed_
    calls_one = [k for k in calls if k is not None and len(k) == 1]
    calls.clear()
    three = _fit(X, y, 3)
    calls_three = [k for k in calls if k is not None and len(k) == 1]
    assert one_steps == 1
    assert three._fe_steps_executed_ >= 2, "the fixture must run a second FE step, or this proves nothing"
    assert calls_one, "the spy must see the direct MI computations"
    repeated = {k: c for k, c in collections.Counter(calls_three).items() if c > 1}
    assert not repeated, f"single-column MI computed more than once in one fit: {repeated}"
    assert len(calls_three) == len(calls_one), f"the second FE step computed {len(calls_three) - len(calls_one)} extra direct MI(s) for columns the first step had scored"
