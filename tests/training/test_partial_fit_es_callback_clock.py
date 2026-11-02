"""The partial_fit wrapper stamps its callback with the clock the callback reads (tracker 2026-11-02 wave 9, TIME9-2)."""

import timeit

import numpy as np
from sklearn.linear_model import SGDRegressor

from mlframe.training import _partial_fit_es_wrapper as w
from mlframe.training.callbacks import UniversalCallback


def test_callback_start_time_uses_the_default_timer_clock(monkeypatch):
    """time.time() is ~1.7e9 seconds, default_timer() is far smaller; mixing them made every elapsed-time check meaningless."""
    seen = []

    class _Recording(UniversalCallback):
        """Remember the callback so the test can read its stamps after the fit."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(self)

    import mlframe.training.callbacks as cbs

    monkeypatch.setattr(cbs, "UniversalCallback", _Recording)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=200)
    w.PartialFitESWrapper(SGDRegressor(random_state=0), metric="rmse", patience=2, max_iter=3, random_state=0).fit(X, y)
    assert seen
    assert abs(seen[0].start_time - timeit.default_timer()) < 1e6
