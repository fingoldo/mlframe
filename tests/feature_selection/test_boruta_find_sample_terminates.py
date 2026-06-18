"""Regression: BorutaShap.find_sample must terminate even when no sub-sample
reaches the KS p>0.95 threshold.

Pre-fix the ``while loop:`` never set ``loop=False`` and ``iteration`` was never
incremented, so the ``iteration==20`` size-growth / exit branch was dead and the
only exit was the KS ``break``. On a frame where no draw passes the KS test the
loop ran forever. The fix increments ``iteration``, grows the sample size on each
20-miss streak, and exits once every size is exhausted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


class _Stub:
    """Minimal duck-type carrying just what ``find_sample`` reads."""

    find_sample = BorutaShap.find_sample
    get_5_percent = staticmethod(BorutaShap.get_5_percent)
    get_5_percent_splits = BorutaShap.get_5_percent_splits

    def __init__(self, preds: np.ndarray):
        self.preds = preds
        self.X = pd.DataFrame({"f": preds})
        self.X_boruta = pd.DataFrame({"f": preds, "shadow_f": preds[::-1]})


def test_find_sample_terminates_when_no_subsample_ever_matches(monkeypatch):
    """Forces the exact bug condition: no draw ever reaches KS p>0.95.

    Pre-fix the KS ``break`` was the ONLY exit (``iteration`` never incremented,
    so the ``iteration==20`` size-growth branch was dead) -- this hangs forever.
    Post-fix the loop grows the sample size on each 20-miss streak and exits once
    every size is exhausted, returning the last draw.
    """
    import mlframe.feature_selection.boruta_shap._shadow_stats as ss

    class _NoMatch:
        pvalue = 0.0  # never > 0.95

    monkeypatch.setattr(ss, "ks_2samp", lambda *a, **k: _NoMatch())

    preds = np.linspace(0.0, 1.0, 400)
    stub = _Stub(preds)

    result = stub.find_sample()  # must return via the size-exhaustion bound, not hang

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0
    assert result.shape[0] <= stub.X_boruta.shape[0]
