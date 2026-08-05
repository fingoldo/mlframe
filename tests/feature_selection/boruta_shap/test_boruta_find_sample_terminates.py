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
        """Groups tests covering NoMatch."""
        pvalue = 0.0  # never > 0.95

    monkeypatch.setattr(ss, "ks_2samp", lambda *a, **k: _NoMatch())

    preds = np.linspace(0.0, 1.0, 400)
    stub = _Stub(preds)

    result = stub.find_sample()  # must return via the size-exhaustion bound, not hang

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0
    assert result.shape[0] <= stub.X_boruta.shape[0]


def test_find_sample_starts_at_5_percent_not_10_percent():
    """FS_BORUTA_ROOT-2: find_sample must start its KS-test sample search at the FIRST (~5%) split
    element, matching its own docstring ("Starts of a 5%") -- starting at element=1 (~10%) skipped
    the documented first size entirely."""
    import mlframe.feature_selection.boruta_shap._shadow_stats as ss

    calls = []
    real_choice = ss.choice

    def _recording_choice(a, size, replace):
        """Records every requested sample size, then delegates to the real np.random.choice."""
        calls.append(size)
        return real_choice(a, size=size, replace=replace)

    orig_choice = ss.choice
    ss.choice = _recording_choice
    try:
        preds = np.linspace(0.0, 1.0, 400)
        stub = _Stub(preds)
        expected_sizes = stub.get_5_percent_splits(stub.X.shape[0])
        stub.find_sample()
    finally:
        ss.choice = orig_choice

    assert calls, "find_sample never called choice()"
    assert calls[0] == expected_sizes[0], f"first sample draw must use the FIRST (~5%) split size {expected_sizes[0]}, got {calls[0]}"


def test_find_sample_no_indexerror_on_tiny_frame():
    """FS_BORUTA_ROOT-2: on a 1-row frame, get_5_percent_splits returns an EMPTY array
    (np.arange(step, length, step) with step >= length) -- find_sample must not IndexError on
    size[element] in that case."""
    preds = np.array([0.5])
    stub = _Stub(preds)
    assert stub.get_5_percent_splits(stub.X.shape[0]).size == 0, "sanity: this fixture must exercise the empty-size-array path"
    result = stub.find_sample()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == stub.X_boruta.shape[0]
