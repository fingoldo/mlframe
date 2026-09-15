"""A candidate whose replayed relevance is NaN must not be counted as selected on every resample.

``np.argpartition`` sorts NaN to the high end, so the top-``n_selected`` slice always included a degenerate candidate, which the report then
showed with selection frequency 1.0 and HIGH confidence.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _mrmr_stability_report as sr


class _Stub:
    """The minimal fitted surface the report reads."""

    def __init__(self, state):
        """Hold the replay state."""
        self._stability_replay_state_ = state
        self._engineered_recipes_ = []

    def _effective_random_seed(self):
        """Fixed seed."""
        return 0


def test_stability_report_nan_relevance_is_not_counted_as_selected(monkeypatch):
    """Candidate 0 replays to NaN MI; with one slot selected it must score 0.0, and the real signal candidate must take the slot."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    cand = np.column_stack([np.zeros(n, dtype=np.int64), y, rng.integers(0, 4, size=n)])
    state = {"cand_codes": cand, "y_codes": y, "cand_names": ["degenerate", "signal", "noise"], "selected_mask": np.array([False, True, False])}
    real = sr._marginal_mi_codes

    def mi_or_nan(x, yb):
        """NaN for the constant column, the real estimate otherwise."""
        return float("nan") if np.all(x == x[0]) else real(x, yb)

    monkeypatch.setattr(sr, "_marginal_mi_codes", mi_or_nan)
    res = sr.selection_stability_report(_Stub(state), n_boot=20, random_state=0, as_text=False)
    freq = res["feature_selection_frequency"]
    assert freq["degenerate"] == 0.0, f"a NaN-relevance candidate was counted as selected: {freq}"
    assert freq["signal"] == 1.0
