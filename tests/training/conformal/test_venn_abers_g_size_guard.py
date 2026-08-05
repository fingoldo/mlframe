"""TRAINING_COMPOSITE_CORE_B-5 regression test: _isotonic_envelopes must guard/warn on a large number of
unique calibration scores, since _ivap_saddle_njit is self-documented O(g^2) with no separability
shortcut -- a caller calibrating on a large, mostly-continuous-score split (g approaching n_cal) could
otherwise hit tens of billions of inner-loop iterations with no warning at all.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.composite import venn_abers
from mlframe.training.composite.venn_abers import _isotonic_envelopes

pytestmark = pytest.mark.fast


def test_below_warn_threshold_no_warning(caplog):
    """A small g (well below the warn threshold) must not log a warning."""
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 1, 200))
    y = (rng.uniform(0, 1, 200) < s).astype(float)
    with caplog.at_level(logging.WARNING, logger=venn_abers.logger.name):
        _isotonic_envelopes(s, y)
    assert not any("unique calibration scores" in r.getMessage() for r in caplog.records)


def test_above_warn_threshold_logs_warning(monkeypatch, caplog):
    """g above the warn threshold (but below the hard cap) must log a warning, not silently run."""
    monkeypatch.setattr(venn_abers, "_VENN_ABERS_G_WARN", 50)
    monkeypatch.setattr(venn_abers, "_VENN_ABERS_G_HARD_CAP", 10_000)
    rng = np.random.default_rng(1)
    s = np.sort(rng.uniform(0, 1, 300))  # 300 unique scores > the monkeypatched warn=50
    y = (rng.uniform(0, 1, 300) < s).astype(float)
    with caplog.at_level(logging.WARNING, logger=venn_abers.logger.name):
        _isotonic_envelopes(s, y)
    assert any("unique calibration scores" in r.getMessage() for r in caplog.records)


def test_above_hard_cap_raises(monkeypatch):
    """g above the hard cap must raise, refusing to run an infeasible O(g^2) computation."""
    monkeypatch.setattr(venn_abers, "_VENN_ABERS_G_HARD_CAP", 50)
    rng = np.random.default_rng(2)
    s = np.sort(rng.uniform(0, 1, 300))  # 300 unique scores > the monkeypatched hard cap=50
    y = (rng.uniform(0, 1, 300) < s).astype(float)
    with pytest.raises(ValueError, match="hard cap"):
        _isotonic_envelopes(s, y)
