"""Regression test: ``redundancy_aggregator='auto'``'s synergy-detector failure must be
distinguishable from a genuine "data judged non-synergistic" decision.

Before the fix, ``self._synergy_auto_decision_`` only carried ``jmim_engaged`` (plus an
``error`` key on failure, which callers were not guaranteed to check) -- both a crashed
detector and a detector that legitimately judged the data non-synergistic set
``jmim_engaged=False``, so a caller checking only that one key could not tell them apart.
``detector_failed`` is now an explicit, always-present boolean disambiguating the two.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import MRMR
import mlframe.feature_selection.filters._synergy_detector as _synergy_detector_mod


def _fast(**kw):
    """Build a fast-fitting MRMR instance for these tests, overridable via kwargs."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def _xy(seed: int = 5, n: int = 160):
    """Build a small synthetic classification fixture with signal on columns 0 and 2."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.4 * X[:, 2] > 0).astype(np.int32)
    return X, y


def test_synergy_detector_success_sets_detector_failed_false():
    """A successful synergy-detector run must record detector_failed=False with no error key."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(redundancy_aggregator="auto").fit(X, y)
    assert hasattr(m, "_synergy_auto_decision_")
    assert m._synergy_auto_decision_["detector_failed"] is False
    assert "error" not in m._synergy_auto_decision_


def test_synergy_detector_crash_sets_detector_failed_true(monkeypatch):
    """A crashing synergy detector must be distinguishable from a "non-synergistic" verdict."""

    def _boom(*args, **kwargs):
        """Stand in for ``detect_synergy`` and always raise, simulating a detector crash."""
        raise RuntimeError("simulated synergy-detector crash")

    monkeypatch.setattr(_synergy_detector_mod, "detect_synergy", _boom)

    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(redundancy_aggregator="auto")
    with pytest.warns(UserWarning, match="synergy detector raised"):
        m.fit(X, y)
    assert m._synergy_auto_decision_["jmim_engaged"] is False
    assert m._synergy_auto_decision_["detector_failed"] is True
    assert "simulated synergy-detector crash" in m._synergy_auto_decision_["error"]
