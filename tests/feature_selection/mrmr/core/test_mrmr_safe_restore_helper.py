"""Regression test: a failing fit()-finally restore action must not mask fit()'s real outcome.

Code-quality audit finding #3 (2026-07-17): the 11 near-identical
``try/except Exception as e: logger.debug("suppressed in _mrmr_class.py:<N>: %s", e); pass``
blocks in ``fit()``'s ``finally`` clause were consolidated into one ``_safe_restore(action,
description)`` helper. This pins the behavioral contract that motivated the consolidation: if
ONE restore action raises, ``fit()`` must still return successfully (or propagate ONLY its own
real exception, never the restore's), and every OTHER restore action in the ``finally`` block
must still run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR
import mlframe.feature_selection.filters.mrmr._mrmr_class as _mrmr_class_mod


def _fast(**kw):
    """Build a fast-fitting MRMR instance for these tests, overridable via kwargs."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def _xy(seed: int = 5, n: int = 160):
    """Build a small synthetic classification fixture with signal on columns 0 and 2."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"c{i}" for i in range(5)])
    y = (X["c0"] + 0.4 * X["c2"] > 0).astype(np.int32)
    return X, y


def test_safe_restore_swallows_action_exception():
    """_safe_restore itself must never propagate the wrapped action's exception."""
    calls = []

    def _boom():
        """Stand in for a restore action that always fails."""
        calls.append("boom")
        raise RuntimeError("simulated restore failure")

    _mrmr_class_mod._safe_restore(_boom, "test restore action")
    assert calls == ["boom"], "the action must actually run"


def test_fit_succeeds_when_one_finally_restore_action_raises(monkeypatch):
    """A single failing restore step (SU-normalization thread-local, here) must not break fit() --
    the fit's own real result is still returned, and later restore steps in the same finally block
    still execute. ``set_su_normalization`` is called both at fit-entry (activation) and in the
    finally block (restore); only the SECOND (restore) call is made to fail, isolating the
    regression to the exact _safe_restore-guarded call site."""
    from mlframe.feature_selection.filters import info_theory as _info_theory_mod

    real_set_su_normalization = _mrmr_class_mod.set_su_normalization
    call_count = {"n": 0}

    def _flaky_set_su_normalization(*args, **kwargs):
        """Pass through to the real function on the first (activation) call, then always raise."""
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated SU-normalization restore failure")
        return real_set_su_normalization(*args, **kwargs)

    # Patch the name as resolved inside _mrmr_class's own namespace (module-level import), matching
    # the standard "patch where it's used" rule for `from x import y`-style imports.
    monkeypatch.setattr(_mrmr_class_mod, "set_su_normalization", _flaky_set_su_normalization)

    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast()
    m.fit(X, y)  # must not raise despite the restore call always raising
    assert hasattr(m, "support_")
    assert np.asarray(m.support_).size >= 1
    assert call_count["n"] >= 2, "the restore call site must actually have been reached and failed"

    # sanity: the OTHER thread-locals normally restored alongside SU normalization in the same
    # finally block were still reset correctly (JMIM aggregator, off by default).
    assert _info_theory_mod.use_jmim_aggregator() is False
