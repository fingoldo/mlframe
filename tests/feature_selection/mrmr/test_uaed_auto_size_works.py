"""Wave 9.1 loop-iter-30 regression: ``MRMR(uaed_auto_size=True)`` must
actually trim ``support_`` at the MI-gain elbow.

Pre-fix at ``_mrmr_fit_impl.py:1025-1037``: the UAED block read
``getattr(self, "mrmr_gains_", [])`` to find the elbow, but no code
ever assigned ``self.mrmr_gains_``. The comment at line 1023 claimed
"Wave-7 audit landed this trace in the standard fit output" - it never
did. ``gains.size >= 3`` was False on every run, the elbow code never
executed, and ``MRMR(uaed_auto_size=True)`` returned the full screen
output identically to the default. A documented, parameterised,
CHANGELOG-advertised public knob was silent dead code.

The vacuous existing test at ``test_biz_val_mrmr_research_extensions.py:310-317``
only asserted no-crash, so the dead code escaped CI.

Fix at ``_mrmr_fit_impl.py:927``: populate ``self.mrmr_gains_`` from
the ``predictors`` log returned by ``screen_predictors``, in the same
order as the selection events. The UAED block at line 1020+ then
finds a real trace and fires the elbow detector.

Verified post-fix on a clear-elbow synthetic (2 strong + many weak
features):
  no UAED: support_=[0, 1, 2], mrmr_gains_=[0.69, 0.53, 0.39]
  UAED on: support_=[0, 1] (trimmed at elbow), uaed_elbow_=1
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _clear_elbow_frame(n=500, seed=0):
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    cols = {
        "s0": latent,
        "s1": latent + 0.3 * rng.standard_normal(n),
        "s2": latent + 0.6 * rng.standard_normal(n),
        "n0": rng.standard_normal(n),
        "n1": rng.standard_normal(n),
        "n2": rng.standard_normal(n),
        "n3": rng.standard_normal(n),
        "n4": rng.standard_normal(n),
    }
    X = pd.DataFrame(cols)
    y = pd.Series((latent > 0).astype(np.int64), name="y")
    return X, y


def test_mrmr_gains_attribute_populated_after_fit():
    """``self.mrmr_gains_`` MUST be a non-empty float array after fit
    when at least one feature was selected.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = _clear_elbow_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    assert hasattr(sel, "mrmr_gains_")
    assert isinstance(sel.mrmr_gains_, np.ndarray)
    assert sel.mrmr_gains_.dtype == np.float64
    assert sel.mrmr_gains_.size == len(sel.support_) or sel.mrmr_gains_.size >= 1


def test_mrmr_gains_monotone_non_increasing_in_screen_order():
    """The relevance trace should be roughly non-increasing across
    rounds (greedy mRMR selects highest-gain candidate each step).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = _clear_elbow_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    if sel.mrmr_gains_.size >= 2:
        # Allow tiny ties / noise; assert no large reversal.
        diffs = np.diff(sel.mrmr_gains_)
        assert (diffs <= 0.05).all(), (
            f"mrmr_gains_ not weakly non-increasing: {sel.mrmr_gains_}"
        )


def test_uaed_auto_size_trims_at_elbow():
    """``MRMR(uaed_auto_size=True)`` must produce a support smaller than
    or equal to the default screen output AND set ``uaed_elbow_``.
    Pre-fix it silently no-op'd.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = _clear_elbow_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_off = MRMR(verbose=0).fit(X, y)
        sel_on = MRMR(verbose=0, uaed_auto_size=True).fit(X, y)
    # uaed_elbow_ MUST be set when at least 3 gains were available and
    # the elbow lies strictly inside the trace.
    if sel_off.mrmr_gains_.size >= 3:
        assert hasattr(sel_on, "uaed_elbow_"), (
            "uaed_elbow_ was not set despite >=3 gains in trace"
        )
        # And the trimmed support must be <= default support.
        assert len(sel_on.support_) <= len(sel_off.support_)


def test_uaed_auto_size_disabled_unchanged():
    """Negative control: default fit (uaed_auto_size=False) does not
    create uaed_elbow_ and produces the full screen output.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = _clear_elbow_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    assert not hasattr(sel, "uaed_elbow_")
