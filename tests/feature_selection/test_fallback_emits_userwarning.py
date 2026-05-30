"""Wave 9.1 loop-iter-39 regression: when the min_features_fallback
engages on an empty support_, MRMR must emit a ``UserWarning`` (not
just a logger.warning) so callers can intercept it.

Pre-fix at ``_mrmr_fit_impl.py:988`` the fallback path emitted only
``logger.warning(...)``. ``warnings.simplefilter('error')`` had no
effect because logger emissions aren't captured by Python's warnings
machinery. Test suites and production guards that wanted to detect
"MRMR returned an uninformative feature" silently missed every
constant-X / degenerate-fit case.

The pre-fix message also didn't tell users WHY the fallback engaged
(degenerate MI ranking - the picked column may carry zero signal).

Severity: medium-high. The fallback is correctness-adjacent: when
every candidate has MI <= 0, the returned ``support_`` is index 0 by
tie-break and carries NO information. Without a programmatic-detectable
warning, downstream models silently train on noise.

Fix at _mrmr_fit_impl.py:988:
1. Keep the ``logger.warning(...)`` for back-compat.
2. ADD ``warnings.warn(..., UserWarning, stacklevel=2)`` so callers
   can use ``simplefilter('error')`` / ``catch_warnings(record=True)``.
3. Detect the "all-zero MI" sub-case and surface "carries NO signal"
   diagnostic.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def test_fallback_emits_userwarning_on_constant_x():
    """The iter-39 contract: ``warnings.catch_warnings(record=True)``
    must capture a UserWarning when fallback engages.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "a": np.ones(n),
        "b": np.full(n, 2.0),
        "c": np.full(n, 3.0),
    })
    y = pd.Series(rng.integers(0, 2, n))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = MRMR(verbose=0, min_features_fallback=1).fit(df, y)
    fallback_warns = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "falling back" in str(w.message)
    ]
    assert len(fallback_warns) >= 1
    # Must surface the "carries NO signal" diagnostic when all MIs are 0.
    msg = str(fallback_warns[0].message)
    assert "NO signal" in msg or "MI <= 0" in msg


def test_fallback_simplefilter_error_can_intercept():
    """sklearn-doctest-style test suites that use
    ``simplefilter('error')`` must be able to convert the fallback
    warning into a raise.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "a": np.ones(n), "b": np.full(n, 2.0), "c": np.full(n, 3.0),
    })
    y = pd.Series(rng.integers(0, 2, n))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="falling back"):
            MRMR(verbose=0, min_features_fallback=1).fit(df, y)


def test_no_fallback_warning_when_min_features_fallback_zero():
    """Negative control: ``min_features_fallback=0`` (the default)
    doesn't trigger the fallback even on degenerate input -- no warning.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({"a": np.ones(n)})
    y = pd.Series(rng.integers(0, 2, n))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MRMR(verbose=0, min_features_fallback=0).fit(df, y)
    fallback_warns = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "falling back" in str(w.message)
    ]
    assert not fallback_warns
