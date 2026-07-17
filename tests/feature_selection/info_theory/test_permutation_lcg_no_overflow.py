"""Wave 9.1 loop-iter-38 regression: ``mi_direct`` sequential-fallback
LCG MUST NOT emit ``RuntimeWarning: overflow encountered in scalar
multiply``.

Pre-fix at ``permutation.py:510, 514``: the non-njit sequential branch
used ``np.uint64`` scalar arithmetic for the LCG state update::

    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(_i + 1)
    state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)

NumPy's scalar-uint64 arithmetic emits ``RuntimeWarning: overflow
encountered in scalar multiply / scalar add`` on EVERY inner Fisher-
Yates iteration. Effects:

1. Hard crash under ``warnings.filterwarnings('error')`` (common in
   test suites; sklearn doctests trip this).
2. Hard error on numpy 2.x with stricter scalar overflow policy.
3. stderr spam at log-noise-level (10^6+ warning lines on a real fit)
   misleading downstream users to suspect MRMR correctness bugs.

Math IS correct under numpy 1.x (scalar wraps to uint64 modulus) but
the warning is a contract violation. The njit-bodied LCG sites are
fine because numba's uint64 wraps silently.

Fix at permutation.py:510 - use Python ``int`` arithmetic with
explicit ``& MASK64`` masking. Bit-exact to the njit ``parallel_mi``
LCG (which IS the iter-18 reproducibility contract this branch mirrors).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def test_no_overflow_warning_under_strict_filterwarnings():
    """The iter-38 contract: a basic MRMR fit MUST NOT raise the
    ``overflow encountered in scalar`` RuntimeWarning even when
    ``warnings.filterwarnings('error')`` is active.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "const": np.ones(n),
            "sig": rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    y = pd.Series((df["sig"] > 0).astype(np.int64))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="overflow encountered in scalar",
            category=RuntimeWarning,
        )
        # The fit must complete without raising.
        sel = MRMR(verbose=0, build_friend_graph=False).fit(df, y)
    assert sel.support_ is not None


def test_no_overflow_warning_emitted_in_normal_mode():
    """Even without 'error' filter, the warning must not be emitted -
    catch all RuntimeWarnings during fit and assert none match the
    overflow pattern.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "sig": rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    y = pd.Series((df["sig"] > 0).astype(np.int64))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MRMR(verbose=0, build_friend_graph=False).fit(df, y)
    overflow_warnings = [w for w in caught if (issubclass(w.category, RuntimeWarning) and "overflow encountered in scalar" in str(w.message))]
    assert not overflow_warnings, f"emitted {len(overflow_warnings)} overflow RuntimeWarning(s): {[str(w.message) for w in overflow_warnings[:3]]}"
