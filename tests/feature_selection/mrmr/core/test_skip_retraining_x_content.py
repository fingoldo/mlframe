"""Wave 9.1 loop-iter-36 regression: ``skip_retraining_on_same_shape``
must include X CONTENT in the signature.

Pre-fix at ``_mrmr_fit_impl.py:120``::

    signature = (X.shape, y.shape, _y_hash_for_sig, _x_cols_sig)

The X CONTENT hash was absent. Refitting the same MRMR instance with
identical shape + column names + y but different X content silently
replayed the prior fit's ``support_``. The companion ``_FIT_CACHE``
path at line 132+ already folded ``_full_x_content_hash``, leaving
the two cache layers with asymmetric guarantees.

Concrete repro: rolling-window retraining where shape/cols/y are
constant across windows but feature distributions drift between
windows -> MRMR returned stale selection forever after the first fit.

Effect: every pipeline that refits the same ``MRMR`` instance silently
broke. Includes sklearn CV with ``clone=False``, online retraining
loops, partial_fit-style patterns.

Severity: high (silent stale selection on every refit). ``MRMR.fit`` is
the documented sklearn surface; ``skip_retraining_on_same_shape=True``
is the default.

Fix: fold ``_full_x_content_hash(X)`` into the signature alongside
``_y_hash_for_sig`` so both layers agree.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def test_refit_on_different_x_content_updates_support():
    """The iter-36 contract: same shape + column names + y but
    different X content -> different support_.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 800
    y = (rng.standard_normal(n) > 0).astype(np.int64)
    # X1: 'a' is correlated with y, 'b' is noise.
    X1 = pd.DataFrame(
        {
            "a": y + 0.3 * rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
    # X2: same shape + column names but swap the correlated role.
    X2 = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": y + 0.3 * rng.standard_normal(n),
        }
    )
    y_s = pd.Series(y)
    m = MRMR(verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X1, y_s)
        names_1 = list(m.get_feature_names_out())
        m.fit(X2, y_s)
        names_2 = list(m.get_feature_names_out())
    assert names_1 != names_2, f"refit on different-content X (same shape+cols+y) silently replayed; both fits returned {names_1}"
    # X1 should pick 'a'; X2 should pick 'b' (the now-correlated feature).
    assert "a" in names_1
    assert "b" in names_2


def test_refit_on_same_x_y_skips_retraining():
    """Negative control: same X content + same y + same shape + same
    column names -> shortcut WILL skip the work and reproduce the
    prior fit (the documented optimisation).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 500
    y = (rng.standard_normal(n) > 0).astype(np.int64)
    X = pd.DataFrame(
        {
            "f0": rng.standard_normal(n),
            "f1": y + 0.3 * rng.standard_normal(n),
        }
    )
    y_s = pd.Series(y)
    m = MRMR(verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y_s)
        first_names = list(m.get_feature_names_out())
        # Same exact inputs -> identical result.
        m.fit(X, y_s)
        second_names = list(m.get_feature_names_out())
    assert first_names == second_names


def test_refit_on_different_y_same_x_updates_support():
    """Sanity: same X but different y MUST refit (already worked
    pre-fix because y_hash was already in the signature; verify the
    iter-36 fix didn't break this).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2)
    n = 500
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
    # y_a depends on 'a'; y_b depends on 'b'.
    y_a = pd.Series((X["a"] > 0).astype(np.int64))
    y_b = pd.Series((X["b"] > 0).astype(np.int64))
    m = MRMR(verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y_a)
        names_a = list(m.get_feature_names_out())
        m.fit(X, y_b)
        names_b = list(m.get_feature_names_out())
    assert names_a != names_b
