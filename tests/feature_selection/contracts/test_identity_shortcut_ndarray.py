"""Wave 9.1 loop-iter-35 regression: ``_fit_identity_shortcut`` must
handle ndarray X without crashing.

Pre-fix at ``mrmr.py:1187-1190``::

    self.feature_names_in_ = (
        X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
        if hasattr(X, "columns") else [f"f{i}" for i in range(n_cols)]
    )

Python parses this ternary as ``A if B1 else (C if B2 else E)`` -
``B1 = hasattr(X.columns, "tolist")`` is evaluated BEFORE the outer
``B2 = hasattr(X, "columns")`` guard, so the inner ``X.columns`` access
raised ``AttributeError`` on ndarray X. The identity-shortcut cache-hit
path (opt-in via ``mrmr_skip_when_prior_was_identity=True``) crashed on
every ndarray fit instead of short-circuiting.

Effect: the documented "skip FE pipeline when prior fit was identity"
optimisation was unreachable for the sklearn-canonical ndarray input
path. Only DataFrame X reached the shortcut at all, and even there the
fallback ``list(X.columns)`` branch was dead code.

Severity: medium-high. Opt-in feature fully broken for ndarray; affects
caching/perf path (not numerical correctness) but the public knob was
silently no-op'd for the default sklearn input form.

Fix: split into a guarded ``if hasattr(X, "columns"):`` block so the
column access happens only after the existence check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_identity_shortcut_ndarray_no_attribute_error():
    """Direct call into ``_fit_identity_shortcut`` with ndarray X must
    succeed. Pre-fix this raised AttributeError on ``X.columns``.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    sel = MRMR(verbose=0)
    # Direct invocation (bypasses cache-hit detection so we exercise
    # the shortcut path itself).
    sel._fit_identity_shortcut(X)
    assert sel.support_.tolist() == list(range(4))
    # feature_names_in_ is an ndarray (sklearn's own convention, BaseEstimator._check_feature_names) --
    # compare via .tolist(), not a bare ``==`` against a list (which broadcasts elementwise and raises
    # "truth value of an array... is ambiguous" under assert).
    assert list(sel.feature_names_in_) == [f"f{i}" for i in range(4)]
    assert sel.n_features_in_ == 4


def test_identity_shortcut_dataframe_preserves_column_names():
    """Negative control: DataFrame X path still uses real column names."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        rng.standard_normal((200, 4)),
        columns=["alpha", "beta", "gamma", "delta"],
    )
    sel = MRMR(verbose=0)
    sel._fit_identity_shortcut(X)
    assert list(sel.feature_names_in_) == ["alpha", "beta", "gamma", "delta"]


def test_identity_shortcut_ndarray_via_cache_hit_completes_without_crash():
    """End-to-end: priming the identity fingerprint cache and then
    fitting an ndarray with ``mrmr_skip_when_prior_was_identity=True``
    must NOT raise AttributeError. Pre-fix this crashed; post-fix it
    either short-circuits (depending on cache state) or falls through
    to the regular fit. Either is acceptable - the test pins only
    that no exception is raised.
    """
    from mlframe.feature_selection.filters.mrmr import (
        MRMR,
        _MRMR_IDENTITY_FP_CACHE,
    )
    from mlframe.feature_selection.filters._mrmr_fingerprints import (
        _mrmr_compute_x_fingerprint,
    )

    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 4))
    y = pd.Series(rng.integers(0, 2, 200))
    _MRMR_IDENTITY_FP_CACHE[_mrmr_compute_x_fingerprint(X)] = True
    sel = MRMR(mrmr_skip_when_prior_was_identity=True, verbose=0)
    # Pre-fix: AttributeError. Post-fix: completes (with or without
    # shortcut firing).
    sel.fit(X, y)
    assert hasattr(sel, "support_")
