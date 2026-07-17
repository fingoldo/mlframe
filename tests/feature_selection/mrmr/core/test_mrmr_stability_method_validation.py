"""Regression test: an invalid ``stability_selection_method`` must raise cleanly, with no
spurious "falling back to classic fit" warning first.

Before the fix, ``fit()`` wrapped the whole stability-selection outer loop in a bare
``except Exception``, so ``_stability_outer_fit``'s own ``ValueError(f"unknown
stability_selection_method={method!r}")`` was caught and turned into a ``UserWarning``
("... Falling back to classic fit.") before the fit fell through to the legacy classic body,
which happened to raise its OWN ``ValueError`` for the same bad value via the generic
``_validate_string_params`` enum check. So a typo already raised eventually, but only after a
confusing, misleading warning claiming the fit would proceed as classic MRMR (it did not --
it raised). A genuine transient failure inside the cluster/complementary-pairs machinery (a
VALID method name whose runtime implementation errors) is not caught by that downstream enum
check at all and is still silently swallowed into a classic-MRMR fallback -- this test only
covers the "typo in the method name" half of the bug, which is what the fix addresses.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.feature_selection.filters import MRMR


def _xy(seed: int = 5, n: int = 160):
    """Build a small synthetic classification fixture with signal on columns 0 and 2."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.4 * X[:, 2] > 0).astype(np.int32)
    return X, y


def _fast(**kw):
    """Build a fast-fitting MRMR instance for these tests, overridable via kwargs."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def test_invalid_stability_selection_method_raises_without_spurious_fallback_warning():
    """An unrecognised ``stability_selection_method`` must raise ValueError, with no
    misleading "falling back to classic fit" warning emitted first."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(stability_selection_method="not_a_real_method")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="stability_selection_method"):
            m.fit(X, y)
    fallback_warnings = [w for w in caught if "Falling back to classic fit" in str(w.message)]
    assert not fallback_warnings, "invalid stability_selection_method must raise immediately, not warn-then-fallback-then-raise"


def test_valid_stability_selection_methods_do_not_raise():
    """Every genuinely valid ``stability_selection_method`` value must fit without raising."""
    X, y = _xy()
    for method in ("classic", "cluster", "complementary_pairs"):
        MRMR._FIT_CACHE.clear()
        m = _fast(stability_selection_method=method, stability_n_bootstrap=5)
        m.fit(X, y)
        assert np.asarray(m.support_).size >= 0
