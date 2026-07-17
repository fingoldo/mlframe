"""Regression pin for the 2026-07-03 dead-analytic-null bug.

The conditional/marginal permutation-null fallback (``_fe_cmi_redundancy_null``) and the CMI redundancy gate
both do ``from ._analytic_mi_null import _HAVE_CHI2, _chi2, _min_expected_cell, analytic_null_enabled`` under a
bare ``except: _HAVE_CHI2 = False``. A concurrent gammaincc refactor renamed ``_chi2`` -> ``_chi2_sf`` and
DROPPED the ``_chi2`` symbol, so that import raised ImportError, the except silently disabled the analytic
null, and EVERY conditional/marginal null (~44 per F2 fit, ~3s) fell to the slow permutation path. These pins
assert (1) ``_chi2`` is importable with a working ``.ppf`` and (2) the analytic path is actually LIVE end to
end: a large, dense (non-sparse) conditional null must NOT reach the GPU permutation kernel.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_chi2_symbol_is_exported_with_ppf():
    from mlframe.feature_selection.filters._analytic_mi_null import _HAVE_CHI2, _chi2

    assert _HAVE_CHI2 is True, "scipy present -> analytic null must be enabled"
    assert _chi2 is not None, "_chi2 must be exported (both null + gate import it for the .ppf floor)"
    # chi2.ppf(0.95, df=4) ~= 9.4877; the floor path calls exactly this.
    assert abs(float(_chi2.ppf(0.95, 4)) - 9.4877) < 1e-3


def test_consumer_import_does_not_disable_analytic():
    """The exact import both consumers perform must succeed (else their local _HAVE_CHI2 is forced False)."""
    from mlframe.feature_selection.filters._analytic_mi_null import (  # noqa: F401
        _HAVE_CHI2,
        _chi2,
        _min_expected_cell,
        analytic_null_enabled,
    )

    assert _HAVE_CHI2 and analytic_null_enabled()


def test_dense_conditional_null_takes_analytic_not_perm(monkeypatch):
    """A large, DENSE conditional null (avg rows/cell far above the sparsity floor) must resolve via the
    analytic chi-square path and NOT invoke the GPU permutation kernel -- the direct behavioural guard that the
    analytic fast-path is live (the bug sent every such call to permute)."""
    import mlframe.feature_selection.filters._fe_cmi_perm_null_gpu as PG
    import mlframe.feature_selection.filters._fe_cmi_redundancy_null as N

    perm_hits = {"n": 0}
    orig = PG.conditional_perm_null_gpu

    def spy(*a, **k):
        perm_hits["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(PG, "conditional_perm_null_gpu", spy)
    monkeypatch.setattr(N, "conditional_perm_null_gpu", spy, raising=False)

    rng = np.random.default_rng(0)
    n = 40_000  # >> analytic min-n (25k)
    x = rng.integers(0, 8, n).astype(np.int64)
    y = rng.integers(0, 8, n).astype(np.int64)
    z = rng.integers(0, 6, n).astype(np.int64)  # k_xyz <= 8*8*6=384; n/cells ~ 104 >> 5 -> dense -> analytic
    floor, null_mean = N._conditional_perm_null(x, y, z, seed=0)
    assert np.isfinite(floor) and np.isfinite(null_mean)
    assert perm_hits["n"] == 0, "dense conditional null hit the GPU permutation kernel -> analytic path is dead"
