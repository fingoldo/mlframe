"""Regression: the marginal-seed permutation null in ``_conditional_perm_null`` must stay
BIT-IDENTICAL after hoisting the fixed-y block (precompute_marginal_y_terms +
marginal_mi_binned_fixed_y) out of the per-perm loop.

Guards the perf hoist (2026-07-05): if a future edit drifts ``marginal_mi_binned_fixed_y`` from
the reference ``_cmi_from_binned(x_perm, y, None)`` path, the floor/null-mean the significance gate
compares against would silently change. Pins exact equality against a brute-force reference loop.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reference_marginal_null(x, y, *, n_permutations=25, quantile=0.95, seed=0, salt=0):
    """The pre-hoist reference: recompute the full marginal MI (re-binning y each perm)."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned

    _xh = np.ascontiguousarray(x, dtype=np.int64).ravel()
    y_i = np.ascontiguousarray(y, dtype=np.int64).ravel()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(salt) & 0xFFFFFFFF]))
    nulls = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        x_perm = _xh[rng.permutation(_xh.size)]
        nulls[i] = float(_cmi_from_binned(x_perm, y_i, None))
    return float(np.quantile(nulls, quantile)), float(np.mean(nulls))


@pytest.mark.parametrize(
    "n,kx,ky,seed,salt",
    [
        (2000, 8, 4, 0, 0),
        (2000, 16, 6, 3, 11),
        (5000, 12, 5, 7, 2),
    ],
)
def test_marginal_perm_null_bit_identical_to_reference(monkeypatch, n, kx, ky, seed, salt):
    # Force the CPU host path (no analytic null at this n; no GPU-resident branch).
    """Marginal perm null bit identical to reference."""
    monkeypatch.setenv("MLFRAME_CMI_GPU", "0")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    monkeypatch.setenv("MLFRAME_FE_CMI_PERM_NULL_GPU", "0")
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _conditional_perm_null

    r = np.random.default_rng(seed * 100 + salt)
    x = r.integers(0, kx, n).astype(np.int64)
    y = r.integers(0, ky, n).astype(np.int64)

    floor, mean = _conditional_perm_null(x, y, None, seed=seed, salt=salt)
    ref_floor, ref_mean = _reference_marginal_null(x, y, seed=seed, salt=salt)

    assert floor == ref_floor, f"marginal floor drifted: {floor!r} != {ref_floor!r}"
    assert mean == ref_mean, f"marginal null-mean drifted: {mean!r} != {ref_mean!r}"
