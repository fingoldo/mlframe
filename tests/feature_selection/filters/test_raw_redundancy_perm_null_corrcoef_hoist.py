"""Regression: the permutation-null in ``raw_retains_linear_signal_given_children``
scores ``abs(corr(perm, ry))`` via a hoisted dot product instead of a per-iteration
``np.corrcoef`` 2x2 rebuild. ``perm`` is a reordering of the raw residual (mean/std
invariant) and ``ry`` is fixed, so the only varying term is the cross dot product;
the result must stay bit-identical (FP reduction order, ~1e-17) to the legacy
``np.corrcoef`` formulation, and the keep/drop verdict must not move.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
    _heldout_ridge_r2,
    raw_retains_linear_signal_given_children,
)


def _legacy_null(rx: np.ndarray, ry: np.ndarray, nperm: int, seed: int) -> np.ndarray:
    """The pre-hoist inner loop, verbatim, as the identity oracle."""
    rng = np.random.default_rng(seed)
    null = np.empty(int(nperm), dtype=np.float64)
    for k in range(int(nperm)):
        perm = rng.permutation(rx)
        null[k] = abs(float(np.corrcoef(perm, ry)[0, 1]))
    return null


def _hoisted_null(rx: np.ndarray, ry: np.ndarray, nperm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rx.shape[0]
    ryc = ry - ry.mean()
    denom = n * float(np.std(rx)) * float(np.std(ry))
    null = np.empty(int(nperm), dtype=np.float64)
    for k in range(int(nperm)):
        perm = rng.permutation(rx)
        null[k] = abs(float(perm @ ryc) / denom) if denom > 0.0 else 0.0
    return null


@pytest.mark.parametrize("n", [200, 2000, 10000])
def test_hoisted_perm_null_bit_identical_to_corrcoef(n: int) -> None:
    rng = np.random.default_rng(7)
    rx = rng.standard_normal(n)
    ry = 0.3 * rx + rng.standard_normal(n)  # mild correlation so |corr| is non-trivial
    legacy = _legacy_null(rx, ry, nperm=32, seed=909)
    hoisted = _hoisted_null(rx, ry, nperm=32, seed=909)
    assert np.max(np.abs(legacy - hoisted)) < 1e-12


def test_keep_verdict_unchanged_with_private_linear_signal() -> None:
    """A raw with genuine private linear signal toward y, not reproduced by a
    non-monotone child, must be KEPT (the hoist preserves the decision)."""
    rng = np.random.default_rng(0)
    n = 4000
    raw = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    y = 2.0 * raw + 0.5 * noise  # raw carries strong private linear signal
    child = raw * raw + rng.standard_normal(n) * 0.1  # non-monotone -> not a linear equivalent
    assert raw_retains_linear_signal_given_children(raw, y, [child], seed=909) is True


def test_drop_verdict_unchanged_when_child_is_linear_equivalent() -> None:
    """A raw whose linear signal is fully reproduced by a linear child carries no
    private residual -> NOT retained."""
    rng = np.random.default_rng(1)
    n = 4000
    raw = rng.standard_normal(n)
    child = raw + rng.standard_normal(n) * 1e-6  # child IS raw linearly
    y = 3.0 * child + rng.standard_normal(n) * 0.01
    assert raw_retains_linear_signal_given_children(raw, y, [child], seed=909) is False


def test_heldout_ridge_r2_separates_good_from_lossy_feature_set() -> None:
    """The downstream no-harm guard reverts a raw-redundancy DROP when the KEPT set's held-out Ridge R^2 falls
    below the raw-only baseline. This pins the guard's core discriminator: the held-out Ridge probe must score
    a linearly-faithful raw set materially ABOVE a linearly-lossy engineered replacement. ``b**3`` is monotone
    in ``b`` (so the rank-MI/CMI drop verdict deems it a subsumer) but is NOT a linear equivalent -- exactly the
    prewarp/product entanglement that harmed I4b/I5 on lognormal terrain."""
    rng = np.random.default_rng(3)
    n = 4000
    b = rng.standard_normal(n)
    a = rng.standard_normal(n)
    y = 2.0 * b + 0.3 * a  # linear in b (and a)
    r_raw = _heldout_ridge_r2(np.column_stack([a, b]), y)  # raw-only baseline: recovers y
    r_lossy = _heldout_ridge_r2(np.column_stack([a, b**3]), y)  # linearly-lossy replacement of b
    assert r_raw is not None and r_lossy is not None
    assert r_raw > 0.9
    # The lossy set is materially worse -> a drop that replaced raw b with b**3 would fall below raw-only by
    # more than the guard's epsilon and be reverted.
    assert r_lossy < r_raw - 0.05
    # Degenerate inputs return None (guard leaves the drop unchanged): too few rows, constant y.
    assert _heldout_ridge_r2(np.column_stack([a[:10], b[:10]]), y[:10]) is None
    assert _heldout_ridge_r2(np.column_stack([a, b]), np.zeros(n)) is None
