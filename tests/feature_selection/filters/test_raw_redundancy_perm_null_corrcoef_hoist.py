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
    y = 2.0 * raw + 0.5 * noise          # raw carries strong private linear signal
    child = raw * raw + rng.standard_normal(n) * 0.1   # non-monotone -> not a linear equivalent
    assert raw_retains_linear_signal_given_children(raw, y, [child], seed=909) is True


def test_drop_verdict_unchanged_when_child_is_linear_equivalent() -> None:
    """A raw whose linear signal is fully reproduced by a linear child carries no
    private residual -> NOT retained."""
    rng = np.random.default_rng(1)
    n = 4000
    raw = rng.standard_normal(n)
    child = raw + rng.standard_normal(n) * 1e-6   # child IS raw linearly
    y = 3.0 * child + rng.standard_normal(n) * 0.01
    assert raw_retains_linear_signal_given_children(raw, y, [child], seed=909) is False
