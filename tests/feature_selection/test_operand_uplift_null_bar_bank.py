"""The operand-uplift null bar with the shared permutation bank and the parallel MI kernel equals the serial reference exactly."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_operand_floor as F


def _codes(n, k, seed):
    """Dense ordinal codes with k levels."""
    return np.random.default_rng(seed).integers(0, k, size=n).astype(np.int64)


@pytest.mark.parametrize("n,ka,kb,ky,seed", [(5000, 10, 10, 2, 0), (3001, 7, 12, 3, 5), (800, 3, 3, 2, 1)])
def test_bar_equals_the_serial_reference(n, ka, kb, ky, seed):
    """Same bar to the last digit for different sizes, level counts, target classes and seeds."""
    a, b, y = _codes(n, ka, 1), _codes(n, kb, 2), _codes(n, ky, 3)
    floor = 0.013
    for q in (0.95, 0.5):
        ref = F._operand_uplift_null_bar_reference(a, b, y, floor, n_perm=20, quantile=q, seed=seed)
        got = F.operand_uplift_null_bar(a, b, y, floor, n_perm=20, quantile=q, seed=seed)
        assert got == ref


def test_the_bank_is_reused_across_pairs_of_the_same_length():
    """Two pairs with the same row count draw the permutations once; another seed or length is another entry."""
    F._PERM_BANK.clear()
    n = 2000
    a, b, y = _codes(n, 5, 1), _codes(n, 6, 2), _codes(n, 2, 3)
    F.operand_uplift_null_bar(a, b, y, 0.01, seed=0)
    first = F._PERM_BANK[(n, 20, 0)]
    F.operand_uplift_null_bar(b, a, y, 0.01, seed=0)
    assert F._PERM_BANK[(n, 20, 0)] is first and len(F._PERM_BANK) == 1
    F.operand_uplift_null_bar(a, b, y, 0.01, seed=1)
    assert len(F._PERM_BANK) == 2


def test_degenerate_inputs_keep_their_contract():
    """Missing codes or a non-positive floor give 1.0; mismatched lengths raise as before."""
    a, b, y = _codes(100, 4, 1), _codes(100, 4, 2), _codes(100, 2, 3)
    assert F.operand_uplift_null_bar(None, b, y, 0.1) == 1.0 and F.operand_uplift_null_bar(a, b, y, 0.0) == 1.0
    with pytest.raises(ValueError):
        F.operand_uplift_null_bar(a, b[:50], y, 0.1)
