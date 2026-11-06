"""``stable_argsort_codes`` orders integer codes exactly as ``np.argsort(kind="stable")`` does, whatever type it sorts through."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._internals import stable_argsort_codes


@pytest.mark.parametrize("n_codes", [1, 2, 5, 256, 257, 5000, 65536, 70000])
def test_same_order_as_the_stable_argsort(n_codes):
    """Equal keys keep their input order, so the permutation is the unique stable one for every width the helper picks."""
    rng = np.random.default_rng(n_codes)
    codes = rng.integers(0, n_codes, 20000).astype(np.int64)
    assert np.array_equal(stable_argsort_codes(codes, n_codes), np.argsort(codes, kind="stable"))


def test_empty_and_single_group_inputs():
    """No rows give no permutation; one group keeps the input order."""
    assert stable_argsort_codes(np.empty(0, dtype=np.int64), 3).size == 0
    assert stable_argsort_codes(np.zeros(7, dtype=np.int64), 1).tolist() == list(range(7))
