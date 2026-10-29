"""Quantile codes of a lazy device array are computed once: columns binned one by one spare the later block call its own transfer, and the result equals a fresh computation."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit
from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes
from mlframe.feature_selection.filters._mrmr_fe_step._step_score_parts2 import _device_codes_to_host

_N = 9000


def _lazy(seed: int = 0, k: int = 3) -> LazyHostCodes:
    """A lazy (n, k) float block on the device."""
    return LazyHostCodes(cp.asarray(np.random.default_rng(seed).standard_normal((_N, k))), np.float64)


def test_block_codes_after_column_codes_cost_no_transfer_and_equal_a_fresh_block():
    """Binning the columns of a block one by one, then the block, moves the codes once; the block codes equal those of an independent copy of the data."""
    block = _lazy()
    with residency_audit() as rep:
        cols = [_device_codes_to_host(block[:, j : j + 1], 10, np.int64) for j in range(3)]
        n_after_columns = len(rep.bulk_d2h)
        whole = _device_codes_to_host(block, 10, np.int64)
        n_after_block = len(rep.bulk_d2h)
    assert n_after_columns == 3 and n_after_block == 3
    np.testing.assert_array_equal(whole, np.concatenate(cols, axis=1))
    np.testing.assert_array_equal(whole, _device_codes_to_host(_lazy(), 10, np.int64))
    assert whole.dtype == np.int64 and whole.shape == (_N, 3)


def test_a_partial_set_of_columns_is_not_enough_and_the_cache_hands_out_copies():
    """With one column of three known the block is binned in full; a returned array can be edited without touching the cache."""
    block = _lazy(seed=1)
    _device_codes_to_host(block[:, 0:1], 10, np.int64)
    assert block.codes_cached(10, np.int64) is None
    whole = _device_codes_to_host(block, 10, np.int64)
    again = block.codes_cached(10, np.int64)
    np.testing.assert_array_equal(again, whole)
    whole[:] = -1
    np.testing.assert_array_equal(block.codes_cached(10, np.int64), again)


def test_other_bin_counts_and_dtypes_do_not_share_an_entry():
    """Codes at another ``nbins`` or code dtype are not served from the cache."""
    block = _lazy(seed=2, k=1)
    ten = _device_codes_to_host(block, 10, np.int64)
    assert block.codes_cached(5, np.int64) is None
    five = _device_codes_to_host(block, 5, np.int64)
    assert ten.max() > five.max()
