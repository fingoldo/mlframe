"""Parallel numba kernels on the startup path load from the on-disk cache instead of compiling in every process.

A production run spent 58 s of every start compiling ``metric_kernels``; most of it was one parallel calibration
kernel that numba refused to cache because it called ``numba.get_num_threads()`` inside ("uses dynamic globals"),
11-12 s per signature. The mutual-information kernels were marked ``cache=False`` for no reason and cost 14.5 s per
process where the cache loads them in 0.7 s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


def _compile_without_cache_refusal(kernel, *args):
    """Call ``kernel`` and return the numba warnings that refused to cache it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernel(*args)
    return [str(w.message) for w in caught if "Cannot cache" in str(w.message)]


def test_the_parallel_calibration_kernel_is_cacheable():
    """The defect: the thread count read inside the kernel made numba refuse to cache it."""
    import numba

    from mlframe.metrics.calibration._calibration_plot import _fast_calibration_binning_prange_kernel as kernel

    y = (np.random.default_rng(0).random(5000) > 0.5).astype(np.int64)
    p = np.random.default_rng(1).random(5000)
    assert not _compile_without_cache_refusal(kernel, y, p, 20, numba.get_num_threads())
    assert type(kernel._cache).__name__ == "FunctionCache"


def test_the_wrapper_matches_the_serial_kernel():
    """Moving the thread count out of the kernel must not change what it computes."""
    from mlframe.metrics.calibration._calibration_plot import _fast_calibration_binning_prange, _fast_calibration_binning_serial

    y = (np.random.default_rng(2).random(20_000) > 0.5).astype(np.int64)
    p = np.random.default_rng(3).random(20_000)
    fp_p, ft_p, h_p = _fast_calibration_binning_prange(y, p, 30)
    fp_s, ft_s, h_s = _fast_calibration_binning_serial(y, p, 30)
    np.testing.assert_array_equal(h_p, h_s)
    np.testing.assert_array_equal(ft_p, ft_s)
    np.testing.assert_allclose(fp_p, fp_s, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["_grok_compute_mutual_information_kernel", "_chatgpt_mi_one_target", "_deepseek_compute_mutual_information_kernel"])
def test_the_mutual_information_kernels_are_cacheable(name):
    """They were marked cache=False although nothing in them prevents caching."""
    from mlframe.feature_selection import mi

    kernel = getattr(mi, name)
    assert kernel.targetoptions.get("parallel"), "the point is the parallel kernels"
    assert type(kernel._cache).__name__ == "FunctionCache"
