"""Bit-identity regression for the fused conditional_mi melt kernels.

``conditional_mi`` melts X onto (Y,Z) twice per candidate -- H(X,Z) and H(X,Y,Z) -- and both callers
discard ``merge_vars``' ``final_classes`` relabel output, needing only the pruned freqs for ``entropy``.
The fused kernels ``_entropy_xz_fused`` / ``_entropy_x_onto_classes`` skip that discarded work. They MUST
stay byte-for-byte identical to ``entropy(merge_vars(...)[1])`` (the hoist in ``_cmi_cuda.py`` relies on the
merge ORDER, so the produced freqs order/values may not drift). These tests FAIL pre-fix: the fused kernels
did not exist before this change.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import unpack_and_sort
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars
from mlframe.feature_selection.filters.info_theory._entropy_kernels import (
    entropy,
    conditional_mi,
    _entropy_xz_fused,
    _entropy_x_onto_classes,
)


def _make(n, nbins, ncols, seed):
    """Helper that make."""
    rng = np.random.default_rng(seed)
    data = np.empty((n, ncols), dtype=np.int32)
    for c in range(ncols):
        # mix uniform / skewed columns so pruning of empty bins actually happens
        if c % 2 == 0:
            data[:, c] = rng.integers(0, nbins, size=n, dtype=np.int32)
        else:
            data[:, c] = (rng.integers(0, max(2, nbins // 2), size=n) ** 2 % nbins).astype(np.int32)
    factors_nbins = np.full(ncols, nbins, dtype=np.int64)
    return data, factors_nbins


@pytest.mark.parametrize("n", [37, 600, 5000, 50_000])
@pytest.mark.parametrize("nbins", [4, 10, 16])
@pytest.mark.parametrize("z_ncols", [1, 2, 3])
def test_entropy_xz_fused_bit_identical(n, nbins, z_ncols):
    """Entropy xz fused bit identical."""
    ncols = 2 + z_ncols
    data, factors_nbins = _make(n, nbins, ncols, seed=n + nbins + z_ncols)
    x = np.array([0], dtype=np.int64)
    z = np.arange(2, 2 + z_ncols, dtype=np.int64)
    indices = unpack_and_sort(x, z)

    _, freqs, _ = merge_vars(data, indices, None, factors_nbins, dtype=np.int32)
    ref = entropy(freqs)
    got = _entropy_xz_fused(data, indices, factors_nbins, np.int32)
    assert got == ref, f"maxabsdiff={abs(got - ref):.3e} (n={n} nbins={nbins} z_ncols={z_ncols})"


@pytest.mark.parametrize("n", [37, 600, 5000, 50_000])
@pytest.mark.parametrize("nbins", [4, 10, 16])
@pytest.mark.parametrize("z_ncols", [1, 2])
def test_entropy_x_onto_classes_bit_identical(n, nbins, z_ncols):
    """Entropy x onto classes bit identical."""
    ncols = 2 + z_ncols
    data, factors_nbins = _make(n, nbins, ncols, seed=n * 3 + nbins + z_ncols)
    x = np.array([0], dtype=np.int64)
    y = np.array([1], dtype=np.int64)
    z = np.arange(2, 2 + z_ncols, dtype=np.int64)
    yz = unpack_and_sort(y, z)
    classes_yz, _, ncls_yz = merge_vars(data, yz, None, factors_nbins, dtype=np.int32)

    # reference (mutates a COPY, as merge_vars overwrites final_classes in place)
    _, freqs_ref, _ = merge_vars(
        data,
        x,
        None,
        factors_nbins,
        current_nclasses=ncls_yz,
        final_classes=classes_yz.copy(),
        dtype=np.int32,
    )
    ref = entropy(freqs_ref)
    got = _entropy_x_onto_classes(data, int(x[0]), classes_yz, ncls_yz, int(factors_nbins[x[0]]))
    assert got == ref, f"maxabsdiff={abs(got - ref):.3e} (n={n} nbins={nbins} z_ncols={z_ncols})"
    # and the fused kernel must NOT mutate the shared classes_yz array
    cyz2, _, _ = merge_vars(data, yz, None, factors_nbins, dtype=np.int32)
    assert np.array_equal(classes_yz, cyz2), "fused kernel mutated classes_yz (must be read-only)"


@pytest.mark.parametrize("n", [600, 5000, 50_000])
@pytest.mark.parametrize("nbins", [4, 16])
def test_conditional_mi_end_to_end_nonnegative_and_stable(n, nbins):
    """conditional_mi wired to the fused kernels: sanity (>=0) + reproducible across dtypes."""
    data, factors_nbins = _make(n, nbins, 4, seed=7 * n + nbins)
    var_is_nominal = np.zeros(4, dtype=np.int64)
    x = np.array([0], dtype=np.int64)
    y = np.array([1], dtype=np.int64)
    z = np.array([2], dtype=np.int64)
    v32 = conditional_mi(data, x, y, z, var_is_nominal, factors_nbins, dtype=np.int32)
    v64 = conditional_mi(data, x, y, z, var_is_nominal, factors_nbins, dtype=np.int64)
    assert v32 >= 0.0
    assert v32 == pytest.approx(v64, abs=1e-9)
