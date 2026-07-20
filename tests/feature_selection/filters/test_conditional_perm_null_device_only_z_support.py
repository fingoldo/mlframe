"""Regression: _conditional_perm_null crashed with TypeError when z_support was None but
z_support_dev was a resident device array and the GPU-materializing branch was skipped
(2026-07-20, found live via wellbore-100k additive-fusion crash: "TypeError: int() argument must
be a string, a bytes-like object or a real number, not 'NoneType'" at
z = np.ascontiguousarray(z_support, dtype=np.int64).ravel()).

Root cause: the only code path that materializes z_support from z_support_dev lives inside the
``if z_support_dev is not None and ... and n_permutations > 1:`` GPU-attempt block. When
n_permutations <= 1 (or _cmi_gpu_enabled is False), that block -- and its materialization -- is
skipped entirely, and the marginal-null branch right below it only fires when z_support_dev is
also None. So z_support=None + z_support_dev=<resident array> + n_permutations<=1 fell through
unguarded to the unconditional ascontiguousarray call.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _conditional_perm_null


def test_device_only_z_support_with_single_permutation_does_not_crash():
    """z_support=None + a resident z_support_dev + n_permutations=1 (skips the GPU-materializing
    block's >1 guard) must not raise TypeError -- it should fall through to the host stratified
    permutation null using the device-materialized z, same as when z_support is passed directly."""
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(0)
    n = 2000
    cand_bin = rng.integers(0, 4, size=n).astype(np.int64)
    y_bin = rng.integers(0, 3, size=n).astype(np.int64)
    z_host = rng.integers(0, 5, size=n).astype(np.int64)
    z_dev = cp.asarray(z_host)

    floor, mean = _conditional_perm_null(
        cand_bin, y_bin, None, n_permutations=1, seed=0, salt=0, z_support_dev=z_dev,
    )
    assert np.isfinite(floor)
    assert np.isfinite(mean)


def test_device_only_z_support_matches_host_z_support_result():
    """Selection-relevant sanity check: the device-materialized path and passing the identical
    z_support directly on the host must produce the SAME (floor, mean) -- same seed/salt, same
    underlying z values, only the materialization route differs."""
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(1)
    n = 2000
    cand_bin = rng.integers(0, 4, size=n).astype(np.int64)
    y_bin = rng.integers(0, 3, size=n).astype(np.int64)
    z_host = rng.integers(0, 5, size=n).astype(np.int64)
    z_dev = cp.asarray(z_host)

    floor_dev, mean_dev = _conditional_perm_null(
        cand_bin, y_bin, None, n_permutations=1, seed=7, salt=3, z_support_dev=z_dev,
    )
    floor_host, mean_host = _conditional_perm_null(
        cand_bin, y_bin, z_host, n_permutations=1, seed=7, salt=3,
    )
    assert floor_dev == floor_host
    assert mean_dev == mean_host
