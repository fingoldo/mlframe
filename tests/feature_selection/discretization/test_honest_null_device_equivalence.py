"""The device root-split pipeline of the honest null returns the matrices of the host column-parallel kernel."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._analytic_mi_null import _min_expected_cell, analytic_null_min_n
from mlframe.feature_selection.filters._honest_null_device import eligible, honest_null_columns_device
from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import honest_null_columns_bycol, permuted_labels


def _gpu():
    """True when a CUDA device answers."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="needs a CUDA device")


def _inputs(n, n_classes, n_perm, seed):
    """Mixed pool: continuous columns (one informative, one with heavy ties), fixed-code columns, a row subsample for the MI."""
    rng = np.random.default_rng(seed)
    k_cont, k_fixed = 6, 2
    raw = rng.normal(size=(n, k_cont))
    raw[:, 0] = np.round(raw[:, 0] * 3)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    raw[:, 1] += 0.35 * (y == 1)
    k = k_cont + k_fixed
    fixed = rng.integers(0, 4, (n, k_fixed)).astype(np.int8)
    is_cont = np.array([True] * k_cont + [False] * k_fixed)
    slot = np.array(list(range(k_cont)) + list(range(k_fixed)), dtype=np.int64)
    qf = np.zeros((k, 3))
    qf[:k_cont] = np.quantile(raw, [0.25, 0.5, 0.75], axis=0).T
    perms = (rng.permutation(n) for _ in range(n_perm))
    y_raw_perm, y_codes_perm = permuted_labels(y, y, perms, n_perm)
    ev = np.sort(rng.choice(n, n * 3 // 5, replace=False)).astype(np.int64)
    return (
        raw,
        is_cont,
        fixed,
        np.full(k, 4, dtype=np.int64),
        y_raw_perm,
        y_codes_perm,
        n_classes,
        20,
        8,
        0.05,
        100,
        0,
        int(analytic_null_min_n()),
        float(_min_expected_cell()),
        qf,
        np.full(k, 3, dtype=np.int64),
        qf.copy(),
        np.full(k, 3, dtype=np.int64),
        ev,
        slot,
    )


@pytest.mark.parametrize("n_classes", [2, 3])
def test_device_pipeline_matches_the_host_kernel(n_classes):
    """Plain and corrected null MI agree for every (permutation, column), including the pairs whose root split is accepted."""
    args = _inputs(30000, n_classes, 24, seed=n_classes)
    assert eligible(30000, n_classes, args[12], args[13])
    host = honest_null_columns_bycol(*args)
    dev = honest_null_columns_device(*args)
    assert dev is not None
    assert len(host) == len(dev) == 2, "plain and corrected null MI must both be compared"
    for a, b in zip(host, dev):
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_small_samples_stay_on_the_host_kernel():
    """Below the analytic regime (permutation-decided roots) the device pipeline declines."""
    args = _inputs(3000, 2, 4, seed=5)
    assert honest_null_columns_device(*args) is None
