"""Device-born external-validation candidate materialise+discretise (_gpu_resident_extval) must produce codes
selection-equivalent to the host path (_materialise_extval_njit float64 + discretize_2d_quantile_batch) it
replaces -- so wiring it into _emit_pair_features keeps the ext-val MI sweep (and thus FE selection) unchanged
while the (n,K) candidate matrix stays off the H2D bus.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_extval import gpu_materialise_extval_codes_host
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise import (
    _materialise_extval_njit,
)
from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch


# Every registry-coded binary op (0..8). Includes div/ratio_abs (float64-promoted) + signed (sign*|b|).
_ALL_OPS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)


@pytest.mark.gpu
@pytest.mark.parametrize("nbins", [10, 16])
def test_device_extval_codes_match_host(nbins: int) -> None:
    """Device-resident ext-val materialise+discretise codes agree with the host njit+quantile-binner path on >=99.9% of rows across every registry op, including div-by-near-zero."""
    rng = np.random.default_rng(7)
    n = 6000
    # param_a spans zeros/negatives; ext factors include a near-zero-denominator column so div/ratio_abs
    # exercise the eps floor + inf/NaN -> rightmost-bin routing (the no-scrub contract).
    param_a = rng.standard_normal(n).astype(np.float64)
    param_a[::50] = 0.0
    ext0 = rng.standard_normal(n).astype(np.float64)
    ext1 = (rng.standard_normal(n) * 1e-7).astype(np.float64)  # near-zero denominators
    ext1[::37] = 0.0
    param_b_list = [ext0, ext1]

    # HOST reference: njit materialise (float64, no scrub) then the production batch quantile binner.
    n_ext, n_ops = len(param_b_list), _ALL_OPS.shape[0]
    ev_buf = np.empty((n, n_ext * n_ops), dtype=np.float64)
    pb_mat = np.empty((n, n_ext), dtype=np.float64)
    for i, pb in enumerate(param_b_list):
        pb_mat[:, i] = pb
    _materialise_extval_njit(np.ascontiguousarray(param_a), pb_mat, _ALL_OPS, ev_buf)
    host_codes = discretize_2d_quantile_batch(ev_buf, n_bins=nbins, dtype=np.int16)

    # DEVICE path under test.
    dev_codes = gpu_materialise_extval_codes_host(param_a, param_b_list, _ALL_OPS, nbins, dtype=np.int16)
    assert dev_codes is not None, "device ext-val materialise returned None on a CUDA host"
    assert dev_codes.shape == host_codes.shape

    # Selection-equivalence bar: the equi-frequency binner is rank-based, so f64 device ops vs the njit f64
    # buffer agree except for GPU FP reduction order in the percentile edges (a handful of edge-adjacent rows).
    agree = float(np.mean(dev_codes == host_codes))
    assert agree >= 0.999, f"device vs host ext-val code agreement {agree:.5f} < 0.999 (nbins={nbins})"


@pytest.mark.gpu
def test_device_extval_none_on_empty() -> None:
    """Degenerate inputs return None (caller keeps the host path), never raise."""
    assert gpu_materialise_extval_codes_host(np.zeros(0), [], _ALL_OPS, 10) is None
    assert gpu_materialise_extval_codes_host(np.zeros(100), [np.zeros(100)], np.array([], dtype=np.int8), 10) is None
