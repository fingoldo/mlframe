"""The device twin of the SU quantiser gives the host codes and bin counts column for column, including the one-class rule (MRMR-QUANTIZE-SU, 2026-11-18)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._cluster_hierarchy import _su_codes_device, _su_codes_of_finite_columns, _su_code_column
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


def _host(col: np.ndarray, n_bins: int):
    """The host reference: the SU quantiser's codes and bin count of one column."""
    got = _su_codes_of_finite_columns([col], n_bins)[0]
    return got if got is not None else _su_code_column(col, n_bins)


def test_device_twin_matches_the_host_quantiser_per_column():
    """Each column of the device twin has the host codes and bin count, including the one-class rule."""
    rng = np.random.default_rng(7)
    n = 6000
    skewed = np.zeros(n)
    skewed[: n // 100] = 1.0
    skewed[n // 100 : n // 50] = 2.0
    cols = [
        rng.standard_normal(n),
        rng.integers(0, 2, n).astype(np.float64),
        rng.integers(0, 3, n).astype(np.float64),
        np.full(n, 3.0),
        skewed,
    ]
    X = np.column_stack(cols).astype(_crit_np_dtype())
    codes_dev, nb_dev = _su_codes_device(cp.asarray(X), 10)
    codes = cp.asnumpy(codes_dev)
    nb = cp.asnumpy(nb_dev)
    for j in range(X.shape[1]):
        ref_codes, ref_nb = _host(X[:, j], 10)
        assert int(nb[j]) == int(ref_nb), f"column {j}: bins {int(nb[j])} != {int(ref_nb)}"
        np.testing.assert_array_equal(codes[:, j], np.asarray(ref_codes).ravel(), err_msg=f"column {j}")


def test_quantize_for_su_device_route_equals_the_cpu_route(monkeypatch):
    """The whole SU quantiser gives the same code matrix and bin counts on the device route as on the CPU route (frame with mixed column kinds)."""
    import pandas as pd

    import mlframe.feature_selection.filters._cluster_hierarchy as H

    rng = np.random.default_rng(11)
    n = 6000
    skewed = np.zeros(n)
    skewed[: n // 100] = 1.0
    frame = pd.DataFrame({
        "cont": rng.standard_normal(n),
        "bin": rng.integers(0, 2, n).astype(np.float64),
        "tri": rng.integers(0, 3, n).astype(np.float64),
        "const": np.full(n, 3.0),
        "skew": skewed,
        "cont2": rng.standard_normal(n) * 5.0 + 1.0,
    })
    monkeypatch.setattr(H, "_su_device_enabled", lambda n_, p_: True)
    dev_codes, dev_nb, _ = H._quantize_for_su(frame)
    monkeypatch.setattr(H, "_su_device_enabled", lambda n_, p_: False)
    cpu_codes, cpu_nb, _ = H._quantize_for_su(frame)
    np.testing.assert_array_equal(np.asarray(dev_nb), np.asarray(cpu_nb))
    np.testing.assert_array_equal(np.asarray(dev_codes), np.asarray(cpu_codes))
