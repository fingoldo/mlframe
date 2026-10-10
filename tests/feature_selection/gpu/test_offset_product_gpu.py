"""The fused CUDA scan of the offset-product family agrees with the CPU kernel and selects the same candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _offset_product_fe as fe
from mlframe.feature_selection.filters._offset_product_gpu import scan_offset_products_gpu
from mlframe.feature_selection.filters._offset_product_kernels import N_BASELINES, scan_offset_products

N_BINS = 10


def _inputs(n: int, seed: int = 0):
    """Scan inputs of the sign-crossing case on the first four columns: unary outputs, tasks, rank target, class codes, clips."""
    r = np.random.default_rng(seed)
    a, b, c, d = (r.random(n) for _ in range(4))
    y = 0.2 * a**2 / b + np.log(c * 2) * np.sin(d / 3)
    cols = [a, b, c, d]
    funcs = fe._unary_funcs("minimal")
    unaries = fe.OFFSET_UNARIES
    rows = np.arange(n)
    U, clips = fe._scan_inputs(cols, funcs, unaries, rows)
    nu = len(unaries)
    tasks = np.array([(i, j, p, q) for i in range(4) for j in range(i + 1, 4) for p in range(nu) for q in range(nu)], dtype=np.int64)
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
    return U, tasks, fe._rank_scaled(y), codes, int(codes.max()) + 1, clips


def _cpu(U, tasks, yr, codes, ky, clips):
    """Reference scan on the CPU."""
    out_shift = np.empty((len(tasks), 2))
    out_mi = np.empty((len(tasks), 2, 1 + N_BASELINES))
    scan_offset_products(U, tasks, yr, codes, ky, N_BINS, clips, out_shift, out_mi)
    return out_shift, out_mi


@pytest.mark.parametrize("n", [3001, 20000])
def test_gpu_scan_matches_cpu_scan(n):
    """Shifts agree to rounding and the MI tables match, including the odd-n split and the NaN pattern of unfittable tasks."""
    args = _inputs(n)
    cpu_shift, cpu_mi = _cpu(*args)
    res = scan_offset_products_gpu(args[0], args[1], args[2], args[3], args[4], N_BINS, args[5])
    assert res is not None
    gpu_shift, gpu_mi = res
    assert np.array_equal(np.isnan(cpu_shift), np.isnan(gpu_shift))
    assert np.array_equal(np.isnan(cpu_mi), np.isnan(gpu_mi))
    ok = ~np.isnan(cpu_shift[:, 0])
    np.testing.assert_allclose(gpu_shift[ok], cpu_shift[ok], rtol=1e-8, atol=1e-10)
    diff = np.abs(gpu_mi[ok] - cpu_mi[ok])
    assert np.quantile(diff, 0.99) < 1e-9
    assert diff.max() < 5e-3, "a rare bin-boundary flip may move one MI value slightly, never by a visible amount"


def test_gpu_scan_declines_a_target_with_too_many_classes():
    """Six joint histograms of a wide target would not fit in shared memory: the function returns None so the caller uses the CPU scan."""
    args = _inputs(2000)
    assert scan_offset_products_gpu(args[0], args[1], args[2], args[3], 200, N_BINS, args[5]) is None


def test_strict_resident_fit_selects_the_same_columns_as_the_cpu_path(monkeypatch):
    """Selection equivalence: with the strict-resident switch on the stage keeps the same offset products (names to 3 digits) as the CPU scan."""
    r = np.random.default_rng(0)
    n = 30000
    a, b, c, d, e, f = (r.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    _, cpu_names, _, _ = fe.hybrid_offset_product_fe(X, y, top_k=10)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    calls = []
    import mlframe.feature_selection.filters._offset_product_gpu as gpu_mod

    orig = gpu_mod.scan_offset_products_gpu

    def spy(*a, **k):
        """Record that the device scan ran."""
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(gpu_mod, "scan_offset_products_gpu", spy)
    _, gpu_names, _, _ = fe.hybrid_offset_product_fe(X, y, top_k=10)
    assert calls, "the strict-resident path did not reach the device scan"
    assert gpu_names == cpu_names


def test_device_built_inputs_match_host_built_inputs():
    """Unary outputs, fills and winsorisation bounds made on the device give the same scan as the host-built block, including a column with non-positive values (frozen log anchor)."""
    from mlframe.feature_selection.filters._offset_product_gpu import scan_offset_products_device

    r = np.random.default_rng(3)
    n = 20000
    cols = [r.random(n), r.random(n) - 0.2, r.random(n), r.random(n)]
    cols[1][::997] = 0.0
    y = 0.2 * cols[0] ** 2 + np.sin(cols[3] / 3) * (cols[2] - 0.4)
    funcs = fe._unary_funcs("minimal")
    unaries = fe.OFFSET_UNARIES
    rows = np.linspace(0, n - 1, 9001).astype(np.int64)
    anchors = [fe._log_anchor(x) for x in cols]
    U, clips = fe._scan_inputs(cols, funcs, unaries, rows, anchors)
    nu = len(unaries)
    tasks = np.array([(i, j, p, q) for i in range(4) for j in range(i + 1, 4) for p in range(nu) for q in range(nu)], dtype=np.int64)
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    codes = np.asarray(encode_y_for_classif_mi(y[rows]), dtype=np.int64)
    ky = int(codes.max()) + 1
    yr = fe._rank_scaled(y[rows])
    _cpu_shift, cpu_mi = _cpu(U, tasks, yr, codes, ky, clips)
    res = scan_offset_products_device(cols, tuple(unaries), rows, tasks, yr, codes, ky, N_BINS, anchors, fe._WINSOR_Q)
    assert res is not None
    dev_mi, dev_clips = res
    np.testing.assert_allclose(dev_clips, clips, rtol=1e-9, atol=1e-12)
    assert np.array_equal(np.isnan(cpu_mi), np.isnan(dev_mi))
    ok = ~np.isnan(cpu_mi)
    diff = np.abs(dev_mi[ok] - cpu_mi[ok])
    assert np.quantile(diff, 0.99) < 1e-9 and diff.max() < 5e-3


def test_device_inputs_decline_a_unary_without_a_device_form():
    """A unary map outside the device set sends the caller to the host path (None), it is not approximated."""
    from mlframe.feature_selection.filters._offset_product_gpu import scan_offset_products_device

    x = [np.random.default_rng(0).random(500) for _ in range(2)]
    tasks = np.array([[0, 1, 0, 0]], dtype=np.int64)
    assert scan_offset_products_device(x, ("identity", "cbrt"), np.arange(500), tasks, np.zeros(500), np.zeros(500, dtype=np.int64), 1, 10, [0.0, 0.0], (0.01, 0.99)) is None
