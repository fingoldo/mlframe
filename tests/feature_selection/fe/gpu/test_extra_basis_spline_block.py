"""The fused cubic B-spline launch equals the cupy Cox-de Boor chain value for value (float32 and double, columns on every knot interval, NaN, rows on the boundaries); the extra-basis scorer scores name blocks to the single-matrix MI and follows the kernel-tuning entry."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _extra_basis_resident as eb


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    return cp


@pytest.mark.parametrize("real", [np.float32, np.float64])
def test_spline_block_equals_the_cupy_chain(real, monkeypatch):
    """Every spline column of the sweep pool, plus rows exactly on the lowest and highest knot and a NaN row."""
    cp = _cp()
    X, eng, meta, _y, _nb = eb._sweep_inputs({"n_samples": 4000})
    monkeypatch.setattr(eb, "crit_float_dtype", lambda: np.dtype(real))  # the element type of the build and of the chain
    monkeypatch.setattr(eb, "_crit_np_dtype", lambda: np.dtype(real))
    names = [c for c in eng.columns if meta[c]["basis"] == "spline"]
    assert names
    xs = []
    for c in names:
        col = X[meta[c]["src"]].to_numpy().astype(real).copy()
        col[0], col[1], col[2] = meta[c]["lo"], meta[c]["hi"], np.nan
        xs.append(cp.asarray(col))
    blk = cp.asnumpy(eb._bspline_block_gpu(cp, xs, [meta[c] for c in names]))
    for j, c in enumerate(names):
        mm = meta[c]
        sp = float(mm["hi"]) - float(mm["lo"])
        sp = sp if sp > 1e-12 else 1e-12
        z = cp.clip((xs[j] - float(mm["lo"])) / sp, 0.0, 1.0)
        ref = cp.asnumpy(eb._bspline_col_gpu(cp, z, mm["knots"], int(mm["idx"]), degree=3))
        np.testing.assert_array_equal(np.isnan(blk[j]), np.isnan(ref), err_msg=c)
        np.testing.assert_array_equal(blk[j][~np.isnan(ref)], ref[~np.isnan(ref)], err_msg=c)


def test_name_blocks_give_the_single_matrix_mi(monkeypatch):
    """A room of a few columns forces many name blocks; the MI equals the one-block run exactly."""
    cp = _cp()
    from mlframe.feature_selection.filters import _device_sort_safe as dss

    X, eng, meta, y, nb = eb._sweep_inputs({"n_samples": 3000})
    names = list(eng.columns)
    whole = eb._mi_by_name_blocks(cp, X, names, meta, y, nb)
    monkeypatch.setattr(dss, "device_room", lambda cp_: 4 * 3 * 3 * 3000 * 4)
    blocked = eb._mi_by_name_blocks(cp, X, names, meta, y, nb)
    assert whole.shape == blocked.shape == (len(names),)
    np.testing.assert_array_equal(whole, blocked)


def test_scorer_follows_the_tuner(monkeypatch):
    """With the host chosen the device scorer answers None (the caller's host scorer runs); with the device chosen it returns the MI vector."""
    _cp()
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    X, eng, meta, y, nb = eb._sweep_inputs({"n_samples": 3000})
    monkeypatch.setattr(eb, "extra_basis_device_chosen", lambda n: False)
    assert eb.extra_basis_eng_mi_resident(X, eng, y, meta, nbins=nb) is None
    monkeypatch.setattr(eb, "extra_basis_device_chosen", lambda n: True)
    got = eb.extra_basis_eng_mi_resident(X, eng, y, meta, nbins=nb)
    assert got is not None and got.shape == (eng.shape[1],)
