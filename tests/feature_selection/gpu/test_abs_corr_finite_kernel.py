"""The one-launch |corr| kernel equals the cupy-operation chain it replaced, case by case, and is faster per call."""

import time

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_abs_corr_gpu as mod
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


def _target(n, seed=1, nan_frac=0.0):
    """A float64 target with an optional share of NaN rows, plus its finiteness mask."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    if nan_frac:
        y[rng.random(n) < nan_frac] = np.nan
    return y, np.isfinite(y)


def _cases():
    """(name, column, target, finite mask) rows: plain, offset, non-finite column and target, odd length, constant, short, correlated."""
    rng = np.random.default_rng(0)
    out = []
    for n in (50_000, 40_001, 9):
        y, f = _target(n)
        for dt in (np.float32, np.float64):
            out.append((f"plain-{n}-{dt.__name__}", rng.standard_normal(n).astype(dt), y, f))
    y, f = _target(30_000)
    out.append(("offset1e7", (1e7 + rng.standard_normal(30_000)).astype(np.float64), y + 1e7, f))
    col = rng.standard_normal(30_000).astype(np.float32)
    col[rng.random(30_000) < 0.1] = np.nan
    col[5] = np.inf
    yn, fn = _target(30_000, nan_frac=0.2)
    out.append(("nonfinite-both", col, yn, fn))
    out.append(("constant", np.full(30_000, 3.0, dtype=np.float32), y, f))
    out.append(("correlated", (2.0 * y + 0.01 * rng.standard_normal(30_000)).astype(np.float32), y, f))
    return out


@pytest.mark.parametrize("name,col,y,fin", _cases(), ids=[c[0] for c in _cases()])
def test_kernel_equals_the_cupy_chain(name, col, y, fin):
    """Same value to 1e-11 relative (the degenerate cases return exactly 0.0 on both sides); the correlated case is near 1, so the check is not vacuous."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    d = cp.asarray(col)
    got = mod.abs_corr_finite_gpu(d, y, fin, 8)
    want = mod._abs_corr_finite_cupy(d, y, fin, 8)
    if want == 0.0:
        assert got == 0.0
    else:
        assert abs(got - want) <= 1e-11 * max(1.0, abs(want)), (got, want)
    if name == "correlated":
        assert want > 0.99


def test_kernel_is_faster_per_call_than_the_chain():
    """Interleaved medians over 32 columns of 40000 rows: the kernel path costs under half of the cupy chain per call."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    rng = np.random.default_rng(3)
    y, f = _target(40_000)
    cols = [cp.asarray(rng.standard_normal(40_000).astype(np.float32)) for _ in range(32)]
    for c in cols[:3]:
        mod.abs_corr_finite_gpu(c, y, f, 8)
        mod._abs_corr_finite_cupy(c, y, f, 8)
    new, old = [], []
    for _ in range(5):
        t = time.perf_counter()
        for c in cols:
            mod.abs_corr_finite_gpu(c, y, f, 8)
        new.append((time.perf_counter() - t) / len(cols))
        t = time.perf_counter()
        for c in cols:
            mod._abs_corr_finite_cupy(c, y, f, 8)
        old.append((time.perf_counter() - t) / len(cols))
    assert float(np.median(new)) < 0.5 * float(np.median(old)), (np.median(new), np.median(old))
