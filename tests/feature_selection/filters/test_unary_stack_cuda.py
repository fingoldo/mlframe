"""The one-launch unary stack reproduces the per-unary cupy reference bit for bit, in both element types and for both smart-log branches, and the cache returns one buffer per operand."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")

_SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-30, -1e-30, 1e30, -1e30, np.inf, -np.inf, np.nan, 2.5, -2.5, 3.0], dtype=np.float64)


def _columns(dtype):
    """Columns that exercise every unary: a normal one with special values (xmin <= 0) and a strictly positive one (xmin > 0, plain log)."""
    rng = np.random.default_rng(1)
    with_special = np.concatenate([rng.standard_normal(5000), _SPECIAL]).astype(dtype)
    positive = (rng.exponential(2.0, 5000) + 1e-3).astype(dtype)
    return {"special": with_special, "positive": positive}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["special", "positive"])
def test_stack_is_bit_identical_to_the_per_unary_reference(dtype, kind):
    """Every row of the stack equals the matching ``_gpu_apply_unary`` result bit for bit (NaNs compare equal)."""
    from mlframe.feature_selection.filters import _usability_njit_pool as pool
    from mlframe.feature_selection.filters._unary_stack_cuda import unary_stack_cuda

    pool._cp = cp
    x = _columns(dtype)[kind]
    xmin = float(np.nanmin(x))
    d_x = cp.asarray(x)
    codes = list(range(17))
    got = unary_stack_cuda(d_x, cp.asarray(codes, dtype=cp.int32), xmin)
    assert got is not None and got.shape == (17, x.shape[0]) and got.dtype == d_x.dtype
    for k, code in enumerate(codes):
        ref = cp.asnumpy(pool._gpu_apply_unary(d_x, code, xmin))
        out = cp.asnumpy(got[k])
        word = np.uint32 if dtype == np.float32 else np.uint64
        same = (ref.view(word) == out.view(word)) | (np.isnan(ref) & np.isnan(out))
        assert same.all(), f"unary code {code} differs at {int(np.flatnonzero(~same)[0])}"


def test_non_float_or_non_vector_input_is_declined():
    """An integer column or a 2-D array is not handled by the kernel, so the caller keeps its own path."""
    from mlframe.feature_selection.filters._unary_stack_cuda import unary_stack_cuda

    codes = cp.asarray([0, 1], dtype=cp.int32)
    assert unary_stack_cuda(cp.arange(10), codes, 0.0) is None
    assert unary_stack_cuda(cp.zeros((3, 3), dtype=cp.float32), codes, 0.0) is None


def test_cache_returns_the_same_buffer_for_a_repeated_operand_and_respects_the_budget():
    """An operand that recurs gets its cached stack back; a budget smaller than one stack is never cached; the oldest entry is evicted first."""
    from mlframe.feature_selection.filters._unary_stack_cuda import UnaryStackCache

    codes = cp.asarray([0, 3, 5], dtype=cp.int32)
    a = cp.asarray(np.linspace(1, 2, 100, dtype=np.float32))
    b = cp.asarray(np.linspace(2, 3, 100, dtype=np.float32))
    one = 3 * 100 * 4
    cache = UnaryStackCache(budget_bytes=one)
    sa = cache.get(a, codes, 1.0)
    assert cache.get(a, codes, 1.0) is sa
    cache.get(b, codes, 2.0)
    assert cache.get(a, codes, 1.0) is not sa  # a was evicted to fit b
    tiny = UnaryStackCache(budget_bytes=one - 1)
    s1 = tiny.get(a, codes, 1.0)
    assert tiny.get(a, codes, 1.0) is not s1  # too big to cache: recomputed each time
