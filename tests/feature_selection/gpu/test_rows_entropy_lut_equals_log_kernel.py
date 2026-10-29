"""The lookup-table row-entropy kernel must equal the double-precision log kernel: same entropy to rounding, identical occupied-cell counts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as bm


def _cuda_ok() -> bool:
    """Whether a CUDA device answers a trivial allocation."""
    try:
        cp.zeros(1)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok(), reason="needs a CUDA device")


def _sparse_counts(rng, k: int, m: int, n: int) -> "cp.ndarray":
    """(k, m) int32 counts of n samples scattered over m cells per row (every row sums to n)."""
    host = np.zeros((k, m), dtype=np.int32)
    for r in range(k):
        np.add.at(host[r], rng.integers(0, m, size=n), 1)
    return cp.asarray(host)


@pytest.mark.parametrize("m", [4000, 4003, 4002])
@pytest.mark.parametrize("n", [500, 1999])
def test_lut_entropy_equals_log_kernel_for_vector_and_scalar_widths(monkeypatch, m, n):
    """M % 4 == 0 uses the int4 kernel, the others the scalar one; both match the log kernel."""
    rng = np.random.default_rng(m + n)
    counts = _sparse_counts(rng, 7, m, n)
    inv_n = 1.0 / n
    monkeypatch.setenv("MLFRAME_ENT_LUT", "0")
    h_log, k_log = bm._rows_entropy_and_k(counts, inv_n)
    monkeypatch.setenv("MLFRAME_ENT_LUT", "1")
    h_lut, k_lut = bm._rows_entropy_and_k(counts, inv_n)
    assert np.allclose(cp.asnumpy(h_lut), cp.asnumpy(h_log), rtol=0.0, atol=1e-12)
    assert np.array_equal(cp.asnumpy(k_lut), cp.asnumpy(k_log))


def test_lut_entropy_handles_rows_that_do_not_sum_to_n(monkeypatch):
    """A masked row (fewer samples than n) keeps the entropy of p = c / n, as the log kernel computes it."""
    rng = np.random.default_rng(5)
    counts = _sparse_counts(rng, 5, 2000, 800)
    inv_n = 1.0 / 1000
    monkeypatch.setenv("MLFRAME_ENT_LUT", "0")
    h_log, _ = bm._rows_entropy_and_k(counts, inv_n)
    monkeypatch.setenv("MLFRAME_ENT_LUT", "1")
    h_lut, _ = bm._rows_entropy_and_k(counts, inv_n)
    assert np.allclose(cp.asnumpy(h_lut), cp.asnumpy(h_log), rtol=0.0, atol=1e-12)


def test_a_count_above_the_table_range_falls_back_to_the_direct_log(monkeypatch):
    """A cell count larger than the stated row count is computed directly, so the result still equals the log kernel."""
    counts = cp.asarray(np.array([[0, 3000, 0, 5]], dtype=np.int32))
    inv_n = 1.0 / 100
    monkeypatch.setenv("MLFRAME_ENT_LUT", "0")
    h_log, _ = bm._rows_entropy_and_k(counts, inv_n)
    monkeypatch.setenv("MLFRAME_ENT_LUT", "1")
    h_lut, _ = bm._rows_entropy_and_k(counts, inv_n)
    assert np.allclose(cp.asnumpy(h_lut), cp.asnumpy(h_log), rtol=0.0, atol=1e-9)
