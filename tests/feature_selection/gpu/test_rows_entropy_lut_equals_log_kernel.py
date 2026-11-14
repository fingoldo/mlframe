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


_SPLIT_M = 1 << 20  # the split path needs 2^20 or more cells per row and fewer than 48 rows


@pytest.mark.parametrize("k", [3, 11])
def test_split_lut_entropy_equals_the_split_log_kernel_and_the_lut_path_runs(monkeypatch, k):
    """Narrow chunk of huge rows (the permutation-null case): the table form of the split kernel equals the double precision split kernel, and it is the one that ran."""
    rng = np.random.default_rng(k)
    n = 20000
    counts = _sparse_counts(rng, k, _SPLIT_M, n)
    inv_n = 1.0 / n
    monkeypatch.setenv("MLFRAME_ENT_LUT", "0")
    h_log, k_log = bm._rows_entropy_and_k(counts, inv_n)
    monkeypatch.setenv("MLFRAME_ENT_LUT", "1")
    calls = []
    from mlframe.feature_selection.filters import _fe_batched_mi_cmi_hist as owner

    real = bm._rows_entropy_lut_split
    monkeypatch.setattr(owner, "_rows_entropy_lut_split", lambda *a, **kw: calls.append(1) or real(*a, **kw))  # the caller lives in the carved histogram module
    h_lut, k_lut = bm._rows_entropy_and_k(counts, inv_n)
    assert calls, "the table form of the split kernel did not run"
    assert np.allclose(cp.asnumpy(h_lut), cp.asnumpy(h_log), rtol=0.0, atol=1e-12)
    assert np.array_equal(cp.asnumpy(k_lut), cp.asnumpy(k_log))


def test_split_lut_entropy_handles_masked_rows_and_counts_above_the_table(monkeypatch):
    """A row with fewer samples than n, and a cell count above the table range, still equal the double precision split kernel."""
    rng = np.random.default_rng(9)
    host = np.zeros((4, _SPLIT_M), dtype=np.int32)
    for r, m in enumerate((20000, 15000, 20000, 20000)):
        np.add.at(host[r], rng.integers(0, _SPLIT_M, size=m), 1)
    host[3, 7] = 30000  # above the stated row count of 20000: the direct log
    counts = cp.asarray(host)
    inv_n = 1.0 / 20000
    monkeypatch.setenv("MLFRAME_ENT_LUT", "0")
    h_log, k_log = bm._rows_entropy_and_k(counts, inv_n)
    monkeypatch.setenv("MLFRAME_ENT_LUT", "1")
    h_lut, k_lut = bm._rows_entropy_and_k(counts, inv_n)
    assert np.allclose(cp.asnumpy(h_lut), cp.asnumpy(h_log), rtol=0.0, atol=1e-12)
    assert np.array_equal(cp.asnumpy(k_lut), cp.asnumpy(k_log))


def test_a_row_count_that_is_not_a_whole_number_keeps_the_double_precision_split_kernel(monkeypatch):
    """inv_n that is not 1/integer: the table form declines (returns None) and the double precision split kernel answers."""
    counts = _sparse_counts(np.random.default_rng(2), 2, _SPLIT_M, 5000)
    monkeypatch.setenv("MLFRAME_ENT_LUT", "1")
    assert bm._rows_entropy_lut_split(cp, counts, 1.0 / 5000.5, 4) is None
    h, k = bm._rows_entropy_and_k(counts, 1.0 / 5000.5)
    assert h.shape == (2,) and k.shape == (2,)
