"""The row-tiled column-major materialise (``MLFRAME_FE_MATERIALISE_TILED``) writes the candidate matrix of the one-thread-per-element kernel bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from tests._cuda_require import require_cuda_device


def _operands(n: int, f: int, seed: int) -> np.ndarray:
    """Operand table with zeros, negatives, infinities and NaN so that every op of the chain meets its edge cases."""
    rng = np.random.default_rng(seed)
    tv = rng.standard_normal((n, f)).astype(np.float32) * np.exp(rng.uniform(-6, 6, (n, f))).astype(np.float32)
    tv[::11, ::3] = 0.0
    tv[5::17, 1::4] = np.inf
    tv[7::19, 2::5] = -np.inf
    tv[3::23, ::6] = np.nan
    return tv


@pytest.mark.parametrize("n", [1, 31, 32, 33, 1000, 4097, 30011])
@pytest.mark.parametrize("f,k", [(1, 3), (7, 40), (100, 257), (384, 50)])
def test_tiled_equals_one_thread_per_element(monkeypatch, n, f, k):
    """Row counts below, at and above the 32-row tile, one operand column up to the shared-memory limit (384), all nine ops."""
    require_cuda_device()
    import cupy as cp

    from mlframe.feature_selection.filters import _gpu_resident_materialise as mat

    rng = np.random.default_rng(n + f + k)
    tv = cp.asarray(_operands(n, f, n + f))
    a = rng.integers(0, f, k).astype(np.int64)
    b = rng.integers(0, f, k).astype(np.int64)
    ops = (np.arange(k) % 9).astype(np.int8)
    monkeypatch.setenv("MLFRAME_FE_MATERIALISE_TILED", "1")
    got = cp.asnumpy(mat._fe_materialise_block_gpu(tv, a, b, ops))
    monkeypatch.setenv("MLFRAME_FE_MATERIALISE_TILED", "0")
    want = cp.asnumpy(mat._fe_materialise_block_gpu(tv, a, b, ops))
    assert got.shape == (n, k)
    assert np.array_equal(got, want)


def test_a_wider_operand_table_keeps_the_plain_kernel(monkeypatch):
    """Above 384 operand columns the tile does not fit 48 KB: the plain kernel runs and the result is still the same."""
    require_cuda_device()
    import cupy as cp

    from mlframe.feature_selection.filters import _gpu_resident_materialise as mat

    n, f, k = 500, 400, 30
    rng = np.random.default_rng(1)
    tv = cp.asarray(_operands(n, f, 2))
    a = rng.integers(0, f, k).astype(np.int64)
    b = rng.integers(0, f, k).astype(np.int64)
    ops = (np.arange(k) % 9).astype(np.int8)
    assert f * mat._TILED_ROWS * 4 > mat._TILED_SHARED_LIMIT_BYTES
    monkeypatch.setenv("MLFRAME_FE_MATERIALISE_TILED", "1")
    got = cp.asnumpy(mat._fe_materialise_block_gpu(tv, a, b, ops))
    monkeypatch.setenv("MLFRAME_FE_MATERIALISE_TILED", "0")
    assert np.array_equal(got, cp.asnumpy(mat._fe_materialise_block_gpu(tv, a, b, ops)))


def test_tiled_source_repeats_the_op_chain_of_the_plain_kernel():
    """The op chain (from ``int op`` to the NaN/inf scrub) is the same text in both kernels, up to indentation: an edit to one without the other fails here."""
    from mlframe.feature_selection.filters import _gpu_resident_materialise as mat

    def chain(src: str) -> list:
        """The lines of the op chain with their indentation removed."""
        lines = src.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "int op = (int)ops[k];")
        end = next(i for i, ln in enumerate(lines) if ln.strip() == "if (isnan(v) || isinf(v)) v = 0.0f;")
        return [ln.strip() for ln in lines[start : end + 1]]

    assert chain(mat._FE_MATERIALISE_CM_SRC) == chain(mat._FE_MATERIALISE_CM_TILED_SRC)
