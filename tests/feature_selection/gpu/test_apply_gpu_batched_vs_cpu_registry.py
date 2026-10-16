"""The GPU batched unary/binary appliers must agree with the CPU transform registries for every GPU-compatible name."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.feature_engineering import (
    apply_gpu_binary_batched,
    apply_gpu_unary_batched,
    create_binary_transformations,
    create_unary_transformations,
    gpu_compatible_binary_names,
    gpu_compatible_unary_names,
)

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() >= 1
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")

_UNARY = sorted(gpu_compatible_unary_names() & set(create_unary_transformations("maximal")))
_BINARY = sorted(gpu_compatible_binary_names() & set(create_binary_transformations("maximal")))


def _columns() -> np.ndarray:
    """Positive, bounded, away-from-zero columns so every unary op is inside its domain on both backends."""
    rng = np.random.default_rng(11)
    return rng.uniform(0.3, 2.5, size=(200, 3))


def test_registries_overlap_the_gpu_name_sets():
    """Non-vacuity: most GPU-compatible names exist in the CPU registry, so the parametrised checks below run."""
    assert len(_UNARY) >= 15 and len(_BINARY) >= 6


@pytest.mark.parametrize("name", _UNARY)
def test_unary_gpu_matches_cpu(name):
    """Column-wise CPU registry output equals the batched GPU output within float32 tolerance."""
    cols = _columns()
    gpu = cp.asnumpy(apply_gpu_unary_batched(cols, [0, 1, 2], name)).astype(np.float64)
    cpu_fn = create_unary_transformations("maximal")[name]
    cpu = np.column_stack([np.asarray(cpu_fn(np.ascontiguousarray(cols[:, j]))) for j in range(3)])
    np.testing.assert_allclose(gpu, cpu, rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("name", _BINARY)
def test_binary_gpu_matches_cpu(name):
    """Element-wise CPU registry output equals the GPU output within float32 tolerance."""
    cols = _columns().astype(np.float32)
    a, b = cols[:, 0], cols[:, 1]
    gpu = cp.asnumpy(apply_gpu_binary_batched(cp.asarray(a), cp.asarray(b), name)).astype(np.float64)
    cpu = np.asarray(create_binary_transformations("maximal")[name](a, b), dtype=np.float64)
    np.testing.assert_allclose(gpu, cpu, rtol=2e-4, atol=1e-5)


def test_non_gpu_name_is_rejected():
    """Negative control: a name outside the compatible set raises instead of silently returning something."""
    with pytest.raises(ValueError):
        apply_gpu_unary_batched(_columns(), [0], "definitely_not_a_transform")
    with pytest.raises(ValueError):
        apply_gpu_binary_batched(cp.ones(3), cp.ones(3), "definitely_not_a_transform")
