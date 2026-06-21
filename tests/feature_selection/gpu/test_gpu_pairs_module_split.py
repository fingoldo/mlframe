"""Wave 99 (2026-05-21): split feature_selection/filters/gpu.py
(1106 lines) into gpu.py (now 909 lines) + new _gpu_pairs.py
(231 lines).

Moved to the sibling file: ``mi_direct_gpu_batched_pairs`` plus its
section separator. The original gpu.py re-exports it so existing
``from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs``
imports continue to work.

The sibling lazy-imports the parent module (``gpu``) inside the
function body to resolve kernel symbols (``compute_joint_hist_multi_pair_cuda``
et al.) AFTER they've been populated by ``_ensure_kernels_inited()`` --
the kernel globals start as None at module-load and a static
import would cache the None.
"""
from __future__ import annotations

from pathlib import Path


def test_mi_direct_gpu_batched_pairs_importable_from_facade() -> None:
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs
    assert callable(mi_direct_gpu_batched_pairs)


def test_other_gpu_symbols_still_importable() -> None:
    from mlframe.feature_selection.filters.gpu import (
        mi_direct_gpu_batched,
        mi_direct_gpu_batched_streamed,
        mi_direct_gpu,
        _GpuBufferPool,
        init_kernels,
        _pin_device_if_needed,
        _ensure_kernels_inited,
    )
    for fn in (
        mi_direct_gpu_batched,
        mi_direct_gpu_batched_streamed,
        mi_direct_gpu,
        init_kernels,
        _pin_device_if_needed,
        _ensure_kernels_inited,
    ):
        assert callable(fn), fn
    assert _GpuBufferPool is not None


def test_facade_below_1k_line_threshold() -> None:
    # Anchor on the repo root (the dir whose ``src/mlframe`` exists) rather than a fixed parent count, so the
    # test survives being moved deeper in the tree (the restructure relocated it into feature_selection/gpu/).
    repo_root = next(p for p in Path(__file__).resolve().parents if (p / "src" / "mlframe").is_dir())
    facade = repo_root / "src" / "mlframe" / "feature_selection" / "filters" / "gpu.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"gpu.py is {n} lines, still over the 1k threshold"


def test_sibling_module_owns_the_moved_symbol() -> None:
    """Identity: facade and sibling expose the SAME function object."""
    from mlframe.feature_selection.filters import gpu, _gpu_pairs
    assert gpu.mi_direct_gpu_batched_pairs is _gpu_pairs.mi_direct_gpu_batched_pairs


def test_sibling_raises_clear_cupy_error_when_unavailable() -> None:
    """When CuPy isn't installed, the moved function must surface a
    RuntimeError pointing the user at the install path. CuPy is not
    required on this CI, so we exercise the import-error branch.
    """
    import sys
    import importlib
    import pytest
    import numpy as np
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs

    if "cupy" in sys.modules:
        pytest.skip("CuPy is installed; the import-error branch is unreachable here")
    with pytest.raises(RuntimeError, match="CuPy"):
        mi_direct_gpu_batched_pairs(
            factors_data=np.zeros((4, 2), dtype=np.int32),
            pairs_a=np.array([0], dtype=np.int32),
            pairs_b=np.array([1], dtype=np.int32),
            factors_nbins=np.array([2, 2], dtype=np.int32),
            classes_y=np.zeros(4, dtype=np.int32),
            freqs_y=np.array([2, 2], dtype=np.int32),
        )
