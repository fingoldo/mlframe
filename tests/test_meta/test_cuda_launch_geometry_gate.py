"""The shared ``cuda_launch_geometry`` gate over the whole package: a kernel launch takes its block size from the device, not from a literal tuned on one card.

``pyutilz.performance.gpu_hw_launch`` derives the block (``block_size_for``) and the grid (``launch_1d``) from the device limits and the kernel's own registers and shared memory.
No launch carries a literal block size and there is no baseline: a new one fails. A launch whose geometry the kernel fixes (one warp per item, a shared array sized by the block, a
tile) says so with ``# launch-ok: <reason>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

gate = pytest.importorskip("py_ci_shared.cuda_launch_geometry", reason="the installed py-ci-shared predates the cuda_launch_geometry gate")

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_no_new_kernel_launch_takes_a_literal_block_size():
    """A launch with a literal block size fails; take the size from the device or mark the launch with the reason its geometry is fixed."""
    gate.assert_cuda_launch_geometry(SRC, min_files=500)
