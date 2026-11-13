"""CUDA kernels read global memory coalesced, or carry a reviewed ``// coalescing-ok: <reason>``: py-ci-shared's ``cuda_coalesced_reads`` gate over the whole package.

The detector (a thread-derived row times a stride, also through a hoisted offset or in a kernel with no loop; a thread-derived run of a run table) and its tests live in py-ci-shared. What this package keeps is the
decisions: a kernel whose strided read stays says why in a comment inside the kernel (a measured reject, a read once per bin, a kept version with a documented win axis), so the reason travels with the code and
there is no baseline file. A new strided read fails the test until it is fixed or reviewed the same way.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.cuda_coalesced_reads import assert_cuda_coalesced_reads

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_no_kernel_reads_global_memory_with_an_unreviewed_stride():
    """A kernel that strides a row-major matrix by its thread index, or walks a thread-derived run, fails unless it is fixed (a column-major copy, staged rows) or marked ``// coalescing-ok:`` with its reason."""
    assert_cuda_coalesced_reads(SRC_ROOT, exclude=("_benchmarks",), min_files=500)
