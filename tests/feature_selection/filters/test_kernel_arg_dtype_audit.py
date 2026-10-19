"""The launch-time kernel argument audit finds a float32 array bound to a ``double*`` parameter, and finds nothing in a real strict-GPU fit."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")

_SRC_ROOT = str(Path(__file__).resolve().parents[3] / "src")

_BAD_LAUNCH = r"""
import cupy as cp
import numpy as np
from mlframe.feature_selection._benchmarks import kernel_arg_dtype_audit as audit

audit.install()
kernel = cp.RawKernel('extern "C" __global__ void scale(const double* x, double* out, const long long n) { long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) out[i] = 2.0 * x[i]; }', "scale")
x = cp.arange(8, dtype=cp.float32)
out = cp.zeros(8, dtype=cp.float64)
kernel((1,), (8,), (x, out, np.int64(8)))
raise SystemExit(1 if audit.report() else 0)
"""


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run a python child with the package source on its path and return the finished process."""
    env = dict(os.environ, PYTHONPATH=_SRC_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONUNBUFFERED="1")
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=900)


def test_audit_reports_a_float32_array_bound_to_a_double_pointer():
    """A launch that binds a float32 array to ``const double*`` is reported as a mismatch and the child exits with code 1."""
    done = _run("-c", _BAD_LAUNCH)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "MISMATCH kernel=scale arg#0: expects float64*, got a float32 array" in done.stdout


def test_audit_finds_no_mismatch_in_a_small_strict_gpu_fit():
    """The quick built-in strict-GPU fit launches only kernels whose array arguments have the declared element types."""
    done = _run("-m", "mlframe.feature_selection._benchmarks.kernel_arg_dtype_audit", "--small")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1500:]
    assert "ARG_DTYPE_MISMATCHES 0" in done.stdout
