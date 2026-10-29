"""Regression: ``_popcount64`` must be callable as plain Python when numba JIT is disabled (the coverage nightly sets NUMBA_DISABLE_JIT=1).

Before the fix the public name was a numba intrinsic, and calling it from Python raised ``NotImplementedError``.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mlframe.feature_selection.filters._family_maxt_null_v2_njit import _popcount64

_PROBE = textwrap.dedent(
    """
    import numpy as np
    from mlframe.feature_selection.filters._family_maxt_null_v2_njit import _popcount64, column_bin_bitsets
    assert _popcount64(np.uint64(0b1011)) == 3
    assert _popcount64(np.uint64(0)) == 0
    assert _popcount64(np.uint64(2**64 - 1)) == 64
    print("ok")
    """
)


def test_popcount64_counts_bits_when_called_from_python():
    """The public popcount returns the number of set bits for uint64 inputs."""
    assert _popcount64(np.uint64(0b1011)) == 3
    assert _popcount64(np.uint64(2**64 - 1)) == 64


@pytest.mark.parametrize("disable_jit", ["0", "1"])
def test_popcount64_importable_and_callable_under_jit_setting(disable_jit):
    """A fresh interpreter can call ``_popcount64`` with JIT on and with NUMBA_DISABLE_JIT=1."""
    env = dict(os.environ, NUMBA_DISABLE_JIT=disable_jit)
    proc = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")
