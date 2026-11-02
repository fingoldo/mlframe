"""Importing the discovery correlation modules compiles no numba kernel (tracker 2026-11-02 wave 16, IMP16-4)."""

import subprocess
import sys


def test_no_kernel_signature_exists_right_after_import():
    """The import-time warm-up compiled six parallel kernels (about 1 s) and started numba's thread pool for users who never run composite discovery."""
    code = (
        "from mlframe.training.composite.discovery import _corr_numba as c, _collinear_numba as k\n"
        "print(len(c._abs_corr_all_kernel.signatures), len(k._keep_mask_kernel.signatures), len(k._column_stats_allfinite.signatures))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert done.returncode == 0, done.stderr[-400:]
    assert done.stdout.split()[-3:] == ["0", "0", "0"]
