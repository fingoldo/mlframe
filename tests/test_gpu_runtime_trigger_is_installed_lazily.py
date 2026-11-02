"""import mlframe does not import pyutilz.system; the CUDA-probe wrapper appears when gpu_dispatch is imported (tracker 2026-11-02 wave 16, IMP16-6)."""

import subprocess
import sys


def _run(code):
    """Run ``code`` in a fresh interpreter and return the last stdout line."""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=1500)
    assert done.returncode == 0, done.stderr[-500:]
    return done.stdout.split()[-1]


def test_import_mlframe_leaves_pyutilz_system_unimported():
    """It imported pyutilz.system (and numba, about 2.3 s) only to replace one function."""
    assert _run("import sys; import mlframe; print('pyutilz.system' in sys.modules)") == "False"


def test_the_wrapper_is_installed_when_gpu_dispatch_is_imported_afterwards():
    """The deferred GPU autoconfig still hooks the first is_cuda_available() call."""
    code = "import mlframe; from pyutilz.system import gpu_dispatch as g; print(getattr(g.is_cuda_available, '_mlframe_gpu_runtime_wrapped', False))"
    assert _run(code) == "True"


def test_the_wrapper_is_installed_when_gpu_dispatch_was_imported_first():
    """pyutilz imported before mlframe: the wrap happens at once."""
    code = "from pyutilz.system import gpu_dispatch as g; import mlframe; print(getattr(g.is_cuda_available, '_mlframe_gpu_runtime_wrapped', False))"
    assert _run(code) == "True"
