"""``import mlframe.models`` must not need PyWavelets: it is the optional ``signal`` extra, used only by the wavelet feature emitter."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_models_package_works_without_pywavelets():
    """The nightly benchmark installs no extras; a module-level ``import pywt`` made ``mlframe.models`` unimportable there."""
    lines = [
        "import sys",
        "sys.modules['pywt'] = None",  # makes `import pywt` raise ImportError, as on a machine without the package
        "import mlframe.models",
        "import mlframe.feature_engineering._timeseries_emit as emit",
        "assert callable(emit._emit_wavelets)",
    ]
    result = subprocess.run([sys.executable, "-c", chr(10).join(lines)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
