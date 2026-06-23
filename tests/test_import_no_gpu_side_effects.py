"""A plain ``import mlframe`` must not probe the GPU: it must not import cupy and must not mutate
``os.environ`` (CUDA_HOME/CUDA_PATH). The CUDA autoconfig + broken-cupy guard are deferred to the
first GPU-dispatch use, so a CPU-only user pays neither the cupy import nor the environment rewrite.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_PROBE = r"""
import os, sys
assert "CUDA_HOME" not in os.environ, "CUDA_HOME unexpectedly preset in clean env"
assert "CUDA_PATH" not in os.environ, "CUDA_PATH unexpectedly preset in clean env"
import mlframe
assert "cupy" not in sys.modules, "import mlframe imported cupy (GPU probe at import time)"
assert "CUDA_HOME" not in os.environ, "import mlframe set CUDA_HOME (env mutation at import time)"
assert "CUDA_PATH" not in os.environ, "import mlframe set CUDA_PATH (env mutation at import time)"
print("CLEAN_IMPORT_OK")
"""


def test_plain_import_does_not_probe_gpu():
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_HOME", "CUDA_PATH")}
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"clean-env import probe failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "CLEAN_IMPORT_OK" in proc.stdout
