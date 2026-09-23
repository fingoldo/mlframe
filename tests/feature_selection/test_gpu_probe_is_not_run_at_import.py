"""Importing the package must not compile a CUDA kernel or import cupy: both build a CUDA context on any host."""

import os
import subprocess
import sys
from pathlib import Path

import mlframe

_SRC = str(Path(mlframe.__file__).resolve().parents[1])


def _run(code: str):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([_SRC, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    # A fresh process each time: "did importing this initialise CUDA" cannot be asked once it already has.
    return subprocess.run([sys.executable, "-W", "always", "-c", code], capture_output=True, text=True, env=env)


def test_importing_feature_selection_neither_probes_cuda_nor_imports_cupy():
    r = _run("import sys, mlframe.feature_selection; print('cupy' in sys.modules)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", "cupy was imported at package-import time"
    assert "NumbaPerformanceWarning" not in r.stderr, f"a kernel was compiled and launched at import:\n{r.stderr}"


def test_the_availability_accessors_still_answer():
    r = _run(
        "from mlframe.feature_selection.filters._batch_mi_noise_gate_kernels import cuda_available, cupy_available\n"
        "print(isinstance(cuda_available(), bool), isinstance(cupy_available(), bool))"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True True"


def test_the_legacy_flag_names_still_read_through_the_accessors():
    r = _run(
        "import mlframe.feature_selection.filters._batch_mi_noise_gate_kernels as k\n"
        "import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as g\n"
        "print(k._CUDA_AVAIL == k.cuda_available(), k._CUPY_AVAIL == k.cupy_available(), g._CUDA_AVAIL == k.cuda_available())"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True True True"
