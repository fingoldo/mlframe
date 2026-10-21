"""A CUDA kernel source must not hard-code ``double*`` for a data column without saying why.

The criterion dtype is float32 by default, so a buffer in that dtype bound to a ``const double*`` parameter reads garbage. A kernel that takes data should be templated on ``{REAL}``; a kernel
whose ``double*`` is a statistic, coefficient or partial sum keeps it, and the file records that with an ``Intentional float64`` comment at the cast or the source.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_HARD_DOUBLE_PTR = re.compile(r"const\s+double\s*\*")
_MARKER = "Intentional float64"


def _kernel_files():
    """Source files outside the benchmarks that declare a ``const double*`` kernel parameter."""
    out = []
    for path in SRC.rglob("*.py"):
        if "_benchmarks" in path.parts:
            continue
        if _HARD_DOUBLE_PTR.search(path.read_text(encoding="utf-8", errors="replace")):
            out.append(path)
    return out


def test_kernel_files_were_found():
    """The scan sees the known kernel modules, so an empty result cannot pass the gate by accident."""
    names = {p.name for p in _kernel_files()}
    assert {"_fe_batched_mi.py", "_cell_moments_kernel.py", "_hermite_fe_mi.py"} <= names


def test_every_hard_double_kernel_file_states_why():
    """Each file with a ``const double*`` kernel parameter carries an ``Intentional float64`` reason."""
    missing = [str(p.relative_to(SRC)) for p in _kernel_files() if _MARKER not in p.read_text(encoding="utf-8", errors="replace")]
    assert not missing, f"hard double* kernel parameter without an '{_MARKER}' reason (template it on {{REAL}} or state why): {missing}"
